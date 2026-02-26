#coding utf-8
"""Modelos de aprendizaje
    
Este módulo contiene las clases asociadas a los modelos de aprendizaje
utilizados durante la realización de este proyecto de fin de master.
"""

import os
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn

class BaselineXGBoost:
    """Clase para entrenar y evaluar un modelo XGBoost como línea base."""

    def __init__(self, **kwargs):
        """
        Inicializa el modelo XGBoost.
        Puedes pasar cualquier hiperparámetro aceptado por xgb.XGBRegressor.
        """
        # Definimos unos parámetros por defecto razonables
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Actualizamos los valores por defecto con los que el usuario pase al 
        # instanciar
        default_params.update(kwargs)
        
        # Instanciamos el modelo con los parámetros finales
        self.model = xgb.XGBRegressor(**default_params)

    def fit(self, X_train, y_train):
        """Entrena el modelo XGBoost."""
        print("Entrenando baseline XGBoost...")
        self.model.fit(X_train, y_train)
        print("Entrenamiento completado.")

    def predict(self, X):
        """Realiza predicciones."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, dataset=None, preprocessor=None):
        """Evalúa el modelo y devuelve las métricas clave.
        Si se proporciona el dataset y el preprocesador, desnormaliza 
        las variables para obtener el error en grados Celsius reales.
        """
        preds = self.predict(X_test)
        
        # --- NUEVO: Desnormalización ---
        if dataset is not None and preprocessor is not None:
            # Desnormalizamos tanto las predicciones como los valores reales
            preds = dataset.inverse_transform_y(preds, preprocessor=preprocessor)
            y_test = dataset.inverse_transform_y(y_test, preprocessor=preprocessor)
            print("\n(Métricas calculadas sobre la escala original en °C)")
        else:
            print("\n(Métricas calculadas sobre datos normalizados)")
        # -------------------------------
        
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print("\n--- Resultados XGBoost Baseline ---")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print("-----------------------------------")
        
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    
    def save_model(self, model_name, folder='../models'):
        """
        Método interno para guardar el modelo sin necesidad de funciones externas.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filepath = os.path.join(folder, model_name)
        # Accedemos al modelo real de la librería para guardar
        self.model.save_model(filepath)
        print(f"Modelo guardado en: {filepath}")

class MLPInicial(nn.Module):
    """Clase para un perceptrón multicapa inicial."""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MLPInicial, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class PINNEmbrittlement(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=3, activation=nn.SiLU, preprocessor=None, feature_names=None):
        """
        Red Neuronal PINN que predice el residual de la fórmula ASTM E900.
        
        Args:
            input_dim: Número de características de entrada.
            hidden_dim: Neuronas en capas ocultas.
            num_layers: Número total de capas (incluyendo la de salida).
            activation: Función de activación de PyTorch (ej. nn.SiLU, nn.Tanh).
            preprocessor: Objeto StandardScaler entrenado.
            feature_names: Lista de nombres de columnas.
        """
        super(PINNEmbrittlement, self).__init__()
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        # Construcción dinámica de la red neuronal multicapa
        layers = []
        current_dim = input_dim
        
        # Añadimos las capas ocultas
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim
            
        # Añadimos la capa de salida lineal
        layers.append(nn.Linear(current_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def astm_e900_pytorch(self, x_real):
        """
        Traducción de tu formula.py a operaciones de PyTorch.
        Recibe tensores en escala REAL (no normalizados).
        """
        prod_form = x_real[:, 0]
        cu = x_real[:, 1]
        ni = x_real[:, 2]
        mn = x_real[:, 3]
        p = x_real[:, 4]
        temp_c = x_real[:, 5]
        fluence = x_real[:, 6] * 1e4 
        
        A_coeff = torch.where(prod_form == 2.0, 0.919, 
                  torch.where(prod_form == 0.0, 1.011, 1.080))
        
        B_coeff = torch.where(prod_form == 2.0, 0.968, 
                  torch.where(prod_form == 0.0, 0.738, 0.819))

        term1 = A_coeff * (5/9) * 1.8943e-12 * (fluence ** 0.5695)
        term2 = ((1.8 * temp_c + 32) / 550) ** -5.47
        term3 = (0.09 + p / 0.012) ** 0.216
        term4 = (1.66 + (ni ** 8.54) / 0.63) ** 0.39
        term5 = (mn / 1.36) ** 0.3
        TTS1 = term1 * term2 * term3 * term4 * term5
        
        ln_phi = torch.log(fluence)
        ln_phi_ref = torch.log(torch.tensor(4.5e20, device=x_real.device))
        
        m_term1 = torch.clamp(113.87 * (ln_phi - ln_phi_ref), min=0.0, max=612.6)
        m_term2 = ((1.8 * temp_c + 32) / 550) ** -5.45
        m_term3 = (0.1 + p / 0.012) ** -0.098
        m_term4 = (0.168 + (ni ** 0.58) / 0.63) ** 0.73
        
        M = B_coeff * m_term1 * m_term2 * m_term3 * m_term4
        cu_term = torch.clamp(torch.clamp(cu, max=0.28) - 0.053, min=0.0)
        
        TTS2 = (5/9) * M * cu_term
        
        return (TTS1 + TTS2).view(-1, 1)

    def unscale_inputs(self, x_scaled):
        """Desnormaliza los inputs usando el preprocesador del dataset."""
        if self.preprocessor is None:
            raise ValueError("¡Error! El modelo PINN necesita un 'preprocessor' entrenado.")
        dummy_y = torch.zeros((x_scaled.shape[0], 1), device=x_scaled.device)
        target_idx = 6 
        
        full_tensor = torch.cat([x_scaled[:, :target_idx], dummy_y, x_scaled[:, target_idx:]], dim=1)
        means = torch.tensor(self.preprocessor.mean_, dtype=torch.float32, device=x_scaled.device)
        scales = torch.tensor(self.preprocessor.scale_, dtype=torch.float32, device=x_scaled.device)
        
        x_real = (full_tensor * scales) + means
        return torch.cat([x_real[:, :target_idx], x_real[:, target_idx+1:]], dim=1)

    def forward(self, x):
        """Paso hacia adelante."""
        nn_residual = self.mlp(x)
        x_real = self.unscale_inputs(x)
        tts_astm_real = self.astm_e900_pytorch(x_real)
        
        target_idx = 6 
        mean_y = self.preprocessor.mean_[target_idx]
        scale_y = self.preprocessor.scale_[target_idx]
        tts_astm_scaled = (tts_astm_real - mean_y) / scale_y
        
        return tts_astm_scaled + nn_residual

def pinn_loss(model, x_batch, y_batch, fluence_idx=6, lambda_data=1.0, lambda_mono=0.5):
    """
    Calcula la pérdida conjunta para entrenar la PINN.
    
    Args:
        model: Instancia de PINNEmbrittlement.
        x_batch: Tensor de características normalizadas (requires_grad=True).
        y_batch: Tensor objetivo normalizado.
        fluence_idx: Índice de la columna de fluencia en x_batch (por defecto 6).
    """
    # 1. Pérdida de datos (MSE normal)
    y_pred = model(x_batch)
    loss_data = nn.functional.mse_loss(y_pred, y_batch.view(-1, 1))
    
    # 2. Pérdida de Monotonía (Gradiente de TTS respecto a Fluencia >= 0)
    # Autograd para obtener la derivada respecto a las entradas
    gradients = torch.autograd.grad(
        outputs=y_pred, 
        inputs=x_batch,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    grad_fluence = gradients[:, fluence_idx]
    
    # Penalizamos si el gradiente es negativo
    loss_mono = torch.mean(torch.nn.functional.relu(-grad_fluence)**2)
    
    # 3. Pérdida de Colocación (Physics Loss)
    # Como la arquitectura ya incluye tts_astm en el forward (Residual), 
    # forzar la red a aprender la física aquí es redundante si no tienes 
    # puntos de colocación extra. Si los tienes, se haría aquí.
    
    total_loss = lambda_data * loss_data + lambda_mono * loss_mono
    
    return total_loss, loss_data, loss_mono

