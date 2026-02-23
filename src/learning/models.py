#coding utf-8
"""Modelos de aprendizaje
    
Este módulo contiene las clases asociadas a los modelos de aprendizaje
utilizados durante la realización de este proyecto de fin de master.
"""

import os
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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