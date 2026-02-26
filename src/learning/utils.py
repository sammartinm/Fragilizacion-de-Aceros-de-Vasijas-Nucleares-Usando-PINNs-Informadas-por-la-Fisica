import os
import xgboost as xgb
import torch

def save_xgboost_model(model, model_name, folder='../models/baselines'):
    """
    Guarda un modelo XGBoost en formato JSON nativo.
    
    Args:
        model: El objeto XGBRegressor ya entrenado.
        model_name: Nombre del archivo (ej: 'xgb_v1.json').
        folder: Carpeta donde se almacenará.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Aseguramos la extensión .json para compatibilidad nativa
    if not model_name.endswith('.json'):
        model_name += '.json'
        
    filepath = os.path.join(folder, model_name)
    model.save_model(filepath)
    print(f"Modelo XGBoost guardado en: {filepath}")

def load_xgboost_model(filepath):
    """
    Carga un modelo XGBoost desde un archivo JSON.
    
    Returns:
        model: El modelo cargado listo para predecir.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el modelo en {filepath}")
        
    # Instanciamos un objeto vacío del mismo tipo (Regresor)
    model = xgb.XGBRegressor()
    model.load_model(filepath)
    print(f"Modelo cargado correctamente desde {filepath}")
    return model
