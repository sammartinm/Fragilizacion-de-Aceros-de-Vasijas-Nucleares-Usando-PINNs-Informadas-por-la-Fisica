"""Módulo para la gestión y preprocesamiento del dataset de radiación.
Modulo: dataset.py
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler


class RadiationDataset(Dataset):
  """Clase Dataset de PyTorch para datos de fragilización por radiación.

  Esta clase gestiona la carga de datos desde un archivo CSV, permite la
  división en subconjuntos y ofrece un método para aplicar preprocesado a
  todo el conjunto de datos basándose en el entrenamiento.
  """

  def __init__(self, csv_path):
    """
    Inicializa el dataset cargando el archivo CSV.

    Args:
      csv_path: Cadena con la ruta al archivo CSV del dataset.
    """
    self.data = pd.read_csv(csv_path)
    #self.data['Fluence_n_cm2'] = np.log10(self.data['Fluence_n_cm2'])

    # Convertimos categorías a códigos numéricos para permitir cálculos.
    if 'Product_Form' in self.data.columns:
      self.data['Product_Form'] = (
          self.data['Product_Form'].astype('category').cat.codes
      )

  def __len__(self):
    """Devuelve la cantidad total de registros en el dataset."""
    return len(self.data)

  def __getitem__(self, idx):
    """Devuelve un registro del dataset convertido a tensor.

    Args:
      idx: Índice de la muestra a obtener.

    Returns:
      Un tensor de PyTorch con los datos de la fila solicitada.
    """
    # Se obtienen los valores de la fila y se transforman a tensor de PyTorch.
    sample = self.data.iloc[idx].values
    return torch.tensor(sample, dtype=torch.float32)

  def data_split(self, test_factor, val_factor=None):
    indices = list(range(len(self)))
      
    if val_factor:
      train_val_idx, test_idx = model_selection.train_test_split(
        indices, test_size=test_factor, random_state=42
        )
      adj_val_factor = val_factor / (1 - test_factor)
      train_idx, val_idx = model_selection.train_test_split(
        train_val_idx, test_size=adj_val_factor, random_state=42
        )
      return Subset(self, train_idx), Subset(self, val_idx), Subset(self, 
                                                                    test_idx)

      # Si no hay val_factor, devolvemos None en el medio
    train_idx, test_idx = model_selection.train_test_split(
      indices, test_size=test_factor, random_state=42
    )
    return Subset(self, train_idx), None, Subset(self, test_idx)

  def preprocess(self, train_set, preprocessor=StandardScaler()):
    """Entrena un preprocesador con el conjunto de train y escala todo el 
    dataset.

    Este método debe ser llamado manualmente por el usuario tras crear el 
    objeto. Al modificar el DataFrame interno 'self.data', todos los objetos 
    Subset creados anteriormente se verán afectados automáticamente.

    Args:
      train_set: El objeto Subset que representa los datos de entrenamiento.
      preprocessor: Objeto con métodos fit/transform (ej. StandardScaler).
    """
    # Extraemos los índices del subconjunto de entrenamiento.
    self.preprocessor = preprocessor
    train_indices = train_set.indices
    train_data = self.data.iloc[train_indices]

    # Entrenamos el preprocesador con los datos de entrenamiento.
    self.preprocessor.fit(train_data)

    # Transformamos el dataset completo (esto devuelve un numpy array de 
    # floats).
    transformed_values = self.preprocessor.transform(self.data)
    
    # En lugar de usar iloc[:, :], reconstruimos el DataFrame.
    # Esto evita el conflicto de tipos (int8 vs float64).
    self.data = pd.DataFrame(
        transformed_values, 
        columns=self.data.columns, 
        index=self.data.index
    )
    

  def inverse_transform_y(self, y_scaled, target_col='DT41J_Celsius', 
                          preprocessor=None):
    """Convierte predicciones normalizadas de vuelta a su escala original 
    (Celsius)."""
    if preprocessor is None:
        raise ValueError("Se requiere el preprocesador entrenado.")
      
    # Buscamos la posición de la columna objetivo
    target_idx = list(self.data.columns).index(target_col)
    mean_y = preprocessor.mean_[target_idx]
    std_y = preprocessor.scale_[target_idx]
      
    return (y_scaled * std_y) + mean_y
  
  def despreprocess(self, y_test, y_scaled):
        """Convierte predicciones normalizadas de vuelta a su escala original 
        (Celsius)."""

        target_col_name = 'DT41J_Celsius'
        preds_real = self.inverse_transform_y(y_scaled, target_col=target_col_name, preprocessor=self.preprocessor)
        y_test_real = self.inverse_transform_y(y_test, target_col=target_col_name, preprocessor=self.preprocessor)
        
        return preds_real, y_test_real
    


def subset_to_numpy(subset, target_col='DT41J_Celsius'):
    """
    Convierte un Subset de PyTorch a arrays de NumPy (X, y).
    Busca el índice real de la columna objetivo.
    """
    # Buscamos el índice de la columna objetivo en el dataframe original
    target_idx = list(subset.dataset.data.columns).index(target_col)
    
    data = np.array([subset[i].numpy() for i in range(len(subset))])
    
    # Extraemos X e Y correctamente
    X = np.delete(data, target_idx, axis=1)
    y = data[:, target_idx]
    
    return X, y

