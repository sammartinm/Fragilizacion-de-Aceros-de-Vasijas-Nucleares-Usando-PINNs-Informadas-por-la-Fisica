from src.config import DATA_DIR
import pandas as pd

def cargar_dataset():
    """Carga el dataset desde un archivo CSV y lo devuelve como un DataFrame de 
    pandas."""
    df = pd.read_csv(DATA_DIR / "df_plotter_cm2.csv")

    df = df.dropna(subset=['DT41J_Celsius', 'Fluence_n_cm2', 'Cu', 'Ni', 
                           'Temperature_Celsius'])
    return df