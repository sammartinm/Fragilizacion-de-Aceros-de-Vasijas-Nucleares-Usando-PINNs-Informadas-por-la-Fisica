"""
Archivo: main.py
Descripción: Script principal para ejecutar el pipeline del TFM.
             1. Carga datos.
             2. Calcula error de la física (ASTM E900).
             3. (Futuro) Entrena la red neuronal.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# --- Truco para importar desde src ---
# Añadimos la carpeta raíz al path de Python para encontrar 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.astm.formula import astm_e900_15

def main():
    print("--- INICIO DEL PROYECTO TFM: Fragilización de Vasijas ---")
    
    # 1. CARGAR DATOS
    ruta_csv = 'data/df_plotter_cm2.csv'
    if not os.path.exists(ruta_csv):
        print(f"ERROR: No encuentro el archivo en {ruta_csv}")
        return

    print(f"Cargando datos desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    
    # Filtro rápido: Eliminar filas con nulos si las hay
    df = df.dropna(subset=['DT41J_Celsius', 'Fluence_n_cm2', 'Cu', 'Ni', 'Temperature_Celsius'])
    
    # 2. CALCULAR BASELINE (FÍSICA)
    print("Calculando predicciones usando norma ASTM E900-15...")
    
    # Importante: Asegurar que Product_Form son strings ('W', 'P', 'F')
    # Si en tu CSV vienen como 'Forging', etc., habría que mapearlos.
    # Viendo tu CSV, parece que ya vienen como 'F', 'W', 'P'.
    
    # Mn: Si tu dataset NO tiene columna Mn, usamos un valor estándar (ej. 1.40)
    if 'Mn' in df.columns:
        mn_values = df['Mn'].values
    else:
        print("AVISO: Columna 'Mn' no encontrada. Usando promedio 1.40%")
        mn_values = np.full(len(df), 1.40)

    try:
        preds_fisica = astm_e900_15(
            cu=df['Cu'].values,
            ni=df['Ni'].values,
            p=df['P'].values,
            mn=mn_values,
            temp_c=df['Temperature_Celsius'].values,
            fluence=df['Fluence_n_cm2'].values,
            product_form=df['Product_Form'].values
        )
        
        # Guardamos la predicción en el dataframe por si queremos verla luego
        df['Pred_ASTM'] = preds_fisica
        
        # 3. CALCULAR ERROR (RMSE)
        rmse = np.sqrt(mean_squared_error(df['DT41J_Celsius'], df['Pred_ASTM']))
        
        print("-" * 30)
        print(f"RESULTADO BASELINE (FÍSICA PURA):")
        print(f"RMSE: {rmse:.4f} °C")
        print("-" * 30)
        
        if rmse < 15.0:
            print(">> ¡ÉXITO! La fórmula física funciona correctamente (Error < 15°C).")
            print(">> Tu objetivo con la Red Neuronal será bajar de este número.")
        else:
            print(">> OJO: El error es alto. Verifica las unidades (¿Fahrenheit vs Celsius?)")
            
    except Exception as e:
        print(f"\nERROR CRÍTICO AL CALCULAR FÓRMULA:\n{e}")

if __name__ == "__main__":
    main()