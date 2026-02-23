# TFM - Análisis de la Fragilización de las Vasijas Nucleares mediante PINNs

# 1. Introducción

## 1.1. Resumen


## 1.2. Objetivos
Se busca construir una MLP+PINN que se pueda utilizar profesionalmente, y que 
tenga mejoría respecto a la ecuación actualmente utilizada. Para ello, este
proyecto se divide en varias pequeñas etapas:

- 0. Ecuación ASTM como referencia precedente a este proyecto.
- 1. XGBoost optimizado como baseline a comparar.
- 2. MLP Inicial.
- 3. MLP + PINN Básica (Usando ecuación ASTM).
- 4. MLP + PINN. Inclusión de término asociado a monotonicidad (castigar si el
    daño baja al subir la fluencia).
- 5. MLP + PINN. Inclusión de residuo mecanístico (castigar si la curva se
    desvía de la ley de la raíz cuadrada de la fluencia).
- 6. Generación de modelo "producto" basándose en los resultados obtenidos en
    las anteriores etapas del proyecto.

## 1.2. Datos Utilizados



# 2. Diseño del Repositorio

### docs

Carpeta que almacena la memoria asociada al TFM.

### models

Carpeta que almacena los modelos creados y entrenados.

### notebooks

Carpeta que contiene diversos notebooks con funciones de prueba y borradores

### src

Carpeta que almacena formalmente las funciones utilizadas en la realización del TFM.

### setup.py

Script creado para poder usar el paquete "casero" source de forma formal.

### Instalación

Crear una env que cumpla los requerimientos, personalmente recomiendo usar mamba. Después, escribir en terminal ```pip install -e .```. El argumento ```-e``` crea un enlace simbólico dentro de la carpeta de librerías de Python que apunta a la carpeta src.

## 2.1 Paquete ```src```

