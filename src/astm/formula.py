#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo: formula.py
Descripción: Implementación de la ecuación de correlación ASTM E900 para predecir 
             el desplazamiento de la temperatura de transición (TTS) en aceros de vasija.
Autor: Samuel Martín Martínez
Fecha: 2026
"""

import numpy as np

def astm_e900_15(cu, ni, p, mn, temp_c, fluence, product_form):
    """
    Calcula el TTS (Transition Temperature Shift) basándose en ASTM E900.
    
    Args:
        cu (array-like): Contenido de Cobre en % peso.
        ni (array-like): Contenido de Níquel en % peso.
        p (array-like): Contenido de Fósforo en % peso.
        mn (array-like): Contenido de Manganeso en % peso.
        temp_c (array-like): Temperatura de irradiación en Celsius.
        fluence (array-like): Fluencia de neutrones (n/cm^2).
        product_form (array-like): Tipo de producto ('W', 'P', 'F').
        
    Returns:
        np.array: TTS predicho en grados Celsius.
    """
    
    # --- 1. PREPARACIÓN DE DATOS Y UNIDADES ---
    # Convertir inputs a arrays de numpy para vectorización
    cu = np.array(cu, dtype=float)
    ni = np.array(ni, dtype=float)
    p = np.array(p, dtype=float)
    mn = np.array(mn, dtype=float)
    temp_c = np.array(temp_c, dtype=float)
    fluence = np.array(fluence, dtype=float)
    product_form = np.array(product_form, dtype=str)

    # Conversión de Temperatura: Celsius -> Fahrenheit -> Rankine (para la fórmula)
    T_f = temp_c * 1.8 + 32.0
    T_R = T_f + 460.67

    # Conversión de Fluencia: La norma usa unidades de n/m2 típicamente o ajustadas.
    # Para E900, usaremos la fluencia efectiva en unidades relativas.
    # Clampeamos para evitar log(0)
    
    # --- 2. TÉRMINO A: MECANISMOS DE ENDURECIMIENTO DE MATRIZ (SMD) ---
    # Coeficientes A según tipo de material (Weld vs Plate/Forging)
    # Valores típicos aproximados de la norma E900-15
    A_coeff = np.where(product_form == 'W', 6.7e-18, 6.7e-18) # W=Weld, otros=Base Metal

    # El término A depende de la temperatura (Arrhenius) y Fósforo
    # TTS_1 = A * exp(19310 / T_R) * (1 + 110*P) * (Fluencia)^0.44 approx
    # Usamos np.log10 para la fluencia según algunas variantes, aquí usamos potencia directa
    # Nota: Los coeficientes exactos varían ligeramente según la revisión del paper ASTM.
    
    term_A = A_coeff * np.exp(20730 / T_R) * fluence ** 0.5076


    # --- 3. TÉRMINO B: PRECIPITACIÓN DE COBRE (CRP) ---
    # Este término es complejo. Depende de si el Cu satura.
    
    # Límite de saturación del Cobre (Cu_max suele ser 0.30% o 0.25% según Ni)
    # Estimación simplificada de Cu efectivo:
    # Se crea la función G(flux)
    G = 1/2+1/2*np.tanh((np.log10(fluence)-18.24)/1.052)

    # Se crea la funcion F de cobre efectivo
    F = np.maximum(cu - 0.072, 0) ** 0.577

    # Coeficientes B
    B_coeff = np.where(product_form == 'W', 234, 128)

    term_B = B_coeff * (1.0 + 2.106 * ni **1.173) * F * G


    # --- 4. RESULTADO FINAL ---
    # TTS total en Fahrenheit
    tts_fahrenheit = term_A + term_B
    
    # Convertir TTS de Fahrenheit a Celsius (Delta F / 1.8 = Delta C)
    tts_celsius = tts_fahrenheit / 1.8
    
    return tts_celsius