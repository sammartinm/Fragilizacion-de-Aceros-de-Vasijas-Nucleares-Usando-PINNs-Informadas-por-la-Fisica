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

    # Conversiones de unidades
    fluence = fluence * 1e4

    # Se calcula la primera parte de la fórmula
    condiciones_A = [
        product_form == 'W',  # ¿Es Soldadura?
        product_form == 'F'   # ¿Es Forja?
    ]
    valores_A = [
        0.919,  # Valor si es Soldadura
        1.011   # Valor si es Forja
    ]
    A_coeff = np.select(condiciones_A, valores_A, default=1.080)

    TTS1 = A_coeff * 5/9 * 1.8943e-12 * fluence **(0.5695) * \
        ((1.8*temp_c+32)/550)**(-5.47) * (0.09 + p/0.012)**0.216 * \
        (1.66+(ni**8.54/0.63))**0.39 * (mn/1.36)**0.3 
    
    # Se calcula la segunda parte de la fórmula
    condiciones = [
        product_form == 'W',  # ¿Es Soldadura?
        product_form == 'F'   # ¿Es Forja?
    ]
    valores = [
        0.968,  # Valor si es Soldadura
        0.738   # Valor si es Forja
    ]
    B_coeff = np.select(condiciones, valores, default=0.819)

    M = B_coeff * np.maximum( np.minimum(113.87*(np.log(fluence) - \
        np.log(4.5e+20)), 612.6), 0) * ((1.8*temp_c+32)/550)**(-5.45) * \
        (0.1+p/0.012)**(-0.098) * (0.168+(ni**0.58)/0.63)**0.73
    
    TTS2 = 5/9 * M * np.maximum(np.minimum(cu,0.28)-0.053,0)

    return TTS1 + TTS2