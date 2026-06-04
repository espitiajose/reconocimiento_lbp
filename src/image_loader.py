"""Carga y preprocesado de imágenes.

Responsabilidad única: tomar la ruta de una foto (ya recortada al rostro por el
estudiante) y devolver una matriz de grises de 120x120 lista para el cálculo LBP.

Decisión del proyecto: la apertura del archivo, la conversión a escala de grises
y el redimensionado se delegan a Pillow (Pillow es una librería de procesamiento
de imágenes de propósito general, no una librería de LBP ni un clasificador, así
que su uso está permitido por el enunciado). Todo lo que viene DESPUÉS (LBP,
histogramas, comparación) se implementa a mano.
"""

import numpy as np
from PIL import Image

# Tamaño al que se normalizan todas las imágenes. 120 divide exacto en un grid
# 6x6 (celdas de 20x20 px), que es lo que usaremos para los histogramas LBP.
TAMANO_POR_DEFECTO = 120


def preparar_imagen(ruta, tamano=TAMANO_POR_DEFECTO):
    """Carga una imagen y la deja lista para el pipeline LBP.

    Pasos:
        1. Abrir el archivo con Pillow.
        2. Convertir a escala de grises ("L" = luminancia de 8 bits, 0-255).
        3. Redimensionar a `tamano` x `tamano` (por defecto 120x120).

    Args:
        ruta: ruta al archivo de imagen (.jpg, .png, etc.).
        tamano: lado del cuadrado de salida en píxeles.

    Returns:
        numpy.ndarray 2D de tipo uint8 y forma (tamano, tamano), con valores
        de intensidad de 0 a 255.
    """
    # Image.open es perezoso; convert("L") fuerza la decodificación y nos da
    # un solo canal de gris. resize usa por defecto un remuestreo bicúbico de
    # buena calidad, suficiente para nuestro caso.
    imagen = Image.open(ruta).convert("L").resize((tamano, tamano))

    # np.asarray nos entrega la matriz 2D de intensidades sin copiar de más.
    return np.asarray(imagen, dtype=np.uint8)
