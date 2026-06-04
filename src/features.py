"""Construcción del descriptor (embedding) a partir de la imagen LBP.

Idea (método clásico de Ahonen et al. para rostros con LBP):

    1. Se calcula la imagen LBP.
    2. Se divide en un grid de 6x6 = 36 celdas.
    3. En cada celda se construye un histograma de 256 bins (cuántas veces
       aparece cada código LBP de 0 a 255).
    4. Se concatenan los 36 histogramas en un solo vector.

Así el vector final guarda CÓMO es la textura y, gracias al grid, también DÓNDE
está (la textura de la zona de los ojos queda separada de la de la boca, etc.).

Longitud del vector: 36 celdas * 256 bins = 9216 valores.
"""

import numpy as np

from . import lbp

# Parámetros del grid sobre la imagen de 120x120.
NUM_CELDAS_LADO = 6          # grid 6x6
BINS_HISTOGRAMA = 256        # un bin por cada posible código LBP (0-255)
LONGITUD_EMBEDDING = NUM_CELDAS_LADO * NUM_CELDAS_LADO * BINS_HISTOGRAMA  # 9216


def histograma_celda(imagen_codigos, fila0, col0, alto, ancho):
    """Histograma normalizado de los códigos LBP de una celda rectangular.

    Args:
        imagen_codigos: imagen de códigos LBP (numpy.ndarray 2D).
        fila0, col0: esquina superior-izquierda de la celda.
        alto, ancho: tamaño de la celda en píxeles.

    Returns:
        numpy.ndarray de 256 valores float que suman 1 (frecuencia relativa).
        Se normaliza para que las celdas sean comparables aunque tengan distinta
        cantidad de píxeles (las celdas del borde pierden algunos al recortar
        el marco en la imagen LBP).
    """
    celda = imagen_codigos[fila0:fila0 + alto, col0:col0 + ancho]
    # np.bincount cuenta ocurrencias de cada valor; minlength=256 garantiza que
    # el histograma siempre tenga los 256 bins aunque algún código no aparezca.
    conteos = np.bincount(celda.ravel(), minlength=BINS_HISTOGRAMA).astype(np.float64)
    total = conteos.sum()
    if total > 0:
        conteos /= total  # frecuencia relativa (suma 1)
    return conteos


def extraer_embedding(matriz_gris):
    """Calcula el vector descriptor (embedding) de una imagen de grises 120x120.

    Args:
        matriz_gris: imagen de grises ya redimensionada (numpy.ndarray 2D),
            típicamente de 120x120 producida por image_loader.preparar_imagen.

    Returns:
        numpy.ndarray de longitud 9216 (36 histogramas de 256 bins concatenados).
    """
    codigos = lbp.imagen_lbp(matriz_gris)
    filas, columnas = codigos.shape

    # Tamaño de cada celda. Usamos división entera; con 118x118 y 6 celdas cada
    # celda mide 19 px y la última absorbe el resto, para no dejar píxeles fuera.
    alto_celda = filas // NUM_CELDAS_LADO
    ancho_celda = columnas // NUM_CELDAS_LADO

    histogramas = []
    for fila_celda in range(NUM_CELDAS_LADO):
        for col_celda in range(NUM_CELDAS_LADO):
            fila0 = fila_celda * alto_celda
            col0 = col_celda * ancho_celda

            # La última celda de cada eje se estira hasta el final para incluir
            # las filas/columnas sobrantes de la división entera.
            alto = alto_celda if fila_celda < NUM_CELDAS_LADO - 1 else filas - fila0
            ancho = ancho_celda if col_celda < NUM_CELDAS_LADO - 1 else columnas - col0

            histogramas.append(histograma_celda(codigos, fila0, col0, alto, ancho))

    # Concatenar los 36 histogramas en un único vector de 9216 valores.
    return np.concatenate(histogramas)
