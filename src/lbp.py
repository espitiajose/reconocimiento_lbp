"""Cálculo de Local Binary Patterns (LBP) implementado a mano.

¿Qué es LBP? Para cada píxel "central" se miran sus 8 vecinos. Si el vecino es
mayor o igual que el central se anota un 1, si es menor un 0. Recorriendo los 8
vecinos en un orden fijo se forma un número binario de 8 bits, que convertido a
decimal da un código entre 0 y 255. Ese código describe la "textura" local
(bordes, esquinas, zonas planas) alrededor del píxel, y es bastante robusto a
cambios de iluminación.

IMPORTANTE: aquí NO se usa skimage.feature.local_binary_pattern ni ninguna otra
implementación de librería. La lógica es propia, como pide el enunciado.

Orden de los 8 vecinos (horario, empezando arriba-izquierda). El bit 7 es el más
significativo y el bit 0 el menos significativo:

        bit7  bit6  bit5
        bit0   C    bit4
        bit1  bit2  bit3
"""

import numpy as np

# Desplazamientos (df, dc) de los 8 vecinos en el MISMO orden que los bits,
# del bit 7 (peso 128) al bit 0 (peso 1). Mantener este orden fijo es clave para
# que todas las imágenes generen códigos comparables.
VECINOS = [
    (-1, -1),  # bit 7  (arriba-izquierda)
    (-1, 0),  # bit 6  (arriba)
    (-1, 1),  # bit 5  (arriba-derecha)
    (0, 1),  # bit 4  (derecha)
    (1, 1),  # bit 3  (abajo-derecha)
    (1, 0),  # bit 2  (abajo)
    (1, -1),  # bit 1  (abajo-izquierda)
    (0, -1),  # bit 0  (izquierda)
]


def codigo_lbp(matriz, i, j):
    """Calcula el código LBP (0-255) del píxel central (i, j).

    Args:
        matriz: imagen de grises (numpy.ndarray 2D).
        i, j: fila y columna del píxel central. Debe ser un píxel interior
            (1 <= i <= filas-2 y 1 <= j <= columnas-2) para que existan los
            8 vecinos.

    Returns:
        Entero entre 0 y 255 con el patrón binario local.
    """
    centro = matriz[i, j]
    codigo = 0
    # Recorremos los vecinos del bit más significativo (peso 128) al menos
    # significativo (peso 1).
    for posicion, (df, dc) in enumerate(VECINOS):
        peso = 1 << (7 - posicion)  # 128, 64, 32, ... , 1
        vecino = matriz[i + df, j + dc]
        if vecino >= centro:
            codigo += peso
    return codigo


def imagen_lbp(matriz):
    """Convierte una imagen de grises en su imagen de códigos LBP.

    Los píxeles del borde (primera/última fila y columna) se omiten porque no
    tienen los 8 vecinos completos. Por eso la salida tiene 2 filas y 2 columnas
    menos que la entrada (p.ej. de 120x120 pasa a 118x118).

    Args:
        matriz: imagen de grises (numpy.ndarray 2D).

    Returns:
        numpy.ndarray 2D de uint8 con los códigos LBP de los píxeles interiores.
    """
    filas, columnas = matriz.shape
    salida = np.zeros((filas - 2, columnas - 2), dtype=np.uint8)
    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            # (i-1, j-1) porque la salida no incluye el borde.
            salida[i - 1, j - 1] = codigo_lbp(matriz, i, j)
    return salida
