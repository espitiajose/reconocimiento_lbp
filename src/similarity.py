"""Similitud coseno entre vectores descriptores.

La similitud coseno mide el ángulo entre dos vectores, ignorando su magnitud:

        cos(theta) = (A . B) / (||A|| * ||B||)

Vale 1 cuando los vectores apuntan en la misma dirección (muy parecidos) y
tiende a 0 cuando son muy distintos. Es la medida que vimos en clase y la que
usamos para comparar los histogramas LBP de dos rostros.

Se implementa de forma explícita (producto punto y normas con numpy); no se usa
ninguna función de "similitud" de librerías de machine learning.
"""

import numpy as np


def cosine_similarity(a, b):
    """Calcula la similitud coseno entre dos vectores.

    Args:
        a, b: vectores (numpy.ndarray 1D) de la misma longitud.

    Returns:
        float entre 0 y 1 (nuestros histogramas no tienen valores negativos).
        Devuelve 0.0 si alguno de los vectores es nulo, para evitar dividir por
        cero.
    """
    producto_punto = float(np.dot(a, b))
    norma_a = float(np.linalg.norm(a))
    norma_b = float(np.linalg.norm(b))

    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0

    return producto_punto / (norma_a * norma_b)
