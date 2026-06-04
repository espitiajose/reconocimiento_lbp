"""Prueba rápida de la canalización (sin necesitar el dataset real).

Genera imágenes sintéticas en memoria y verifica las propiedades matemáticas
clave del pipeline:

    - El embedding tiene longitud 9216 (36 celdas x 256 bins).
    - Cada histograma de celda suma ~1 (está normalizado).
    - La similitud coseno de un vector consigo mismo es 1.0.
    - Dos imágenes muy distintas dan similitud menor que dos imágenes iguales.

Uso:
    python3 test_pipeline.py
"""

import numpy as np

from src.features import LONGITUD_EMBEDDING, NUM_CELDAS_LADO, extraer_embedding
from src.similarity import cosine_similarity


def _imagen_constante(valor, tamano=120):
    """Imagen gris uniforme (todos los píxeles iguales)."""
    return np.full((tamano, tamano), valor, dtype=np.uint8)


def _imagen_ruido(tamano=120):
    """Imagen con un patrón determinista tipo tablero de gradientes."""
    fila = np.arange(tamano, dtype=np.uint8)
    return np.outer(fila, fila).astype(np.uint8)


def main():
    embedding_a = extraer_embedding(_imagen_constante(100))
    embedding_b = extraer_embedding(_imagen_ruido())

    # 1) Longitud del embedding.
    assert embedding_a.shape[0] == LONGITUD_EMBEDDING, "Longitud de embedding inesperada"
    print(f"[OK] Longitud del embedding = {embedding_a.shape[0]} (esperado {LONGITUD_EMBEDDING})")

    # 2) Cada histograma de celda suma ~1 (36 celdas de 256 bins).
    celdas = embedding_a.reshape(NUM_CELDAS_LADO * NUM_CELDAS_LADO, 256)
    sumas = celdas.sum(axis=1)
    assert np.allclose(sumas, 1.0), "Algún histograma de celda no está normalizado"
    print(f"[OK] Los {celdas.shape[0]} histogramas de celda suman 1.0")

    # 3) Similitud coseno de un vector consigo mismo = 1.0.
    auto = cosine_similarity(embedding_b, embedding_b)
    assert abs(auto - 1.0) < 1e-9, "cos(v, v) debería ser 1.0"
    print(f"[OK] cosine_similarity(v, v) = {auto:.6f}")

    # 4) Coherencia: cos(v, v) >= cos(v, w) para imágenes distintas.
    cruzada = cosine_similarity(embedding_a, embedding_b)
    assert auto >= cruzada, "La auto-similitud debería ser la mayor"
    print(f"[OK] cos(v, w) entre imágenes distintas = {cruzada:.6f} (<= 1.0)")

    print("\nTodas las verificaciones pasaron.")


if __name__ == "__main__":
    main()
