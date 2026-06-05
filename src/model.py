"""Modelo de reconocimiento: vector representativo por persona + clasificación.

Estrategia (siguiendo el plan del profesor, adaptado a LBP):

    - Para cada persona se promedian los embeddings de sus imágenes de
      entrenamiento, obteniendo un único "vector representativo" U.
    - Para clasificar una imagen nueva se calcula su embedding V y se compara
      con U1, U2, U3 por similitud coseno. Gana la persona con mayor similitud.
    - Si la mejor similitud no supera un umbral, se devuelve "Desconocido"
      (la cara probablemente no pertenece al dataset).

Este es un clasificador propio (vecino-medio por coseno); no se usa ninguna
librería de clasificación.
"""

import numpy as np

from . import features
from . import lbp as lbp_mod
from .image_loader import preparar_imagen
from .similarity import cosine_similarity

# Umbral por defecto para decidir "Desconocido". Es configurable; conviene
# ajustarlo viendo las similitudes reales que produce el dataset.
UMBRAL_DESCONOCIDO = 0.85

# Etiqueta que se devuelve cuando ninguna similitud supera el umbral.
ETIQUETA_DESCONOCIDO = "Desconocido"


def embedding_desde_ruta(ruta):
    """Atajo: ruta de imagen -> embedding LBP (vector de 9216 valores)."""
    return features.extraer_embedding(preparar_imagen(ruta))


def embedding_y_lbp_desde_ruta(ruta):
    """Ruta de imagen -> (embedding, codigos_lbp) sin computar LBP dos veces.

    Returns:
        (embedding, codigos): embedding ndarray 9216, codigos ndarray 2D uint8.
    """
    gris = preparar_imagen(ruta)
    codigos = lbp_mod.imagen_lbp(gris)
    embedding = features.extraer_embedding_desde_codigos(codigos)
    return embedding, codigos


def construir_vectores_representativos(rutas_por_persona):
    """Construye el vector representativo U de cada persona.

    Args:
        rutas_por_persona: dict {persona: [rutas de entrenamiento]}.

    Returns:
        dict {persona: U} donde U es un numpy.ndarray de longitud 9216 (la media
        de los embeddings de entrenamiento de esa persona).
    """
    vectores_u = {}
    for persona, rutas in rutas_por_persona.items():
        embeddings = [embedding_desde_ruta(ruta) for ruta in rutas]
        # Media elemento a elemento (eje 0) de todos los embeddings de la persona.
        vectores_u[persona] = np.mean(np.array(embeddings), axis=0)
    return vectores_u


def comparar(embedding_v, vectores_u):
    """Similitud coseno del embedding V contra cada vector representativo.

    Args:
        embedding_v: embedding de la imagen a clasificar.
        vectores_u: dict {persona: U}.

    Returns:
        dict {persona: similitud} (float por persona).
    """
    return {
        persona: cosine_similarity(u, embedding_v)
        for persona, u in vectores_u.items()
    }


def clasificar(embedding_v, vectores_u, umbral=UMBRAL_DESCONOCIDO):
    """Decide a qué persona corresponde un embedding.

    Args:
        embedding_v: embedding de la imagen a clasificar.
        vectores_u: dict {persona: U}.
        umbral: similitud mínima para aceptar una identificación.

    Returns:
        (etiqueta, similitudes):
            etiqueta: nombre de la persona ganadora, o "Desconocido".
            similitudes: dict {persona: similitud} para inspección.
    """
    similitudes = comparar(embedding_v, vectores_u)

    # Persona con mayor similitud. Usar items() evita problemas de tipos
    # porque key recibe directamente el valor (float) en lugar de Optional[float].
    mejor_persona, mejor_valor = max(similitudes.items(), key=lambda item: item[1])

    if mejor_valor < umbral:
        return ETIQUETA_DESCONOCIDO, similitudes
    return mejor_persona, similitudes
