"""Descubrimiento del dataset y partición estratificada 70/30.

Estructura esperada en la raíz del proyecto:

    reconocimiento_lbp/
    ├── user1/  (10 imágenes)
    ├── user2/  (10 imágenes)
    └── user3/  (10 imágenes)

"Estratificada por persona" significa que el 70/30 se aplica DENTRO de cada
persona, para que entrenamiento y prueba tengan a las tres por igual.
"""

import os

# Carpeta raíz del proyecto = carpeta que contiene a src/ (un nivel arriba de
# este archivo). Así los scripts funcionan sin importar desde dónde se ejecuten.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Extensiones de imagen aceptadas.
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")

# Nombres de las carpetas de personas, en orden.
PERSONAS = ("user1", "user2", "user3")


def listar_por_persona(raiz=RAIZ_PROYECTO):
    """Lista las rutas de imágenes de cada persona.

    Args:
        raiz: carpeta donde están user1/, user2/, user3/.

    Returns:
        dict {nombre_persona: [rutas ordenadas]}. Solo incluye personas cuya
        carpeta existe y contiene al menos una imagen.
    """
    rutas_por_persona = {}
    for persona in PERSONAS:
        carpeta = os.path.join(raiz, persona)
        if not os.path.isdir(carpeta):
            continue
        # Orden alfabético del nombre de archivo -> resultados reproducibles.
        archivos = sorted(
            nombre for nombre in os.listdir(carpeta)
            if nombre.lower().endswith(EXTENSIONES)
        )
        rutas = [os.path.join(carpeta, nombre) for nombre in archivos]
        if rutas:
            rutas_por_persona[persona] = rutas
    return rutas_por_persona


def split_estratificado(rutas_por_persona, fraccion_train=0.7):
    """Divide las imágenes de cada persona en entrenamiento y prueba.

    El corte se hace por nombre de archivo ya ordenado (determinista): las
    primeras `fraccion_train` van a entrenamiento y el resto a prueba. Con 10
    imágenes y 0.7 quedan 7 de entrenamiento y 3 de prueba por persona.

    Args:
        rutas_por_persona: dict como el que devuelve listar_por_persona.
        fraccion_train: proporción para entrenamiento (0 a 1).

    Returns:
        (train, test): dos dicts {persona: [rutas]}.
    """
    train = {}
    test = {}
    for persona, rutas in rutas_por_persona.items():
        corte = int(round(len(rutas) * fraccion_train))
        train[persona] = rutas[:corte]
        test[persona] = rutas[corte:]
    return train, test
