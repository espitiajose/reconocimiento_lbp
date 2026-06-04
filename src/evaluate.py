"""Evaluación del sistema: accuracy y matriz de confusión.

Procedimiento (Fase 10 del plan):
    1. Listar las imágenes y partir 70/30 estratificado por persona.
    2. Construir U1, U2, U3 SOLO con el 70% de entrenamiento.
    3. Clasificar cada imagen del 30% de prueba.
    4. Reportar accuracy (aciertos/total) y la matriz de confusión 3x3.

En la evaluación NO se aplica el umbral de "Desconocido": todas las imágenes de
prueba pertenecen al dataset, así que medimos la identificación pura quedándonos
siempre con la persona más parecida.

Uso:
    python3 -m src.evaluate
"""

from . import dataset
from .model import comparar, construir_vectores_representativos, embedding_desde_ruta


def evaluar(fraccion_train=0.7):
    """Ejecuta la evaluación y muestra los resultados por consola.

    Returns:
        (accuracy, matriz_confusion, personas) por si se quiere reutilizar.
    """
    rutas_por_persona = dataset.listar_por_persona()
    if not rutas_por_persona:
        print("No se encontraron imágenes en user1/, user2/, user3/.")
        print("Coloca las fotos recortadas antes de evaluar.")
        return None

    train, test = dataset.split_estratificado(rutas_por_persona, fraccion_train)
    vectores_u = construir_vectores_representativos(train)

    personas = list(vectores_u.keys())
    indice = {persona: i for i, persona in enumerate(personas)}

    # matriz[real][predicho] = número de imágenes.
    matriz = [[0] * len(personas) for _ in personas]
    aciertos = 0
    total = 0

    print("=== Detalle de clasificación (conjunto de prueba) ===")
    for persona_real, rutas in test.items():
        for ruta in rutas:
            embedding = embedding_desde_ruta(ruta)
            similitudes = comparar(embedding, vectores_u)
            predicho = max(similitudes, key=similitudes.get)

            matriz[indice[persona_real]][indice[predicho]] += 1
            total += 1
            ok = predicho == persona_real
            if ok:
                aciertos += 1

            marca = "OK " if ok else "XX "
            print(f"  {marca} real={persona_real} pred={predicho} "
                  f"sim={similitudes[predicho]:.4f}")

    accuracy = aciertos / total if total else 0.0
    _imprimir_resumen(accuracy, aciertos, total, matriz, personas)
    return accuracy, matriz, personas


def _imprimir_resumen(accuracy, aciertos, total, matriz, personas):
    """Imprime accuracy y la matriz de confusión con encabezados."""
    print("\n=== Resultados ===")
    print(f"Accuracy: {accuracy:.4f}  ({aciertos}/{total} aciertos)")

    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    encabezado = "        " + "".join(f"{p:>8}" for p in personas)
    print(encabezado)
    for i, persona in enumerate(personas):
        fila = "".join(f"{valor:>8}" for valor in matriz[i])
        print(f"{persona:>8}{fila}")


if __name__ == "__main__":
    evaluar()
