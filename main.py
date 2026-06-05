"""Programa interactivo: clasifica una imagen ingresada por el usuario.

Flujo:
    1. Entrena con TODAS las imágenes de user1/, user2/, user3/ (construye U1, U2, U3).
    2. Pide por teclado la ruta de una imagen a evaluar.
    3. Extrae su embedding LBP, lo compara por similitud coseno con cada U y
       muestra la persona predicha (o "Desconocido") junto con las similitudes.
    4. Muestra ventana matplotlib con imagen original, imagen LBP segmentada,
       plano 2D de vectores e histogramas comparados.

Uso:
    python main.py
"""

import os

from src import dataset, visualize
from src.model import (
    UMBRAL_DESCONOCIDO,
    clasificar,
    construir_vectores_representativos,
    embedding_y_lbp_desde_ruta,
)


def main():
    # --- Entrenamiento: vector representativo por persona ---
    rutas_por_persona = dataset.listar_por_persona()
    if not rutas_por_persona:
        print("No se encontraron imágenes en user1/, user2/, user3/.")
        print("Coloca al menos una foto recortada en cada carpeta y reintenta.")
        raise FileNotFoundError("No se encontraron imágenes de entrenamiento.")

    print(
        "Entrenando con las imágenes de:",
        ", ".join(f"{p} ({len(r)} img)" for p, r in rutas_por_persona.items()),
    )
    vectores_u = construir_vectores_representativos(rutas_por_persona)

    # --- Imagen a evaluar ---
    ruta = input("\nRuta de la imagen a evaluar: ").strip()
    if not os.path.isfile(ruta):
        print(f"No existe el archivo: {ruta}")
        return

    embedding, codigos_lbp = embedding_y_lbp_desde_ruta(ruta)
    etiqueta, similitudes = clasificar(embedding, vectores_u, umbral=UMBRAL_DESCONOCIDO)

    # --- Resultado en consola ---
    print("\n=== Resultado ===")
    print(f"Predicción: {etiqueta}")
    visualize.imprimir_similitudes_conseno(similitudes)
    print(f"(umbral para 'Desconocido': {UMBRAL_DESCONOCIDO})")

    label = f"{os.path.basename(ruta)} → {etiqueta}"
    visualize.imprimir_histograma_consola(embedding, label=label)

    # --- Ventana matplotlib unificada ---
    visualize.mostrar_ventana_completa(
        ruta, embedding, vectores_u, similitudes,
        codigos_lbp=codigos_lbp, label_v=label,
    )


if __name__ == "__main__":
    main()
