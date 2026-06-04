# Reconocimiento de rostros con LBP + similitud coseno

Proyecto educativo de visión por computador: identifica a cuál de tres personas
(`user1`, `user2`, `user3`) se parece más una imagen nueva, usando **Local Binary
Patterns (LBP)** y **comparación de histogramas por similitud coseno**.

El LBP, los histogramas y la clasificación están **implementados a mano**. Se usan
librerías (numpy, Pillow, scikit-image) solo para abrir las imágenes, convertirlas
a gris y redimensionarlas — **no** se usa ninguna librería que implemente LBP ni
clasificadores.

## ¿Cómo funciona? (resumen)

```text
foto recortada
   → abrir + gris + resize a 120x120 (Pillow)
   → LBP por píxel (código 0-255 comparando con 8 vecinos)
   → grid 6x6 (36 celdas de ~20x20) → histograma de 256 bins por celda
   → concatenar → embedding de 9216 valores
Entrenamiento: promedio de embeddings por persona → U1, U2, U3
Clasificar:    cosine_similarity(Uk, V) → la mayor gana (o "Desconocido")
```

## Estructura

```text
reconocimiento_lbp/
├── user1/ user2/ user3/   # tus 10 imágenes por persona (recortadas al rostro)
├── src/
│   ├── image_loader.py     # abrir + gris + resize 120x120
│   ├── lbp.py              # cálculo LBP (propio)
│   ├── features.py         # histogramas por celda → embedding
│   ├── similarity.py       # similitud coseno
│   ├── model.py            # U1/U2/U3 + clasificación
│   ├── dataset.py          # listado + split 70/30 estratificado
│   └── evaluate.py         # accuracy + matriz de confusión
├── main.py                 # programa interactivo (input de la imagen a evaluar)
├── test_pipeline.py        # prueba rápida de la canalización (sin dataset real)
└── requirements.txt
```

## Instalación

```bash
pip3 install -r requirements.txt
```

(Requiere Python 3. Dependencias: numpy, Pillow, scikit-image.)

## Preparar el dataset

1. Toma ~10 fotos por persona: frontales, con ligeras variaciones de expresión e
   iluminación.
2. **Recorta el rostro** de cada foto (con cualquier editor) y deja la cara
   aproximadamente centrada.
3. Guarda las imágenes en `user1/`, `user2/`, `user3/` (formatos `.jpg`, `.png`,
   `.bmp`). El programa se encarga de pasarlas a gris y redimensionarlas.

## Uso

Verificar que la canalización funciona (no necesita imágenes reales):

```bash
python3 test_pipeline.py
```

Evaluar el sistema (partición 70/30, accuracy y matriz de confusión):

```bash
python3 -m src.evaluate
```

Clasificar una imagen nueva de forma interactiva:

```bash
python3 main.py
# Ruta de la imagen a evaluar: ruta/a/tu/foto.jpg
```

## Notas

- **Partición:** 70% entrenamiento / 30% prueba, estratificada por persona y
  determinista (orden por nombre de archivo).
- **Umbral de "Desconocido":** configurable en `src/model.py`
  (`UMBRAL_DESCONOCIDO`, por defecto 0.85). Ajústalo observando las similitudes
  reales de tu dataset.
- **Mejoras futuras:** LBP uniforme (59 bins), distancia chi-cuadrado, vecino más
  cercano (1-NN) y detección automática de rostro.
