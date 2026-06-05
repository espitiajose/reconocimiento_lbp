"""Visualización en consola y gráfica de embeddings y similitudes."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np

NUM_CELDAS = 6  # grid 6x6 — mismo valor que features.NUM_CELDAS_LADO


# ── Consola ───────────────────────────────────────────────────────────────────


def imprimir_histograma_consola(embedding, label="embedding", ancho=60, bins=32):
    histo_global = embedding.reshape(36, 256).sum(axis=0)
    grupos = np.array_split(histo_global, bins)
    valores = np.array([g.sum() for g in grupos])
    maximo = valores.max() if valores.max() > 0 else 1.0

    print(f"\n{'─'*60}")
    print(f"  Histograma LBP — {label}")
    print(f"{'─'*60}")
    for i, v in enumerate(valores):
        codigo_inicio = i * (256 // bins)
        barra = int(v / maximo * ancho)
        print(
            f"  {codigo_inicio:3d}-{codigo_inicio + 256//bins - 1:<3d} "
            f"│{'█'*barra}{' '*(ancho-barra)}│ {v:.4f}"
        )
    print(f"{'─'*60}")


def imprimir_similitudes_conseno(similitudes, ancho=40):
    print(f"\n{'─'*55}")
    print("  Similitud coseno vs. vectores representativos")
    print(f"{'─'*55}")
    for persona, sim in sorted(similitudes.items(), key=lambda x: -x[1]):
        barra = int(sim * ancho)
        print(f"  {persona:>8} │{'█'*barra}{' '*(ancho-barra)}│ {sim:.6f}")
    print(f"{'─'*55}")


# ── Helpers internos ──────────────────────────────────────────────────────────


def _histo_grupos(embedding, bins=32):
    h = embedding.reshape(36, 256).sum(axis=0)
    grupos = np.array_split(h, bins)
    return np.array([g.sum() for g in grupos])


def _dibujar_imagen_lbp(ax, codigos_lbp, embedding):
    """Panel: imagen LBP en escala de grises + grid 6×6 + código dominante/celda."""
    ax.imshow(codigos_lbp, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Imagen LBP + segmentación 6×6", fontsize=10)
    ax.axis("off")

    filas, columnas = codigos_lbp.shape
    alto_celda = filas // NUM_CELDAS
    ancho_celda = columnas // NUM_CELDAS

    # Precomputar código dominante de cada celda desde el embedding
    # embedding = 36 histogramas de 256 bins en orden (fila_celda, col_celda)
    hists_celdas = embedding.reshape(NUM_CELDAS, NUM_CELDAS, 256)

    for fila_celda in range(NUM_CELDAS):
        for col_celda in range(NUM_CELDAS):
            y0 = fila_celda * alto_celda
            x0 = col_celda * ancho_celda
            alto = alto_celda if fila_celda < NUM_CELDAS - 1 else filas - y0
            ancho = ancho_celda if col_celda < NUM_CELDAS - 1 else columnas - x0

            # Borde de celda
            rect = patches.Rectangle(
                (x0 - 0.5, y0 - 0.5),
                ancho,
                alto,
                linewidth=0.8,
                edgecolor="cyan",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Código LBP dominante (bin con mayor frecuencia en esa celda)
            hist_celda = hists_celdas[fila_celda, col_celda]
            codigo_dom = int(np.argmax(hist_celda))
            cx = x0 + ancho / 2
            cy = y0 + alto / 2
            ax.text(
                cx,
                cy,
                str(codigo_dom),
                ha="center",
                va="center",
                fontsize=5.5,
                color="yellow",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.55, lw=0),
            )


# ── Ventana unificada ─────────────────────────────────────────────────────────


def mostrar_ventana_completa(
    ruta_imagen,
    embedding_v,
    vectores_u,
    similitudes,
    codigos_lbp=None,
    label_v="consulta",
):
    """Muestra en UNA ventana matplotlib:
    - Imagen original (grises)
    - Imagen LBP con grid 6×6 y código dominante por celda (si codigos_lbp dado)
    - Plano cartesiano 2D (PCA) con todos los vectores
    - Histogramas LBP comparados: consulta vs. cada representativo
    """
    personas = list(vectores_u.keys())
    n = len(personas)
    colores = plt.cm.tab10.colors

    mostrar_lbp = codigos_lbp is not None
    n_cols_top = 2 + (1 if mostrar_lbp else 0)  # img [+ lbp] + 2D

    fig = plt.figure(figsize=(4 * n_cols_top + 4 * n, 10))
    fig.suptitle(f"Análisis LBP — {label_v}", fontsize=13, fontweight="bold")

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.4, 1], hspace=0.4)

    top = gridspec.GridSpecFromSubplotSpec(
        1, n_cols_top, subplot_spec=outer[0], wspace=0.35
    )
    bot = gridspec.GridSpecFromSubplotSpec(1, n + 1, subplot_spec=outer[1], wspace=0.35)

    ax_img = fig.add_subplot(top[0])
    if mostrar_lbp:
        ax_lbp = fig.add_subplot(top[1])
        ax_2d = fig.add_subplot(top[2])
    else:
        ax_2d = fig.add_subplot(top[1])

    ax_hists = [fig.add_subplot(bot[i]) for i in range(n + 1)]

    # ── Panel: imagen original ────────────────────────────────────────────────
    try:
        img = plt.imread(ruta_imagen)
        ax_img.imshow(img, cmap="gray" if img.ndim == 2 else None)
    except Exception:
        ax_img.text(
            0.5,
            0.5,
            "imagen\nno disponible",
            ha="center",
            va="center",
            transform=ax_img.transAxes,
        )
    ax_img.set_title("Imagen original (gris)", fontsize=10)
    ax_img.axis("off")

    # ── Panel: imagen LBP + grid ──────────────────────────────────────────────
    if mostrar_lbp:
        _dibujar_imagen_lbp(ax_lbp, codigos_lbp, embedding_v)

    # ── Panel: plano 2D PCA ───────────────────────────────────────────────────
    U = np.array([vectores_u[p] for p in personas])
    V = embedding_v.reshape(1, -1)
    todos = np.vstack([U, V])

    media = todos.mean(axis=0)
    centrado = todos - media
    _, _, Vt = np.linalg.svd(centrado, full_matrices=False)
    proyectado = centrado @ Vt[:2].T

    ax_2d.set_title("Vectores en plano cartesiano (PCA 2D)", fontsize=10)
    ax_2d.set_xlabel("PC 1", fontsize=8)
    ax_2d.set_ylabel("PC 2", fontsize=8)
    ax_2d.axhline(0, color="lightgray", linewidth=0.8)
    ax_2d.axvline(0, color="lightgray", linewidth=0.8)
    ax_2d.grid(True, linestyle=":", alpha=0.5)

    origen = np.zeros(2)
    for i, persona in enumerate(personas):
        x, y = proyectado[i]
        ax_2d.annotate(
            "",
            xy=(x, y),
            xytext=origen,
            arrowprops=dict(arrowstyle="->", color=colores[i], lw=2),
        )
        ax_2d.scatter(x, y, color=colores[i], s=80, zorder=5)
        ax_2d.text(x, y, f"  {persona}", color=colores[i], fontsize=9, va="center")

    x_v, y_v = proyectado[-1]
    ax_2d.annotate(
        "",
        xy=(x_v, y_v),
        xytext=origen,
        arrowprops=dict(arrowstyle="->", color="black", lw=2, linestyle="dashed"),
    )
    ax_2d.scatter(x_v, y_v, color="black", s=120, zorder=6, marker="*")
    ax_2d.text(x_v, y_v, "  consulta", color="black", fontsize=9, va="center")

    for i, persona in enumerate(personas):
        sim = similitudes[persona]
        mid_x = (proyectado[i, 0] + x_v) / 2
        mid_y = (proyectado[i, 1] + y_v) / 2
        ax_2d.text(
            mid_x,
            mid_y,
            f"cos={sim:.3f}",
            fontsize=8,
            color=colores[i],
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    # ── Paneles: histogramas LBP ──────────────────────────────────────────────
    x_bins = np.arange(32)
    v_q = _histo_grupos(embedding_v)

    ax_hists[0].bar(x_bins, v_q, color="black", alpha=0.75, width=0.9)
    ax_hists[0].set_title("consulta", fontsize=9, color="black")
    ax_hists[0].set_xlabel("grupo LBP", fontsize=7)
    ax_hists[0].set_ylabel("frecuencia", fontsize=7)
    ax_hists[0].tick_params(labelsize=6)

    for i, persona in enumerate(personas):
        v_u = _histo_grupos(vectores_u[persona])
        ax = ax_hists[i + 1]
        ax.bar(x_bins, v_q, color="black", alpha=0.3, width=0.9, label="consulta")
        ax.bar(x_bins, v_u, color=colores[i], alpha=0.6, width=0.9, label=persona)
        ax.set_title(
            f"{persona}  cos={similitudes[persona]:.4f}", fontsize=9, color=colores[i]
        )
        ax.set_xlabel("grupo LBP", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    plt.show()
