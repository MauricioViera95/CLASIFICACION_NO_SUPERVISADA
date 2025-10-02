# ========================================================================
# slic_gmm_model.py
# ------------------------------------------------------------------------
# Funciones para ejecutar el algoritmo de superpíxeles y utilizarlo como 
# insumo para el GMM
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-09)
# ========================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import mixture
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm
import itertools
import matplotlib.patches as mpatches
import time


def compute_mean_data(data, labels, compute_std=False):
    data_dim = data.shape[2] * (2 if compute_std else 1)
    nlabels = labels.max() + 1
    out = np.zeros([nlabels, data_dim])
    for label in np.unique(labels):
        region = data[labels == label]
        if compute_std:
            out[label] = np.hstack([region.mean(axis=0), region.std(axis=0)])
        else:
            out[label] = region.mean(axis=0)
    return out


def mean_color(image, labels):
    out = np.zeros_like(image)
    for label in np.unique(labels):
        indices = np.nonzero(labels == label)
        out[indices] = np.mean(image[indices], axis=0)
    return out


from skimage.segmentation import slic, find_boundaries
from skimage.morphology import binary_dilation, disk
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
from tqdm import tqdm

def overlay_boundaries(img_rgb, segments, color=(255, 255, 0), radius=1):
    """
    Devuelve una copia de img_rgb con los límites de 'segments' coloreados.
    radius: grosor del borde (0=solo 'thick'; 1-2 mayor grosor).
    """
    bnd = find_boundaries(segments, mode='thick')
    if radius and radius > 0:
        bnd = binary_dilation(bnd, disk(radius))
    out = img_rgb.copy()
    out[bnd] = color
    return out

def entrenar_slic_gmm(img, crs, meta, numSegments=128):
    # Asegurar orden R,G,B
    image = np.swapaxes(np.swapaxes(img[[0, 1, 2]], 0, 1), 1, 2).copy()
    image_RGB = image.copy()
    image_full = image.copy()

    # --- Superpíxeles iniciales ---
    segments = slic(image, n_segments=numSegments, sigma=5)

    # === Cálculo de características por superpíxel ===
    mean_data_std = compute_mean_data(image_full, segments, compute_std=True)

    # === Búsqueda GMM por BIC ===
    print("\n\U0001F4CA Evaluación GMM por segmentos (SLIC)...")
    lowest_bic = np.inf
    bic = []
    n_components_range = np.arange(2, 20)
    cv_types = ['full']

    best_gmm = None
    for cv_type in cv_types:
        for n_components in tqdm(n_components_range, desc=f"{cv_type} GMM", unit="modelo"):
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(mean_data_std)
            bic_val = gmm.bic(mean_data_std)
            bic.append(bic_val)
            if bic_val < lowest_bic:
                lowest_bic = bic_val
                best_gmm = gmm

    bic = np.array(bic)
    best_n_components = n_components_range[np.argmin(bic)]
    print(f"\nMejor modelo GMM: {best_n_components} componentes")

    # === FIGURA COMBINADA 1: (izq) SLIC con bordes amarillos + (der) BIC ===
    print("Mostrando segmentación SLIC y gráfico BIC...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

    # Izquierda: superpíxeles con borde amarillo más notorio
    slic_overlay = overlay_boundaries(image_RGB, segments, color=(255, 255, 0), radius=1)  # ↑ grosor con radius
    axes[0].imshow(slic_overlay)
    axes[0].set_title(f"Superpíxeles (n={numSegments})", fontsize=14, fontweight='bold', pad=10)
    axes[0].axis("off")

    # Derecha: curva BIC
    axes[1].plot(n_components_range, bic, marker='o', linestyle='-', color='cornflowerblue', label='BIC')
    axes[1].axhline(y=min(bic), color='darkorange', linestyle='--', linewidth=1.5, label='Mínimo BIC')
    axes[1].set_title('BIC por número de componentes', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Número de componentes', fontsize=12)
    axes[1].set_ylabel('BIC', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=11)
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    # === Etiquetado con el mejor GMM ===
    segment_labels = best_gmm.predict(mean_data_std)
    final_labels = np.array(list(map(lambda s: segment_labels[s], segments.flatten()))).reshape(segments.shape)

    # Imagen RGB promedio por clase (colores reales)
    mc_img = mean_color(image_RGB, final_labels)

    # === FIGURA COMPARATIVA 2: RGB | SLIC óptimo | GMM+SLIC (leyenda externa) ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5), dpi=130)

    axes2[0].imshow(image_RGB)
    axes2[0].set_title("Imagen RGB", fontsize=13, fontweight='bold', pad=8)
    axes2[0].axis('off')

    segments_opt = slic(image, n_segments=best_n_components, sigma=5)
    slic_opt_overlay = overlay_boundaries(image_RGB, segments_opt, color=(255, 255, 0), radius=1)
    axes2[1].imshow(slic_opt_overlay)
    axes2[1].set_title(f"Súperpíxeles óptimos (n={best_n_components})", fontsize=13, fontweight='bold', pad=8)
    axes2[1].axis('off')

    axes2[2].imshow(mc_img)
    axes2[2].set_title(f"GMM + SLIC ({best_n_components} clases)", fontsize=13, fontweight='bold', pad=8)
    axes2[2].axis('off')

    # Leyenda externa (a la derecha)
    legend_patches = []
    for i in range(best_n_components):
        color_val = np.mean(mc_img[final_labels == i], axis=0) / 255.0  # normalizado [0,1]
        legend_patches.append(mpatches.Patch(color=color_val, label=f"Clase {i+1}"))

    # Colocar fuera del último subplot
    axes2[2].legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=11,
        frameon=True,
        facecolor='white'
    )

    # Dejar espacio a la derecha para la leyenda externa
    plt.subplots_adjust(right=0.83) 
    plt.show()
