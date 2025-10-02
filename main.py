# ========================================================================
# main.py
# ------------------------------------------------------------------------
# Módulo principal de algoritmos de clasificación no supervisada
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-07)
# ========================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from tqdm import tqdm
from sklearn.decomposition import PCA
import itertools

from config import raster_path, output_dir
from utils import leer_ventana_raster, raster_to_X, save_geotiff
from kmeans_model import entrenar_kmeans
from gmm_model import seleccionar_mejor_gmm

# === Definición de ventana (ajusta si es necesario)
col_off, row_off = 15000, 15000
width, height = 1500, 1500

# 1. Leer solo una ventana del raster
img, meta, crs = leer_ventana_raster(raster_path, col_off, row_off, width, height)

# 2. Convertir a matriz y aplicar PCA (solo RGB)
X_base, rows, cols = raster_to_X(img)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_base[:, :3])

# 3. Evaluación con KMeans
print("\n\U0001F535 Iniciando evaluación KMeans...\n")
k_vals = [2, 3, 4, 5, 6]
resultados_kmeans = []
kmeans_resultados_img = []
kmeans_labels_list = []

for k in tqdm(k_vals, desc="Evaluando KMeans", unit="modelo"):
    labels, model, inertia, sil_score = entrenar_kmeans(X_base[:, :3], k)
    resultados_kmeans.append({"k": k, "inertia": inertia, "silhouette": sil_score})
    resultado_img = labels.reshape(rows, cols)
    kmeans_resultados_img.append((k, resultado_img))
    kmeans_labels_list.append(labels)

# 4. Evaluación con GMM
print("\n\U0001F7E3 Iniciando evaluación GMM...\n")
componentes_lista = [2, 3, 4, 5, 6]
tipos_cov = ['spherical', 'full']
labels_gmm, modelo_gmm, bics = seleccionar_mejor_gmm(X_pca, componentes_lista, tipos_cov)
resultado_gmm = labels_gmm.reshape(rows, cols)


# 6. Visualización individual por modelo (KMeans)
print("\n\U0001F4CA Visualización individual KMeans...")
for i, (k, resultado_img) in enumerate(kmeans_resultados_img):
    labels = kmeans_labels_list[i]
    cmap = plt.cm.get_cmap('Spectral', k)

    # Crear figura con 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

    # --- Subplot 1: Imagen segmentada ---
    im = axes[0].imshow(resultado_img, cmap=cmap, norm=Normalize(vmin=0, vmax=k-1))
    axes[0].set_title(f"KMeans k={k} - Imagen segmentada",
                      fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    # Leyenda
    legend_patches = [mpatches.Patch(color=cmap(j), label=f"Clase {j+1}") for j in range(k)]
    axes[0].legend(handles=legend_patches, loc="lower left", fontsize=11,
                   frameon=True, facecolor="white")

    # --- Subplot 2: PCA ---
    N = 1000
    R = np.random.randint(low=0, high=X_pca.shape[0], size=N)
    axes[1].scatter(
        X_pca[R, 0],
        X_pca[R, 1],
        c=labels[R],
        cmap=cmap,
        s=18,
        edgecolors="k", linewidths=0.3, alpha=0.9
    )
    axes[1].set_title(f"KMeans k={k} - Dispersión PCA",
                      fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("PCA 1", fontsize=12)
    axes[1].set_ylabel("PCA 2", fontsize=12)
    axes[1].tick_params(axis="both", labelsize=11)
    axes[1].grid(True, alpha=0.4)

    # Ajustar todo compacto y legible
    plt.tight_layout()
    plt.show()

# 7. Visualización resumen KMeans (solo resultados + RGB)
rgb_img = np.transpose(img[[0, 1, 2]], (1, 2, 0)).astype(np.uint8)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), dpi=130)
axes = axes.flatten()

axes[0].imshow(rgb_img)
axes[0].set_title("Imagen RGB", fontsize=14, fontweight="bold", pad=8)
axes[0].axis('off')

for idx, (k, resultado_img) in enumerate(kmeans_resultados_img):
    if idx + 1 >= len(axes): 
        break
    ax = axes[idx + 1]
    cmap = plt.cm.get_cmap('Spectral', k)
    ax.imshow(resultado_img, cmap=cmap, norm=Normalize(vmin=0, vmax=k - 1))
    ax.set_title(f"KMeans k={k}", fontsize=13, fontweight="bold", pad=6)
    ax.axis('off')

plt.tight_layout()
plt.show()


# 8. Gráficos separados: Inercia y Silhouette
inertia_vals = [r['inertia'] for r in resultados_kmeans]
silhouette_vals = [r['silhouette'] for r in resultados_kmeans]
ks = [r['k'] for r in resultados_kmeans]

plt.figure(figsize=(8, 4))
plt.plot(ks, inertia_vals, 'bo-', label='Inercia')
plt.title("Gráfico del codo (Inercia)")
plt.xlabel("k")
plt.ylabel("Inercia")
plt.grid(True)
plt.xticks(ks)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(ks, silhouette_vals, 'r^-', label='Silhouette')
plt.title("Índice de Silhouette")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.grid(True)
plt.xticks(ks)
plt.tight_layout()
plt.show()

# 9. Visualización GMM (Mejor Modelo + PCA en un solo gráfico)
print("\n\U0001F4CA Visualización GMM Mejor Modelo...")

num_clases_gmm = len(np.unique(labels_gmm))
cmap_gmm = plt.cm.get_cmap('jet', num_clases_gmm)

# Crear figura con 2 subplots lado a lado
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

# --- Subplot 1: Imagen segmentada GMM ---
im = axes[0].imshow(resultado_gmm, cmap=cmap_gmm,
                    norm=Normalize(vmin=0, vmax=num_clases_gmm - 1))
axes[0].set_title("GMM Mejor modelo", fontsize=14, fontweight="bold", pad=10)
axes[0].axis('off')

# Leyenda
legend_patches = [mpatches.Patch(color=cmap_gmm(j), label=f"Clase {j+1}") 
                  for j in range(num_clases_gmm)]
axes[0].legend(handles=legend_patches, loc="lower left", fontsize=11,
               frameon=True, facecolor="white")

# --- Subplot 2: PCA ---
N = 1000
R = np.random.randint(low=0, high=X_pca.shape[0], size=N)
axes[1].scatter(
    X_pca[R, 0],
    X_pca[R, 1],
    c=labels_gmm[R],
    cmap=cmap_gmm,
    s=18,
    edgecolors="k", linewidths=0.3, alpha=0.9
)
axes[1].set_title("GMM - Dispersión PCA", fontsize=14, fontweight="bold", pad=10)
axes[1].set_xlabel("PCA 1", fontsize=12)
axes[1].set_ylabel("PCA 2", fontsize=12)
axes[1].tick_params(axis="both", labelsize=11)
axes[1].grid(True, alpha=0.4)

# Ajustar layout compacto
plt.tight_layout()
plt.show()

# 10. Gráfico BIC GMM tipo barras agrupadas
bic_values = [b[2] for b in bics]
n_components_range = sorted(set(b[1] for b in bics))
cv_types = sorted(set(b[0] for b in bics))
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])

plt.figure(figsize=(8, 5), dpi=130)  # más compacto
bars = []
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    yvals = [b[2] for b in bics if b[0] == cv_type]
    bars.append(plt.bar(xpos, yvals, width=.2, color=color))

plt.xticks(n_components_range, fontsize=11)
plt.yticks(fontsize=11)
plt.title('BIC score por modelo', fontsize=14, fontweight="bold", pad=10)
plt.xlabel('Número de componentes', fontsize=12)
plt.ylabel('BIC', fontsize=12)

# Marca el mejor modelo
xpos = np.argmin(bic_values) % len(n_components_range) + .65 + 0.2 * (np.argmin(bic_values) // len(n_components_range))
plt.text(xpos, min(bic_values) * 1.02, '*', fontsize=16, color='red')

plt.legend([b[0] for b in bars], cv_types, fontsize=11)
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()


# 11. Visualización resumen GMM
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(rgb_img)
axes[0].set_title("Imagen RGB")
axes[0].axis('off')
axes[1].imshow(resultado_gmm, cmap=cmap_gmm, norm=Normalize(vmin=0, vmax=num_clases_gmm - 1))
axes[1].set_title("GMM Mejor modelo")
axes[1].axis('off')
plt.tight_layout()
plt.show()

import os
os.environ["OMP_NUM_THREADS"] = "1"

from slic_gmm_model import entrenar_slic_gmm
from utils import leer_ventana_raster
from config import raster_path, output_dir

# === Leer ventana del raster
col_off, row_off = 15000, 15000
width, height = 1500, 1500
img, meta, crs = leer_ventana_raster(raster_path, col_off, row_off, width, height)

# === Ejecutar modelo SLIC + GMM
entrenar_slic_gmm(img, crs, meta)
