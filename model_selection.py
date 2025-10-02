# ========================================================================
# model_selection.py
# ------------------------------------------------------------------------
# Funciones para gráficar los hiperparámetros a tomar en cuenta para 
# seleccionar el mejor modelo de K-Means y GMM
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-07)
# ========================================================================

# model_selection.py
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_kmeans_scores(scores):
    ks = [s['k'] for s in scores]
    inertia = [s['inertia'] for s in scores]
    silhouette = [s['silhouette'] for s in scores]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(ks, inertia, 'b-o', label='Inercia')
    ax2.plot(ks, silhouette, 'r--s', label='Silhouette')
    ax1.set_xlabel('Número de clusters (k)')
    ax1.set_ylabel('Inercia', color='b')
    ax2.set_ylabel('Silhouette Score', color='r')
    plt.title("Evaluación de KMeans")
    plt.grid(True)
    plt.show()

def plot_gmm_bic(bics, componentes_lista, tipos_cov):
    bic_values = np.array([b[2] for b in bics])
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
    bars = []

    plt.figure(figsize=(8, 6))
    for i, (cov, color) in enumerate(zip(tipos_cov, color_iter)):
        inicio = i * len(componentes_lista)
        fin = inicio + len(componentes_lista)
        xpos = np.array(componentes_lista) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic_values[inicio:fin], width=.2, color=color))
    plt.title('BIC por modelo GMM')
    plt.xlabel('Componentes')
    plt.legend([b[0] for b in bars], tipos_cov)
    plt.tight_layout()
    plt.show()
