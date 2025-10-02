# ========================================================================
# kmeans_model.py
# ------------------------------------------------------------------------
# Función para ejecutar el algoritmo K-Means
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-07)
# ========================================================================

# kmeans_model.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def entrenar_kmeans(X, n_clusters):
    modelo = KMeans(n_clusters=n_clusters, random_state=0)
    modelo.fit(X)
    labels = modelo.labels_
    inertia = modelo.inertia_
    sil_score = silhouette_score(X, labels, sample_size=10000)  # limitar por performance
    return labels, modelo, inertia, sil_score
