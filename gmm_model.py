# ========================================================================
# gmm_model.py
# ------------------------------------------------------------------------
# Función que ejecuta el algoritmo GMM y busca el modelo más óptimo
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-07)
# ========================================================================

# gmm_model.py
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

def entrenar_gmm(X, n_components=6, covariance_type='full'):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.fit(X)
    return gmm.predict(X), gmm

def seleccionar_mejor_gmm(X, componentes_lista, tipos_cov):
    mejor_bic = float('inf')
    mejor_modelo = None
    bics = []

    total_iters = len(componentes_lista) * len(tipos_cov)
    iterador = tqdm(total=total_iters, desc="Evaluando GMM")

    for cov in tipos_cov:
        for n in componentes_lista:
            modelo = GaussianMixture(n_components=n, covariance_type=cov)
            modelo.fit(X)
            bic_valor = modelo.bic(X)
            bics.append((cov, n, bic_valor))
            if bic_valor < mejor_bic:
                mejor_bic = bic_valor
                mejor_modelo = modelo
            iterador.update(1)
    iterador.close()
    return mejor_modelo.predict(X), mejor_modelo, bics