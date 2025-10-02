# Clasificacion_no_supervisada
Este repositorio implementa un flujo de clasificación no supervisada sobre ortofotos de alta resolución, utilizando algoritmos de clustering como KMeans, Gaussian Mixture Models (GMM) y un enfoque híbrido SLIC + GMM. El objetivo es segmentar superficies urbanas a partir de ortofotos y comparar el desempeño de los diferentes modelos.

📂 Estructura de Archivos

* main.py → Script principal. Coordina lectura de raster, reducción con PCA y ejecución de modelos (KMeans, GMM, SLIC+GMM).

* config.py → Configuración de rutas de entrada y salida (raster, shapefile, directorio de resultados).

* utils.py → Funciones auxiliares para leer ventanas raster, convertir imágenes a matrices y guardar GeoTIFF.

* kmeans_model.py → Entrenamiento y evaluación de KMeans (inercia y Silhouette).

* gmm_model.py → Entrenamiento y selección del mejor GMM basado en BIC.

* slic_gmm_model.py → Segmentación avanzada usando superpíxeles (SLIC) y clasificación con GMM.

* model_selection.py → Funciones de visualización de métricas (codo, Silhouette, BIC).




⚙️ Requisitos
Se recomienda crear un entorno virtual con Python 3.10.

Dependencias principales:
* numpy
* matplotlib
* tqdm
* scikit-learn
* rasterio
* scikit-image
* geopandas
* fiona

▶️ Ejecución
Ejecutar el flujo principal:
python main.py

El script:

* Carga una ventana definida de la ortofoto (config.py).
* Aplica PCA sobre las bandas RGB.
* Ejecuta KMeans y GMM, comparando métricas.
* Aplica segmentación avanzada con SLIC + GMM.
* Genera gráficas y resultados en el directorio de salida (output_dir).

📊 Resultados esperados

* Segmentaciones por KMeans y GMM.
* Segmentación optimizada con SLIC + GMM.
* Gráficas comparativas:
  * Codo (inercia).
  * Índice Silhouette.
  * BIC por modelo GMM.
  * Dispersión PCA coloreada por clusters.
