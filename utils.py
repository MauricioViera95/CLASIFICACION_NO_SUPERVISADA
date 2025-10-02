# ========================================================================
# utils.py
# ------------------------------------------------------------------------
# Determinación de la ventana ráster a ser utilizada como caso de estudio
# ------------------------------------------------------------------------
# @author   Mauricio Viera-Torres <ronnyvieramt95@gmail.com>
# @address  La Tola (Ecuador)
# @version  1.0.0 (2025-07-07)
# ========================================================================

import rasterio
import numpy as np
from rasterio.windows import Window

def leer_ventana_raster(ruta_raster, col_off, row_off, width, height):
    with rasterio.open(ruta_raster) as src:
        window = Window(col_off, row_off, width, height)
        img = src.read(window=window)
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update({
            "height": height,
            "width": width,
            "transform": transform
        })
        crs = src.crs
    return img, meta, crs

def raster_to_X(img):
    bands, rows, cols = img.shape
    return img.reshape(bands, rows * cols).T, rows, cols

def save_geotiff(filename, array2d, meta, crs):
    import rasterio
    meta.update({"count": 1, "dtype": 'uint8', "crs": crs})
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(array2d.astype('uint8'), 1)