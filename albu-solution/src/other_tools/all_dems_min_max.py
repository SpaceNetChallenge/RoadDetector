import os
import numpy as np
from osgeo import gdal

def minmax():
    root = r'D:\tmp\map3d\testing'
    all_mean = []
    for fn in os.listdir(root):
        if 'DSM' not in fn or 'xml' in fn:
            continue
        dsm = gdal.Open(os.path.join(root, fn))
        dtm = gdal.Open(os.path.join(root, fn.replace('DSM', "DTM")))
        dsm_band = dsm.GetRasterBand(1)
        dtm_band = dtm.GetRasterBand(1)
        stats_dsm = dsm_band.GetStatistics(True, True)
        mi, ma, mean, std = stats_dsm
        stats_dtm = dtm_band.GetStatistics(True, True)
        mi2, ma2, mean2, std2 = stats_dtm
        print(std)
        all_mean.append(ma - ma2)
    print(np.mean(all_mean))


minmax()