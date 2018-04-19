import os

import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo

from .eval import Evaluator
import shutil


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir_path = os.path.join(self.config.dataset_path, 'results', self.config.folder + ("_test" if self.test else ""))

        if self.config.save_images:
            os.makedirs(self.dir_path, exist_ok=True)

    def process_data(self, predicted, model, data, prefix=""):
        names, samples, masks = self.get_data(data)
        for i in range(len(names)):
            self.prev_name = names[i]
            self.full_pred = np.squeeze(predicted[i,...])
            if samples is not None:
                self.full_image = (samples[i,...] * 255).astype(np.uint8)
            if masks is not None:
                self.full_mask = (np.squeeze(masks[i,...]) * 255).astype(np.uint8)
            self.on_image_constructed(prefix)

    def save(self, name, prefix=""):
        assert self.full_pred.shape == (self.config.img_rows - self.border*2, self.config.img_cols - self.border*2)
        cv2.imwrite(os.path.join(self.dir_path, prefix + 'mask_{}'.format(name)), (self.full_pred * 255).astype(np.uint8))


class GdalSaver(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = self.ds.paths
        path = os.path.join(self.config.dataset_path, 'results', self.config.folder)
        shutil.rmtree(path, True)
        os.makedirs(path, exist_ok=True)

    def save(self, name, prefix=""):
        #todo has mask
        has_mask = False
        ref_file = os.path.join(self.paths['images'], name)
        res_path_geo = os.path.join(self.config.dataset_path, 'results', self.config.folder, prefix + name)
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(res_path_geo, self.full_pred.shape[1], self.full_pred.shape[0], 1, gdal.GDT_Float32)
        if os.path.isfile(ref_file):
            gdalData = gdal.Open(ref_file, gdal.GA_ReadOnly)
            nodata = gdalData.GetRasterBand(1).GetNoDataValue()
            if has_mask:
                mask = np.array(gdalData.GetRasterBand(4).ReadAsArray())
                empty_mask = np.zeros((self.full_pred.shape[0], self.full_pred.shape[1]))
                empty_mask[0:mask.shape[0], 0:mask.shape[1]] = mask
                empty_mask = np.invert(empty_mask.astype(np.bool))
                self.full_pred[empty_mask] = nodata
            geoTrans = gdalData.GetGeoTransform()
            outRaster.SetGeoTransform(geoTrans)
            CopyDatasetInfo(gdalData, outRaster)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(self.full_pred)
        outRaster.FlushCache()


class GdalFullEvaluator(GdalSaver, FullImageEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, name, prefix=""):
        GdalSaver.save(self, name, prefix)
