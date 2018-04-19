import os
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo
from scipy.spatial.distance import dice
import tqdm
import numpy as np
import cv2

def merge_tiffs(root):
    os.makedirs(os.path.join(root, 'merged'), exist_ok=True)
    prob_files = {f for f in os.listdir(root) if os.path.splitext(f)[1] in ['.tif', '.tiff']}
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    if not unfolded:
        unfolded = prob_files

    for prob_file in tqdm.tqdm(unfolded):
        probs = []
        for fold in range(5):
            prob = os.path.join(root, 'fold{}_'.format(fold) + prob_file)
            prob_arr = cv2.imread(prob, cv2.IMREAD_GRAYSCALE)
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)

        res_path_geo = os.path.join(root, 'merged', prob_file)
        cv2.imwrite(res_path_geo, prob_arr)

def merge_tiffs_defferent_folders(roots, res):
    os.makedirs(os.path.join(res), exist_ok=True)
    prob_files = {f for f in os.listdir(roots[0]) if os.path.splitext(f)[1] in ['.tif', '.tiff']}

    for prob_file in tqdm.tqdm(prob_files):
        probs = []
        for root in roots:
            prob = gdal.Open(os.path.join(root, prob_file), gdal.GA_ReadOnly)
            geotrans = prob.GetGeoTransform()
            prob_arr = prob.ReadAsArray()
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)
        # prob_arr = np.clip(probs[0] * 0.7 + probs[1] * 0.3, 0, 1.)

        res_path_geo = os.path.join(res, prob_file)
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(res_path_geo, prob_arr.shape[1], prob_arr.shape[0], 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform(geotrans)
        CopyDatasetInfo(prob, outRaster)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(prob_arr)
        outRaster.FlushCache()

def all_dice(pred_path, gt_path):
    all_d= []
    for im in os.listdir(pred_path):
        img_ds = gdal.Open(os.path.join(pred_path, im), gdal.GA_ReadOnly)
        img = img_ds.GetRasterBand(1).ReadAsArray()
        gt_ds = gdal.Open(os.path.join(gt_path, im.replace('RGB', "GTI")), gdal.GA_ReadOnly)
        gt = gt_ds.GetRasterBand(1).ReadAsArray()
        dsm_ds = gdal.Open(os.path.join(gt_path, im.replace('RGB', 'DSM')), gdal.GA_ReadOnly)
        band_dsm = dsm_ds.GetRasterBand(1)
        nodata = band_dsm.GetNoDataValue()
        dsm = band_dsm.ReadAsArray()
        img[dsm==nodata] = 0
        gt[dsm==nodata] = 0

        d = 1 - dice(img.flatten() > .4, gt.flatten() >= 1)
        print(im, d)
        all_d.append(d)
    print(np.mean(all_d))

if __name__ == "__main__":
    #upscale bce adam 8489
    root = os.path.join('..', '..')
    # all_dice(os.path.join(root, 'results', 'resnet34'), os.path.join('training'))
    merge_tiffs(os.path.join(root, 'results', 'results', 'resnet34_test'))

