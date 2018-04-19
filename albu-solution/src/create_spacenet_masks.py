#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten
"""

import os
import argparse
from collections import defaultdict
from osgeo import gdal
import numpy as np
import shutil

# add path and import apls_tools
from other_tools import apls_tools

rescale = {
    '2': {
        1: [25.48938322, 1468.79676441],
        2: [145.74823054, 1804.91911021],
        3: [155.47927199, 1239.49848332]
    },
    '4': {
        1: [79.29799666, 978.35058431],
        2: [196.66026711, 1143.74207012],
        3: [170.72954925, 822.32387312]
    },
    '3': {
        1: [46.26129032, 1088.43225806],
        2: [127.54516129, 1002.20322581],
        3: [141.64516129, 681.90967742]
    },
    '5': {
        1: [101.63250883, 1178.05300353],
        2: [165.38869258, 1190.5229682 ],
        3: [126.5335689, 776.70671378]
    }
}

def calc_rescale(im_file_raw, m, percentiles):
    srcRaster = gdal.Open(im_file_raw)
    for band in range(1, 4):
        b = srcRaster.GetRasterBand(band)
        band_arr_tmp = b.ReadAsArray()
        bmin = np.percentile(band_arr_tmp.flatten(),
                             percentiles[0])
        bmax= np.percentile(band_arr_tmp.flatten(),
                            percentiles[1])
        m[band].append((bmin, bmax))

    # for k, v in m.items():
    #     print(k, np.mean(v, axis=0))
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', type=str, nargs='+')
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()
    buffer_meters = 2
    burnValue = 255

    path_apls = r'/wdata'
    test = not args.training
    path_outputs = os.path.join(path_apls, 'train' if not test else 'test', 'masks{}m'.format(buffer_meters))
    path_images_8bit = os.path.join(path_apls, 'train' if not test else 'test', 'images')
    for d in [path_outputs, path_images_8bit]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    for path_data in args.datasets:
        path_data = path_data.rstrip('/')
        test_data_name = os.path.split(path_data)[-1]
        test_data_name = '_'.join(test_data_name.split('_')[:3]) + '_'
        path_images_raw = os.path.join(path_data, 'RGB-PanSharpen')
        path_labels = os.path.join(path_data, 'geojson/spacenetroads')

        # iterate through images, convert to 8-bit, and create masks
        im_files = os.listdir(path_images_raw)
        m = defaultdict(list)
        for im_file in im_files:
            if not im_file.endswith('.tif'):
                continue

            name_root = im_file.split('_')[-1].split('.')[0]

            # create 8-bit image
            im_file_raw = os.path.join(path_images_raw, im_file)
            im_file_out = os.path.join(path_images_8bit, test_data_name + name_root + '.tif')
            # convert to 8bit

            # m = calc_rescale(im_file_raw, m, percentiles=[2,98])
            # continue
            rescale_type = test_data_name.split('_')[1]
            if not os.path.isfile(im_file_out):
                apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
                                           outputPixType='Byte',
                                           outputFormat='GTiff',
                                           rescale_type=rescale[rescale_type],
                                           percentiles=[2,98])

            if test:
                continue
            # determine output files
            label_file = os.path.join(path_labels, 'spacenetroads_' + test_data_name + name_root + '.geojson')
            label_file_tot = os.path.join(path_labels, label_file)
            output_raster = os.path.join(path_outputs, test_data_name + name_root + '.png')

            print("\nname_root:", name_root)
            print("  output_raster:", output_raster)

            # create masks
            mask, gdf_buffer = apls_tools.get_road_buffer(label_file_tot, im_file_out,
                                                          output_raster,
                                                          buffer_meters=buffer_meters,
                                                          burnValue=burnValue,
                                                          bufferRoundness=6,
                                                          plot_file=None,
                                                          figsize= (6,6),  #(13,4),
                                                          fontsize=8,
                                                          dpi=200, show_plot=False,
                                                          verbose=False)

        for k, v in m.items():
            print(test_data_name, k, np.mean(v, axis=0))

###############################################################################
if __name__ == "__main__":
    main()
