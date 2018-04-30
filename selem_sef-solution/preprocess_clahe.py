import os
from multiprocessing.pool import Pool

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.external import tifffile

from params import args

wdata_dir = args.wdata_dir

def transform(f):
    path = f
    city_dir_name = f.split("/")[-3]
    image = tifffile.imread(path)
    bands = []
    for band in range(8):
        bands.append(equalize_adapthist(image[..., band]) * 2047)
    img = np.array(np.stack(bands, axis=-1), dtype="uint16")
    clahe_city_dir = os.path.join(wdata_dir, city_dir_name)
    os.makedirs(clahe_city_dir, exist_ok=True)
    mul_dir = os.path.join(clahe_city_dir, 'CLAHE-MUL-PanSharpen')
    os.makedirs(mul_dir, exist_ok=True)
    tifffile.imsave(os.path.join(mul_dir, f.split("/")[-1]), img, planarconfig='contig')


def multi_transform(files, transform):
    pool = Pool(8)
    results = pool.map(transform, files)
    pool.close()
    pool.join()
    return results

for city_dir in args.dirs_to_process:
    print("preprocess data for dir ", city_dir)
    mul_dir = os.path.join(city_dir, 'MUL-PanSharpen')
    files = [os.path.join(mul_dir, f) for f in os.listdir(mul_dir)]
    multi_transform(files, transform)
