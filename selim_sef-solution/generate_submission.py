from multiprocessing.pool import Pool

import cv2
import numpy as np
import os

from params import args
from tools.vectorize import to_line_strings

folders = [
    'all_masks/linknet_inception',
    'all_masks/inception-unet',
    'all_masks/clahe_inception-swish',
    'all_masks/clahe_linknet_inception',
    'all_masks/clahe_linknet_inception_lite',
    'all_masks/clahe_linknet_resnet50'
]


def predict(f):
    image_id = f.split('MUL-PanSharpen_')[1].split(".tif")[0]
    masks = []
    for folder in folders:
        masks.append(cv2.imread(os.path.join(folder, f + ".png")) / 255)
    mask = np.average(np.array(masks), axis=0)
    line_strings = to_line_strings(mask, threashold=0.25, sigma=0.5, dilation=1)
    result = ""
    if len(line_strings) > 0:
        for line_string in line_strings:
            result += '{image_id},"{line}"\n'.format(image_id=image_id, line=line_string)
    else:
        result += "{image_id},{line}\n".format(image_id=image_id, line="LINESTRING EMPTY")

    return result


def multi_predict(X, predict):
    pool = Pool(4)
    results = pool.map(predict, X)
    pool.close()
    pool.join()
    return results


f_submit = open(args.output_file + ".txt", "w")

for city_dir in args.dirs_to_process:
    print("ensemble for dir ", city_dir)
    pool = Pool(4)

    test_dir = os.path.join(city_dir, 'MUL-PanSharpen')
    files = sorted(os.listdir(test_dir))
    city_results = multi_predict(files, predict)
    for line in city_results:
        f_submit.write(line)

f_submit.close()
