# -*- coding: utf-8 -*-
from os import path, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
import timeit
import cv2
from tqdm import tqdm
import sys
from shapely.wkt import loads

cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

if __name__ == '__main__':
    t0 = timeit.default_timer()

    masks_folder = path.join('/wdata', sys.argv[1])
    sz = int(sys.argv[2])
    thickness = int(sys.argv[3])
    radius = int(0.85 * thickness)
    ratio = 1300.0 / sz
    
    train_folders = []
    for i in range(4, len(sys.argv)):
        train_folders.append(sys.argv[i])
        
    if not path.isdir(masks_folder):
        mkdir(masks_folder)
        
    for dd in train_folders:
        df = pd.read_csv(path.join(dd, 'summaryData', '{0}.csv'.format(dd.split('/')[-1])))
        city = df['ImageId'].values[0].split('_')[2]
        print('creating masks for', city)
        if not path.isdir(path.join(masks_folder, city)):
            mkdir(path.join(masks_folder, city))
        for img_id in tqdm(df['ImageId'].unique()):
            lines = [loads(s) for s in df[df['ImageId'] == img_id]['WKT_Pix']]
            img = np.zeros((sz, sz), np.uint8)
            img2 = np.zeros((sz, sz), np.uint8)
            img3 = np.zeros((sz, sz), np.uint8)
            
            d = {}
            
            for l in lines:
                if len(l.coords) == 0:
                    continue
                x, y = l.coords.xy
                for i in range(len(x)):
                    x[i] /= ratio
                    y[i] /= ratio
                
                x_int = int(round(x[0] * 10))
                y_int = int(round(y[0] * 10))
                h = x_int * 100000 + y_int
                if not (h in d.keys()):
                    d[h] = 0
                d[h] = d[h] + 1
                
                for i in range(len(x) - 1):
                    x_int = int(round(x[i+1] * 10))
                    y_int = int(round(y[i+1] * 10))
                    h = x_int * 100000 + y_int
                    if not (h in d.keys()):
                        d[h] = 0
                    if i == len(x) - 2:
                        d[h] = d[h] + 1
                    else:
                        d[h] = d[h] + 2
                    cv2.line(img, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), 255, thickness)
            for h in d.keys():
                if d[h] > 2:
                    x_int = int(h / 100000)
                    y_int = h - x_int * 100000
                    x_int = int(x_int / 10)
                    y_int = int(y_int / 10)
                    cv2.circle(img2, (x_int, y_int), radius, 255, -1)
            img = img[..., np.newaxis]
            img2 = img2[..., np.newaxis]
            img3 = img3[..., np.newaxis]
            img = np.concatenate([img, img2, img3], axis=2)
            cv2.imwrite(path.join(masks_folder, city, '{0}{1}'.format(img_id, '.png')), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))