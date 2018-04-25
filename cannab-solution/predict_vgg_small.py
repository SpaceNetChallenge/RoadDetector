# -*- coding: utf-8 -*-
import sys
from os import path, listdir, mkdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import timeit
import cv2
from models import get_vgg_unet_small
import skimage.io
from tqdm import tqdm

input_shape = (672, 672)

def preprocess_inputs(x):
    zero_msk = (x == 0)
    x = x / 8.0
    x -= 127.5
    x[zero_msk] = 0
    return x

models_folder = '/wdata/nn_models'
pred_folder = '/wdata/predictions'
model_name = 'vgg_small'

cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

ignored_cities = [1]

if __name__ == '__main__':
    t0 = timeit.default_timer()
        
    test_folders = []
    
    for i in range(1, len(sys.argv) - 1):
        test_folders.append(sys.argv[i])
    
    if not path.isdir(pred_folder):
        mkdir(pred_folder)
    
    if not path.isdir(path.join(pred_folder, model_name)):
        mkdir(path.join(pred_folder, model_name))    
        
    for it in [0]:        
        models = []
        
        if not path.isdir(path.join(pred_folder, model_name, str(it))):
            mkdir(path.join(pred_folder, model_name, str(it)))
            
        for i in range(4):
            if i in ignored_cities or not path.isfile(path.join(models_folder, 'vgg2_small_model_weights4_{0}_{1}.h5'.format(cities[i], it))):
                models.append(None)
                continue
            if not path.isdir(path.join(path.join(pred_folder, model_name, str(it), cities[i]))):
                mkdir(path.join(path.join(pred_folder, model_name, str(it), cities[i])))           
            model = get_vgg_unet_small(input_shape, weights=None)
            model.load_weights(path.join(models_folder, 'vgg2_small_model_weights4_{0}_{1}.h5'.format(cities[i], it)))
            models.append(model)
            
        print('Predictiong fold', it)
        for d in test_folders:  
            for f in tqdm(sorted(listdir(path.join(d, 'MUL')))):
                if path.isfile(path.join(d, 'MUL', f)) and '.tif' in f:
                    img_id = f.split('MUL_')[1].split('.')[0]
                    cinp = np.zeros((4,))
                    cinp[cities.index(img_id.split('_')[2])] = 1.0
                    cid = cinp.argmax()
                    if cid in ignored_cities:
                        continue
                    fpath = path.join(d, 'MUL', f)
                    img = skimage.io.imread(fpath, plugin='tifffile')
                    img = cv2.resize(img, (650, 650))
                    pan = skimage.io.imread(path.join(d, 'PAN', 'PAN_{0}.tif'.format(img_id)), plugin='tifffile')
                    pan = cv2.resize(pan, (650, 650))
                    pan = pan[..., np.newaxis]
                    img = np.concatenate([img, pan], axis=2)
                    img = cv2.copyMakeBorder(img, 11, 11, 11, 11, cv2.BORDER_REFLECT_101)
                    inp = []
                    inp.append(img)
                    inp.append(np.rot90(img, k=1))
                    inp = np.asarray(inp)
                    inp = preprocess_inputs(inp)
                    pred = models[cid].predict(inp)
                    mask = pred[0] + np.rot90(pred[1], k=3)
                    mask /= 2
                    mask = mask[11:661, 11:661, ...]
                    mask = mask * 255
                    mask = mask.astype('uint8')
                    cv2.imwrite(path.join(pred_folder, model_name, str(it), cities[cid], '{0}.png'.format(img_id)), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))