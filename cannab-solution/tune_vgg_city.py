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
from sklearn.model_selection import KFold
import cv2
from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import ModelCheckpoint
from models import get_vgg_unet, dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded
import skimage.io
import keras.backend as K

input_shape = (512, 512)

def preprocess_inputs(x):
    zero_msk = (x == 0)
    x = x / 8.0
    x -= 127.5
    x[zero_msk] = 0
    return x

masks_folder = r'/wdata/masks_22'
models_folder = r'/wdata/nn_models'

cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

city_id = -1

all_files = []
all_city_inp = []
all_masks = []

def rotate_image(image, angle, scale):
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result

def batch_data_generator(train_idx, batch_size):
    inputs = []
    inputs2 = []
    outputs = []
    while True:
        np.random.shuffle(train_idx)
        for i in train_idx:
            for j in range(1):
                img = skimage.io.imread(all_files[i], plugin='tifffile')
                msk = all_masks[i]
                if random.random() > 0.5:
                    scale = 0.9 + random.random() * 0.2
                    angle = random.randint(0, 41) - 24
                    img = rotate_image(img, angle, scale)
                    msk = rotate_image(msk, angle, scale)
                    
                x0 = random.randint(0, img.shape[1] - input_shape[1])
                y0 = random.randint(0, img.shape[0] - input_shape[0])
                img = img[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1]]

                if random.random() > 0.5:
                    img = img[:, ::-1, ...]
                    otp = otp[:, ::-1]
                
                rot = random.randrange(4)
                if rot > 0:
                    img = np.rot90(img, k=rot)
                    otp = np.rot90(otp, k=rot)
                    
                inputs.append(img)
                inputs2.append(all_city_inp[i])
                outputs.append(otp)
            
                if len(inputs) == batch_size:
                    inputs = np.asarray(inputs)
                    outputs = np.asarray(outputs, dtype='float')
                    outputs = outputs[..., np.newaxis]
                    inputs = preprocess_inputs(inputs)
                    inputs2 = np.asarray(inputs2)
                    yield [inputs, inputs2], outputs                
                    inputs = []
                    inputs2 = []
                    outputs = []

def val_data_generator(val_idx, batch_size, validation_steps):
    while True:
        inputs = []
        inputs2 = []
        outputs = []
        step_id = 0
        for i in val_idx:
            img0 = skimage.io.imread(all_files[i], plugin='tifffile')
            for x0 in [0, 512]:
                for y0 in [0, 512]:
                    img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                    otp = all_masks[i][y0:y0+input_shape[0], x0:x0+input_shape[1]]
                    inputs.append(img)
                    inputs2.append(all_city_inp[i])
                    outputs.append(otp)
                    if len(inputs) == batch_size:
                        step_id += 1
                        inputs = np.asarray(inputs)
                        outputs = np.asarray(outputs, dtype='float')
                        outputs = outputs[..., np.newaxis]
                        inputs = preprocess_inputs(inputs)
                        inputs2 = np.asarray(inputs2)
                        yield [inputs, inputs2], outputs                
                        inputs = []
                        inputs2 = []
                        outputs = []
                        if step_id == validation_steps:
                            break
                
if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    fold_nums = [0, 1]

    train_folders = []
    for i in range(1, len(sys.argv)):
        train_folders.append(sys.argv[i])
    
    if not path.isdir(models_folder):
        mkdir(models_folder)
        
    for d in train_folders:  
        for f in sorted(listdir(path.join(d, 'MUL-PanSharpen'))):
            if path.isfile(path.join(d, 'MUL-PanSharpen', f)) and '.tif' in f:
                img_id = f.split('PanSharpen_')[1].split('.')[0]
                all_files.append(path.join(d, 'MUL-PanSharpen', f))
                cinp = np.zeros((4,))
                cid = cities.index(img_id.split('_')[2])
                cinp[cid] = 1.0
                all_city_inp.append(cinp)
                msk = cv2.imread(path.join(masks_folder, cities[cid], '{0}{1}'.format(img_id, '.png')), cv2.IMREAD_UNCHANGED)[..., 0]
                msk = (msk > 127) * 1
                msk = msk.astype('uint8')
                all_masks.append(msk)
    all_files = np.asarray(all_files)
    all_city_inp = np.asarray(all_city_inp)
    all_masks = np.asarray(all_masks)

    batch_size = 4
    it = -1
    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    for all_train_idx, all_val_idx in kf.split(all_files):
        it += 1
        
        if it not in fold_nums:
            continue
        
        for cid in [1, 2, 3]:
            city_id = cid
            train_idx = []
            val_idx = []
            
            for i in all_train_idx:
                if all_city_inp[i][city_id] == 1:
                    train_idx.append(i)
            for i in all_val_idx:
                if all_city_inp[i][city_id] == 1:
                    val_idx.append(i)
                    
            validation_steps = int(4 * len(val_idx) / batch_size)
            steps_per_epoch = int(len(train_idx) / batch_size)
    
            if validation_steps == 0 or steps_per_epoch == 0:
                print("No data for city", cities[city_id])
                continue
            
            print('Training city', cities[city_id], 'fold', it)
            print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)
            
            np.random.seed(it+1)
            random.seed(it+1)
            tf.set_random_seed(it+1)
    
            print('Training model', it)
        
            model = get_vgg_unet(input_shape, weights=None)
            model.load_weights(path.join(models_folder, 'vgg_model3_weights3_{0}.h5'.format(it)))
            model.compile(loss=dice_logloss2,
                          optimizer=Adam(lr=5e-5),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])       
            model_checkpoint = ModelCheckpoint(path.join(models_folder, 'vgg_model3_weights_{0}_{1}.h5'.format(cities[city_id], it)), monitor='val_dice_coef_rounded', 
                                               save_best_only=True, save_weights_only=True, mode='max')
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=25, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint])
            model.load_weights(path.join(models_folder, 'vgg_model3_weights_{0}_{1}.h5'.format(cities[city_id], it)))
            model.optimizer = Adam(lr=1e-6)
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=5, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint])
            
            np.random.seed(it+111)
            random.seed(it+111)
            tf.set_random_seed(it+111)
        
            model = get_vgg_unet(input_shape, weights=None)
            model.load_weights(path.join(models_folder, 'vgg_model3_weights_{0}_{1}.h5'.format(cities[city_id], it)))
            model.compile(loss=dice_logloss3,
                          optimizer=Adam(lr=1e-4),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])       
            model_checkpoint2 = ModelCheckpoint(path.join(models_folder, 'vgg_model3_weights2_{0}_{1}.h5'.format(cities[city_id], it)), monitor='val_dice_coef_rounded', 
                                               save_best_only=True, save_weights_only=True, mode='max')
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=20, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint2])       
            K.clear_session()
            
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))