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
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.callbacks import ModelCheckpoint
from models import dice_coef, dice_logloss2, dice_logloss3, dice_coef_rounded, dice_logloss, get_inception_resnet_v2_unet
import skimage.io
import keras.backend as K

input_shape = (320, 320)

means = [[290.42, 446.84, 591.88, 442.45, 424.66, 418.13, 554.13, 354.34, 566.86],
         [178.33, 260.14, 287.4, 161.44, 211.46, 198.83, 453.27, 228.99, 242.67],
         [357.82, 344.64, 436.76, 452.17, 290.35, 439.7, 440.43, 393.6, 452.5],
         [386.98, 415.74, 601.29, 755.34, 527.79, 729.95, 641, 611.41, 697.17]]
stds = [[75.42, 177.98, 288.81, 250.24, 260.55, 220.09, 299.67, 191.47, 285.25],
        [16.4, 45.69, 79.42, 61.91, 99.64, 81.17, 210.34, 106.31, 80.89],
        [35.23, 58, 89.42, 115.7, 90.45, 109.5, 144.61, 136.77, 99.11],
        [37.9, 59.95, 99.56, 131.14, 96.26, 107.79, 98.77, 92.2, 107.9]]
def preprocess_inputs_std(x, city_id):
    zero_msk = (x == 0)
    x = np.asarray(x, dtype='float32')
    for i in range(9):
        x[..., i] -= means[city_id][i]
        x[..., i] /= stds[city_id][i]
    x[zero_msk] = 0
    return x

masks_folder = r'/wdata/masks_smallest'
models_folder = r'/wdata/nn_models'

cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum']

city_id = -1

all_files = []
all_pan_files = []
all_city_inp = []
all_masks = []

def rotate_image(image, angle, scale):
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result


def batch_data_generator(train_idx, batch_size):
    inputs = []
    outputs = []
    while True:
        np.random.shuffle(train_idx)
        for i in train_idx:
            for j in range(1):
                img = skimage.io.imread(all_files[i], plugin='tifffile')
                pan = skimage.io.imread(all_pan_files[i], plugin='tifffile')
                pan = cv2.resize(pan, (325, 325))
                pan = pan[..., np.newaxis]
                img = np.concatenate([img, pan], axis=2)
                msk = cv2.imread(all_masks[i], cv2.IMREAD_UNCHANGED)[..., 0]
                
                if random.random() > 0.5:
                    scale = 0.9 + random.random() * 0.2
                    angle = random.randint(0, 41) - 24
                    img = rotate_image(img, angle, scale)
                    msk = rotate_image(msk, angle, scale)
                
                x0 = random.randint(0, img.shape[1] - input_shape[1])
                y0 = random.randint(0, img.shape[0] - input_shape[0])
                img = img[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                msk = (msk > 127) * 1
                msk = msk[..., np.newaxis]
                otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1], :]

                if random.random() > 0.5:
                    img = img[:, ::-1, ...]
                    otp = otp[:, ::-1, ...]
                
                rot = random.randrange(4)
                if rot > 0:
                    img = np.rot90(img, k=rot)
                    otp = np.rot90(otp, k=rot)
                    
                inputs.append(img)
                outputs.append(otp)
            
                if len(inputs) == batch_size:
                    inputs = np.asarray(inputs)
                    outputs = np.asarray(outputs, dtype='float')
                    inputs = preprocess_inputs_std(inputs, city_id)
                    yield inputs, outputs
                    inputs = []
                    outputs = []

def val_data_generator(val_idx, batch_size, validation_steps):
    while True:
        inputs = []
        outputs = []
        step_id = 0
        for i in val_idx:
            img0 = skimage.io.imread(all_files[i], plugin='tifffile')
            pan = skimage.io.imread(all_pan_files[i], plugin='tifffile')
            pan = cv2.resize(pan, (325, 325))
            pan = pan[..., np.newaxis]
            img0 = np.concatenate([img0, pan], axis=2)
            msk = cv2.imread(all_masks[i], cv2.IMREAD_UNCHANGED)[..., 0:1]
            msk = (msk > 127) * 1
            for x0, y0 in [(0, 0)]:
                img = img0[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                otp = msk[y0:y0+input_shape[0], x0:x0+input_shape[1], :]
                inputs.append(img)
                outputs.append(otp)
                if len(inputs) == batch_size:
                    step_id += 1
                    inputs = np.asarray(inputs)
                    outputs = np.asarray(outputs, dtype='float')
                    inputs = preprocess_inputs_std(inputs, city_id)
                    yield inputs, outputs
                    inputs = []
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
        for f in sorted(listdir(path.join(d, 'MUL'))):
            if path.isfile(path.join(d, 'MUL', f)) and '.tif' in f:
                img_id = f.split('MUL_')[1].split('.')[0]
                all_files.append(path.join(d, 'MUL', f))
                all_pan_files.append(path.join(d, 'PAN', 'PAN_{0}.tif'.format(img_id)))
                cinp = np.zeros((4,))
                cid = cities.index(img_id.split('_')[2])
                cinp[cid] = 1.0
                all_city_inp.append(cinp)
                all_masks.append(path.join(masks_folder, cities[cid], '{0}{1}'.format(img_id, '.png')))
    all_files = np.asarray(all_files)
    all_pan_files = np.asarray(all_pan_files)
    all_city_inp = np.asarray(all_city_inp)
    all_masks = np.asarray(all_masks)

    batch_size = 16
    it = -1
    kf = KFold(n_splits=4, shuffle=True, random_state=1)
    for all_train_idx, all_val_idx in kf.split(all_files):
        it += 1
        
        if it not in fold_nums:
            continue
        
        for cid in [0, 1, 3]:
            city_id = cid
            train_idx = []
            val_idx = []
            
            for i in all_train_idx:
                if all_city_inp[i][city_id] == 1:
                    train_idx.append(i)
            for i in all_val_idx:
                if all_city_inp[i][city_id] == 1:
                    val_idx.append(i)
                    
            validation_steps = int(len(val_idx) / batch_size)
            steps_per_epoch = int(len(train_idx) / batch_size)
            
            if validation_steps == 0 or steps_per_epoch == 0:
                print("No data for city", cities[city_id])
                continue
            
            print('Training city', cities[city_id], 'fold', it)
            print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)
            
            np.random.seed(it+1)
            random.seed(it+1)
            tf.set_random_seed(it+1)        
    
            print('Training model', it, cities[city_id])
            
            model = get_inception_resnet_v2_unet(input_shape)
                
            model.compile(loss=dice_logloss3,
                          optimizer=SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
            
            model_checkpoint = ModelCheckpoint(path.join(models_folder, 'inception_smallest_model_weights_{0}_{1}.h5'.format(cities[city_id], it)), monitor='val_dice_coef_rounded', 
                                               save_best_only=True, save_weights_only=True, mode='max')
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=15, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint])
            for l in model.layers:
                l.trainable = True
            model.compile(loss=dice_logloss3,
                          optimizer=Adam(lr=1e-3),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])
            
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=30, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint])
            model.optimizer = Adam(lr=2e-4)
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=15, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint])
            
            np.random.seed(it+222)
            random.seed(it+222)
            tf.set_random_seed(it+222)
            model.load_weights(path.join(models_folder, 'inception_smallest_model_weights_{0}_{1}.h5'.format(cities[city_id], it)))
            model.compile(loss=dice_logloss,
                          optimizer=Adam(lr=5e-4),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])       
            model_checkpoint2 = ModelCheckpoint(path.join(models_folder, 'inception_smallest_model_weights2_{0}_{1}.h5'.format(cities[city_id], it)), monitor='val_dice_coef_rounded', 
                                               save_best_only=True, save_weights_only=True, mode='max')
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=20, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint2])
            optimizer=Adam(lr=1e-5)
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=10, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint2])
            
            np.random.seed(it+333)
            random.seed(it+333)
            tf.set_random_seed(it+333)
            model.load_weights(path.join(models_folder, 'inception_smallest_model_weights2_{0}_{1}.h5'.format(cities[city_id], it)))
            model.compile(loss=dice_logloss2,
                          optimizer=Adam(lr=5e-5),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])       
            model_checkpoint3 = ModelCheckpoint(path.join(models_folder, 'inception_smallest_model_weights3_{0}_{1}.h5'.format(cities[city_id], it)), monitor='val_dice_coef_rounded', 
                                               save_best_only=True, save_weights_only=True, mode='max')
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint3])
            
            np.random.seed(it+444)
            random.seed(it+444)
            tf.set_random_seed(it+444)
            model.load_weights(path.join(models_folder, 'inception_smallest_model_weights3_{0}_{1}.h5'.format(cities[city_id], it)))
            model.compile(loss=dice_logloss3,
                          optimizer=Adam(lr=2e-5),
                          metrics=[dice_coef, dice_coef_rounded, metrics.binary_crossentropy])       
            model_checkpoint4 = ModelCheckpoint(path.join(models_folder, 'inception_smallest_model_weights4_{0}_{1}.h5'.format(cities[city_id], it)), monitor='val_dice_coef_rounded', 
                                               save_best_only=True, save_weights_only=True, mode='max')
            model.fit_generator(generator=batch_data_generator(train_idx, batch_size),
                                  epochs=40, steps_per_epoch=steps_per_epoch, verbose=2,
                                  validation_data=val_data_generator(val_idx, batch_size, validation_steps),
                                  validation_steps=validation_steps,
                                  callbacks=[model_checkpoint4])
            K.clear_session()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))