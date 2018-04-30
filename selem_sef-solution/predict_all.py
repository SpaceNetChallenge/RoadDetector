import os
import queue
import threading

import cv2

import gc

from model_name_encoder import decode_params
from tools.stats import mean_bands

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from skimage.external import tifffile
from tensorflow.python.client import device_lib

from tools.mul_img_utils import stretch_8bit

import numpy as np
from keras.preprocessing.image import flip_axis
from tqdm import tqdm

from models import make_model
from params import args
from tools.tiling import generate_tiles, combine_tiles
import keras.backend as K

city_models = {
    'AOI_2_Vegas': [
        '000_vegas_linknet_inception.h5',
        '010_vegas_inception-unet.h5',
        '100_vegas_inception-swish.h5',
        '100_vegas_linknet_inception.h5',
        '100_vegas_linknet_inception_lite.h5',
        '101_vegas_linknet_resnet50.h5'
    ],
    'AOI_3_Paris': [
        '000_paris_linknet_inception.h5',
        '010_paris_inception-unet.h5',
        '100_paris_inception-swish.h5',
        '100_paris_linknet_inception.h5',
        '100_paris_linknet_inception_lite.h5',
        '101_paris_linknet_resnet50.h5'
    ],
    'AOI_4_Shanghai': [
        '000_shanghai_linknet_inception.h5',
        '010_shanghai_inception-unet.h5',
        '100_shanghai_inception-swish.h5',
        '100_shanghai_linknet_inception.h5',
        '100_shanghai_linknet_inception_lite.h5',
        '101_shanghai_linknet_resnet50.h5'
    ],
    'AOI_5_Khartoum': [
        '000_khartoum_linknet_inception.h5',
        '010_khartoum_inception-unet.h5',
        '100_khartoum_inception-swish.h5',
        '100_khartoum_linknet_inception.h5',
        '100_khartoum_linknet_inception_lite.h5',
        '101_khartoum_linknet_resnet50.h5'
    ]
}

folders = [
    'linknet_inception',
    'inception-unet',
    'clahe_inception-swish',
    'clahe_linknet_inception',
    'clahe_linknet_inception_lite',
    'clahe_linknet_resnet50'
]

networks = [
    'linknet_inception',
    'inception-unet',
    'inception-swish',
    'linknet_inception',
    'linknet_inception_lite',
    'linknet_resnet50'
]



def do_tta(x, tta_types):
    if 'hflip' in tta_types:
        x = flip_axis(x, 1)
    if 'vflip' in tta_types:
        x = flip_axis(x, 0)
    return x


def undo_tta(x, tta_types):
    if 'hflip' in tta_types:
        x = flip_axis(x, 1)
    if 'vflip' in tta_types:
        x = flip_axis(x, 0)
    return x


gpus = [x.name for x in device_lib.list_local_devices() if x.name[:4] == '/gpu']

cities = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum']


def predict_all_models():
    dirs = args.dirs_to_process
    for dir in dirs:
        print(dir)
        city_id = next(x for x in cities if x in dir)
        city_dir_name = dir.split("/")[-1]

        models = city_models[city_id]
        for wi, weights in enumerate(models):
            out_dir = folders[wi]
            network = networks[wi]
            clahe, preprocessing_function, stretch_and_mean = decode_params(weights)
            ohe_city_flag = network != "inception-unet"
            channels = 8
            if ohe_city_flag:
                channels = 12
            print("Predict City ", city_id)
            if clahe:
                subdir = 'CLAHE-MUL-PanSharpen'
            else:
                subdir = 'MUL-PanSharpen'
            if clahe:
                test_dir = os.path.join(args.wdata_dir, city_dir_name, subdir)
            else:
                test_dir = os.path.join(dir, subdir)
            print("####TEST DIR######")
            print(test_dir)
            def data_loader(q, ):
                for f in tqdm(sorted(os.listdir(test_dir))):
                    img_path = os.path.join(test_dir, f)
                    arr = tifffile.imread(img_path)

                    image = np.stack([arr[..., 4], arr[..., 2], arr[..., 1], arr[..., 0], arr[..., 3], arr[..., 5], arr[..., 6], arr[..., 7]], axis=-1)
                    if stretch_and_mean:
                        image = stretch_8bit(image) * 255
                        mean_band = mean_bands[city_id]

                        for band in range(len(mean_band)):
                            image[..., band] -= mean_band[band]

                    if ohe_city_flag:
                        ohe_city = np.zeros((image.shape[0], image.shape[1], 4), dtype="float32")
                        ohe_city[..., cities.index(city_id)] = 2047
                        image = np.concatenate([image, ohe_city], axis=-1)
                        image = np.array(image, dtype="float32")

                    if preprocessing_function == 'caffe':
                        image_rgb = image[..., :3]
                        image_bgr = image_rgb[..., ::-1]
                        image[..., :3] = image_bgr
                        if not stretch_and_mean:
                            image = image / 8. - 127.5
                    else:
                        if stretch_and_mean:
                            image = image / 255
                        else:
                            image = image / 1024. - 1
                    padded = np.zeros((1312, 1312, channels), dtype="float32")
                    padded[:1300, :1300, :] = image
                    image = padded
                    q.put((f, image))

                q.put((None, None))


            def predictor(q):
                model = make_model(network, (None, None, channels))
                model.load_weights('trained_models/' + weights)

                while True:
                    f, image = q.get()
                    if image is None:
                        break

                    preds = []
                    for tta in [None, 'hflip', 'vflip', 'hflip+vflip']:
                        ttas = []
                        if tta:
                            ttas = tta.split("+")
                        img = do_tta(image, ttas)
                        pred = model.predict(np.expand_dims(img, axis=0), batch_size=1, verbose=0)[0]
                        pred = undo_tta(pred, ttas)
                        preds.append(pred)
                    mask = np.average(np.array(preds), axis=0)
                    all_masks_dir = "all_masks"
                    os.makedirs(all_masks_dir, exist_ok=True)
                    model_mask_dir = os.path.join(all_masks_dir, out_dir)
                    os.makedirs(model_mask_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(model_mask_dir, f + ".png"), mask * 255)
                del model
                K.clear_session()

            q = queue.Queue(maxsize=5)

            loader = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
            loader.start()

            predictor = threading.Thread(target=predictor, name='Predictor', args=(q,))
            predictor.start()

            loader.join()
            predictor.join()
            gc.collect()


if __name__ == '__main__':
    predict_all_models()
