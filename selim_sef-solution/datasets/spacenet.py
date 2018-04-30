import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import pygeoif
from keras.preprocessing.image import Iterator, img_to_array, load_img
from skimage.external.tifffile import tifffile

from tools.mul_img_utils import stretch_8bit
from tools.stats import mean_bands

cities = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum']

os.makedirs("masks", exist_ok=True)
def get_city_id(city_dir):
    return next(x for x in cities if x in city_dir)


def generate_ids(city_dirs, clahe):
    print("Generate image ids for dirs:  " + str(city_dirs))
    ids = []
    for city_dir in city_dirs:
        city_id = get_city_id(city_dir)
        subdir = "MUL-PanSharpen"
        #if clahe:
        #    subdir = "CLAHE-MUL-PanSharpen"
        mul_dir = os.path.join(city_dir, subdir)
        for f in os.listdir(mul_dir):
            if f.endswith(".tif"):
                ids.append((city_id, f.split(".tif")[0].split("MUL-PanSharpen_")[1]))
    return sorted(ids)


def get_groundtruth(city_dirs):
    gt = {}
    for city_dir in city_dirs:
        summary_dir = os.path.join(city_dir, 'summaryData')

        path_to_csv = os.path.join(summary_dir, city_dir.split("/")[-1] + ".csv")
        print("Processing CSV: " + path_to_csv)
        matrix = pd.read_csv(path_to_csv).as_matrix()
        for line in matrix:
            id = line[0]
            linestring = line[1]
            gt_lines = gt.get(id, [])
            gt_lines.append(linestring)
            gt[id] = gt_lines
    return gt


class MULSpacenetDataset(Iterator):
    def __init__(self,
                 data_dirs,
                 wdata_dir,
                 image_ids,
                 crop_shape,
                 preprocessing_function='tf',
                 random_transformer=None,
                 batch_size=8,
                 crops_per_image=3,
                 thickness=16,
                 shuffle=True,
                 image_name_template=None,
                 masks_dict=None,
                 stretch_and_mean=None,
                 ohe_city=True,
                 clahe=False,
                 seed=None):
        self.data_dirs = data_dirs
        self.image_ids = image_ids
        self.wdata_dir = wdata_dir
        self.clahe = clahe
        self.image_name_template = image_name_template
        self.masks_dict = masks_dict
        self.random_transformer = random_transformer
        self.crop_shape = crop_shape
        self.stretch_and_mean = stretch_and_mean
        self.ohe_city = ohe_city
        self.crops_per_image = crops_per_image
        self.preprocessing_function = preprocessing_function
        self.thickness = thickness
        if seed is None:
            seed = np.uint32(time.time() * 1000)

        super(MULSpacenetDataset, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    def transform_mask(self, mask, image):
        mask[np.where(np.all(image[..., :3] == (0, 0, 0), axis=-1))] = 0
        return mask

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            city, id = self.image_ids[image_index]

            for data_dir in self.data_dirs:
                city_dir_name = data_dir.split("/")[-1]
                if city in data_dir:
                    img_name = self.image_name_template.format(id=id)
                    if self.clahe:
                        data_dir = os.path.join(self.wdata_dir, city_dir_name)
                        path = os.path.join(data_dir, img_name)
                    else:
                        path = os.path.join(data_dir, img_name)
                    break

            arr = tifffile.imread(path)

            image = np.stack([arr[..., 4], arr[..., 2], arr[..., 1], arr[..., 0], arr[..., 3], arr[..., 5], arr[..., 6], arr[..., 7]], axis=-1)
            if self.stretch_and_mean:
                image = stretch_8bit(image) * 255
            if self.ohe_city:
                ohe_city = np.zeros((image.shape[0], image.shape[1], 4), dtype="float32")
                ohe_city[..., cities.index(city)] = 2047
                image = np.concatenate([image, ohe_city], axis=-1)
                image = np.array(image, dtype="float32")

            lines = self.masks_dict[id]
            mask = np.zeros((image.shape[0], image.shape[1], 1))
            # lines in wkt format, pygeoif
            if os.path.exists("masks/" + id + ".png"):
                mask = img_to_array(load_img("masks/" + id + ".png", grayscale=True)) / 255.
            else:
                mask = np.zeros((image.shape[0], image.shape[1], 1))
                # lines in wkt format, pygeoif
                for line in lines:
                    if "LINESTRING EMPTY" == line:
                        continue
                    points = pygeoif.from_wkt(line).coords
                    for i in range(1, len(points)):
                        pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                        pt2 = (int(points[i][0]), int(points[i][1]))
                        cv2.line(mask, pt1, pt2, (1,), thickness=self.thickness)
                cv2.imwrite("masks/" + id + ".png", mask * 255)
            ori_height = image.shape[0]
            ori_width = image.shape[1]

            mask = self.transform_mask(mask, image)
            if self.random_transformer is not None:
                image, mask = self.random_transformer.random_transform(image, mask)

            if self.stretch_and_mean:
                mean_band = mean_bands[city]

                for band in range(len(mean_band)):
                    image[..., band] -= mean_band[band]
            if self.crop_shape is not None:
                crops = 0
                tries = 0
                while crops < self.crops_per_image:
                    tries += 1
                    if self.random_transformer is None:
                        y_start = (ori_height - self.crop_shape[0]) // 2
                        x_start = (ori_width - self.crop_shape[1]) // 2
                    else:
                        y_start = random.randint(0, ori_height - self.crop_shape[0] - 1)
                        x_start = random.randint(0, ori_width - self.crop_shape[1] - 1)
                    y_end = y_start + self.crop_shape[0]
                    x_end = x_start + self.crop_shape[1]
                    crop_image = image[y_start:y_end, x_start:x_end, :]
                    crop_mask = mask[y_start:y_end, x_start:x_end, :]
                    if self.random_transformer is None:
                        batch_x.append(crop_image)
                        batch_y.append(crop_mask)
                        crops += 1
                    elif np.count_nonzero(crop_image) > 100 or tries > 20:
                        batch_x.append(crop_image)
                        batch_y.append(crop_mask)
                        crops += 1
            else:
                batch_x.append(image)
                batch_y.append(mask)
        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        if self.preprocessing_function == 'caffe':
            batch_x_rgb = batch_x[..., :3]
            batch_x_bgr = batch_x_rgb[..., ::-1]
            batch_x[..., :3] = batch_x_bgr
            if not self.stretch_and_mean:
                batch_x = batch_x / 8. - 127.5
        else:
            if self.stretch_and_mean:
                batch_x = batch_x / 255
            else:
                batch_x = batch_x / 1024. - 1
        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    def transform_batch_x(self, batch_x):
        return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
