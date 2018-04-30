import os
import numpy as np
from skimage.external import tifffile

from tqdm import tqdm

from params import args
from tools.mul_img_utils import stretch_8bit

cities = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum', ]


def calc_stats(img_dir):
    city_mean_value = {}
    for city in cities:
        city_mean = []
        city_mean_img = np.zeros((1300, 1300, 8))
        num_images = 0
        city_dir = os.path.join(img_dir, city + '_Roads_Train', 'MUL-PanSharpen')
        for f in tqdm(os.listdir(city_dir)):
            if f.endswith(".tif"):
                arr = tifffile.imread(os.path.join(city_dir, f))
                image = np.stack([arr[..., 4], arr[..., 2], arr[..., 1], arr[..., 0], arr[..., 3], arr[..., 5], arr[..., 6], arr[..., 7]], axis=-1)
                image = stretch_8bit(image)
                if image is not None:
                    city_mean_img += (image * 255.)
                    num_images += 1

        for i in range(8):
            city_mean.append(np.mean(city_mean_img[..., i] / num_images))
        city_mean_value[city] = city_mean

    return city_mean_value


if __name__ == '__main__':
    print(calc_stats(args.img_dir))
