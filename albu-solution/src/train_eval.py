import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os

from augmentations.transforms import get_flips_colors_augmentation

from dataset.reading_image_provider import ReadingImageProvider
from dataset.raw_image import RawImageType
from pytorch_utils.train import train
from pytorch_utils.concrete_eval import FullImageEvaluator
from utils import update_config, get_csv_folds
import argparse
import json
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('--fold', type=int)
parser.add_argument('--training', action='store_true')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
config = Config(**cfg)
skip_folds = []

if args.fold is not None:
    skip_folds = [i for i in range(4) if i != int(args.fold)]

test = not args.training
config = update_config(config, dataset_path=os.path.join(config.dataset_path, 'test' if test else 'train'))

paths = {
    'masks': os.path.join(config.dataset_path, 'masks2m'),
    'images': os.path.join(config.dataset_path, 'images')
}

fn_mapping = {
    'masks': lambda name: os.path.splitext(name)[0] + '.png'
}

image_suffix = 'img'

def train_roads():
    ds = ReadingImageProvider(RawImageType, paths, fn_mapping, image_suffix=image_suffix)

    folds = get_csv_folds('folds4.csv', ds.im_names)
    num_workers = 0 if os.name == 'nt' else 2
    for fold, (train_idx, val_idx) in enumerate(folds):
        if args.fold is not None and int(args.fold) != fold:
            continue
        train(ds, fold, train_idx, val_idx, config, num_workers=num_workers, transforms=get_flips_colors_augmentation())

class RawImageTypePad(RawImageType):
    def finalyze(self, data):
        return self.reflect_border(data, 22)


def eval_roads():
    global config
    rows, cols = 1344, 1344
    config = update_config(config, target_rows=rows, target_cols=cols)
    ds = ReadingImageProvider(RawImageTypePad, paths, fn_mapping, image_suffix=image_suffix)

    folds = [([], list(range(len(ds)))) for i in range(4)]
    num_workers = 0 if os.name == 'nt' else 2
    keval = FullImageEvaluator(config, ds, test=test, flips=3, num_workers=num_workers, border=22)
    for fold, (t, e) in enumerate(folds):
        if args.fold is not None and int(args.fold) != fold:
            continue
        keval.predict(fold, e)


if __name__ == "__main__":
    if test:
        eval_roads()
    else:
        train_roads()
