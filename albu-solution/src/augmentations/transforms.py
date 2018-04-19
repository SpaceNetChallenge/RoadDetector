"""
usage:
data = {'image': im, 'mask': mask, 'whatever': hi}
augs = Compose([VerticalFlip(), HorizontalFlip()])
data = augs(**data)
now augmentations are applied to data
every augmentation is only applied to fields defined as targets, all other are passed through
"""
import random
import numpy as np
from .composition import Compose
from . import functional as F

class BasicTransform:
    """
    base class for all transforms
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, **kwargs):
        """
        override it if you need to apply different transforms to data
        for example you can define method apply_to_boxes and apply it to bounding boxes
        """
        if random.random() < self.prob:
            params = self.get_params()
            return {k: self.apply(a, **params) if k in self.targets else a for k, a in kwargs.items()}
        return kwargs

    def apply(self, img, **params):
        """
        override this method with transform you need to apply
        """
        raise NotImplementedError

    def get_params(self):
        """
        dict of transform parameters for apply
        """
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError


class DualTransform(BasicTransform):
    """
    transfrom for segmentation task
    """
    @property
    def targets(self):
        return 'image', 'mask'


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """
    @property
    def targets(self):
        return 'image'


class VerticalFlip(DualTransform):
    def apply(self, img, **params):
        return F.vflip(img)


class HorizontalFlip(DualTransform):
    def apply(self, img, **params):
        return F.hflip(img)


class RandomFlip(DualTransform):
    def apply(self, img, flipCode=0):
        return F.flip(img, flipCode)

    def get_params(self):
        return {'flipCode': random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, times=0):
        return F.rot90(img, times)

    def get_params(self):
        return {'times': random.randint(0, 4)}


class RandomRotate(DualTransform):
    def __init__(self, angle_limit=90, prob=.5):
        super().__init__(prob)
        self.angle_limit = angle_limit

    def apply(self, img, angle=0):
        return F.rotate(img, angle)

    def get_params(self):
        return {'angle': random.uniform(-self.angle_limit, self.angle_limit)}


class RandomShiftScaleRotate(DualTransform):
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        super().__init__(prob)
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        return F.shift_scale_rotate(img, angle, scale, dx, dy)

    def get_params(self):
        return {'angle': random.uniform(-self.rotate_limit, self.rotate_limit),
                'scale': random.uniform(1-self.scale_limit, 1+self.scale_limit),
                'dx': round(random.uniform(-self.shift_limit, self.shift_limit)),
                'dy': round(random.uniform(-self.shift_limit, self.shift_limit))}


class CenterCrop(DualTransform):
    def __init__(self, height, width, prob=0.5):
        super().__init__(prob)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class Jitter_HSV(ImageOnlyTransform):
    def __init__(self, hue_shift_limit=(-20, 20), sat_shift_limit=(-35, 35), val_shift_limit=(-35, 35), prob=0.5):
        super().__init__(prob)
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0):
        hue_shift, sat_shift, val_shift = F.fix_shift_values(image, hue_shift, sat_shift, val_shift)
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return{'hue_shift': np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
               'sat_shift': np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
               'val_shift': np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])}


class Jitter_channels(ImageOnlyTransform):
    def __init__(self, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20), prob=0.5):
        super().__init__(prob)
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0):
        r_shift, g_shift, b_shift = F.fix_shift_values(image, r_shift, g_shift, b_shift)
        return F.shift_channels(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return{'r_shift': np.random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
               'g_shift': np.random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
               'b_shift': np.random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])}


class RandomBlur(ImageOnlyTransform):
    def __init__(self, blur_limit=7, prob=.5):
        super().__init__(prob)
        self.blur_limit = blur_limit

    def apply(self, image, ksize=3):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': np.random.choice(np.arange(3, self.blur_limit + 1, 2))
        }


class RandomCLAHE(ImageOnlyTransform):
    def __init__(self, clipLimit=4.0, tileGridSize=(8, 8), prob=0.5):
        super().__init__(prob)
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def apply(self, img, clipLimit=2):
        return F.clahe(img, clipLimit, self.tileGridSize)

    def get_params(self):
        return {"clipLimit": np.random.uniform(1, self.clipLimit)}

class ChannelShuffle(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.channel_shuffle(img)

class InvertImg(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.invert(img)

class ToTensor(BasicTransform):
    def __init__(self, num_classes=1):
        super().__init__(prob=1.)
        self.num_classes = num_classes

    def __call__(self, **kwargs):
        kwargs.update({'image': F.img_to_tensor(kwargs['image'])})
        if 'mask' in kwargs:
            kwargs.update({'mask': F.mask_to_tensor(kwargs['mask'], self.num_classes)})
        return kwargs


def get_flips_colors_augmentation(prob=.5):
    """
    you can compose transforms and apply them sequentially
    """
    return Compose([
        RandomFlip(0.5),
        Transpose(0.5),
        RandomShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=30, prob=.75),
        Jitter_HSV()
    ])

