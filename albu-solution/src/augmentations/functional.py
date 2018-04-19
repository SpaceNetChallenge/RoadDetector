import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import math
from functools import wraps


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    """
    wrapper to clip results of transform to image dtype value range
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)
    return wrapped_function


def fix_shift_values(img, *args):
    """
    shift values are normally specified in uint, but if your data is float - you need to remap values
    """
    if img.dtype == np.float32:
        return list(map(lambda x: x / 255, args))
    return args


def vflip(img):
    return cv2.flip(img, 0)


def hflip(img):
    return cv2.flip(img, 1)


def flip(img, code):
    return cv2.flip(img, code)


def transpose(img):
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


def rot90(img, times):
    img = np.rot90(img, times)
    return np.ascontiguousarray(img)


def rotate(img, angle):
    """
    rotate image on specified angle
    :param angle: angle in degrees
    """
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img = cv2.warpAffine(img, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img


def shift_scale_rotate(img, angle, scale, dx, dy):
    """
    :param angle: in degrees
    :param scale: relative scale
    """
    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx*width, height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    return img


def center_crop(img, height, width):
    h, w, c = img.shape
    dy = (h-height)//2
    dx = (w-width)//2
    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]
    return img


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    maxval = np.max(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    h, s, v = cv2.split(img)
    h = cv2.add(h, hue_shift)
    h = np.where(h < 0, maxval - h, h)
    h = np.where(h > maxval, h - maxval, h)
    h = h.astype(dtype)
    s = clip(cv2.add(s, sat_shift), dtype, maxval)
    v = clip(cv2.add(v, val_shift), dtype, maxval)
    img = cv2.merge((h, s, v)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def shift_channels(img, r_shift, g_shift, b_shift):
    img[...,0] = clip(img[...,0] + r_shift, np.uint8, 255)
    img[...,1] = clip(img[...,1] + g_shift, np.uint8, 255)
    img[...,2] = clip(img[...,2] + b_shift, np.uint8, 255)
    return img


def clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output


def blur(img, ksize):
    return cv2.blur(img, (ksize, ksize))


def invert(img):
    return 255 - img


def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img


def img_to_tensor(im):
    return np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32)


def mask_to_tensor(mask, num_classes):
    if num_classes > 1:
        mask = img_to_tensor(mask)
    else:
        mask = np.expand_dims(mask / (255. if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    return mask

