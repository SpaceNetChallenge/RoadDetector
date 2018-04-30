from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis
import numpy as np


class RandomTransformer:
    def __init__(self, rotation_range=None,
                 height_shift_range=None,
                 width_shift_range=None,
                 shear_range=None,
                 zoom_range=[1., 1.],
                 channel_shift_range=0,
                 horizontal_flip=None, vertical_flip=None, fill_mode='constant', cval=0):
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        self.cval = cval

    def random_transform(self, x, mask):
        """Randomly augment a image tensor and mask.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 2

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range and np.random.random() < 0.3:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            uniform = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            tx = uniform * x.shape[img_row_axis]
            tmx = uniform * mask.shape[img_row_axis]
        else:
            tx = 0
            tmx = 0

        if self.width_shift_range:
            random_uniform = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            ty = random_uniform * x.shape[img_col_axis]
            tmy = random_uniform * mask.shape[img_col_axis]
        else:
            ty = 0
            tmy = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        transform_matrix_mask = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix
            transform_matrix_mask = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            shift_matrix_mask = np.array([[1, 0, tmx],
                                          [0, 1, tmy],
                                          [0, 0, 1]])

            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
            transform_matrix_mask = shift_matrix_mask if transform_matrix_mask is None else np.dot(transform_matrix_mask, shift_matrix_mask)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
            transform_matrix_mask = shear_matrix if transform_matrix_mask is None else np.dot(transform_matrix_mask, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
            transform_matrix_mask = zoom_matrix if transform_matrix_mask is None else np.dot(transform_matrix_mask, zoom_matrix)
        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis, fill_mode=self.fill_mode, cval=self.cval)

        if transform_matrix_mask is not None:
            h, w = mask.shape[img_row_axis], mask.shape[img_col_axis]
            transform_matrix_mask = transform_matrix_offset_center(transform_matrix_mask, h, w)
            mask[:, :, :] = apply_transform(mask[:, :, :], transform_matrix_mask, img_channel_axis, fill_mode='constant', cval=0.)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                mask = flip_axis(mask, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                mask = flip_axis(mask, img_row_axis)

        return x, mask

def do_tta(x, tta_types):
    if 'hflip' in tta_types:
        x = flip_axis(x, 1)
    if 'vflip' in tta_types:
        x = flip_axis(x, 0)
    if 'channel_shift' in tta_types:
        x = random_channel_shift(x, 0.2, 2)
    return x

def undo_tta(x, tta_types):
    if 'hflip' in tta_types:
        x = flip_axis(x, 1)
    if 'vflip' in tta_types:
        x = flip_axis(x, 0)
    return x
