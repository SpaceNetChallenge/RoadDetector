import keras.backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.losses import mean_squared_error
import numpy as np


def dice_coef_clipped(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = _to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = K.tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = K.tf.log(prediction_tensor / (1 - prediction_tensor))

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.tf.cast(
            K.tf.sigmoid(prediction_tensor) > 0.5, K.tf.float32)
    return K.mean(K.tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))


def ceneterline_loss(y, p):
    centerline = get_eroded(y)
    p = p * centerline
    return dice_coef_loss_bce(centerline, p, dice=0.5, bce=0.5, bootstrapping='soft', alpha=1)

def get_eroded(y):
    structure = np.asarray(np.zeros((3, 3, 1)), dtype="float32")
    filter = K.tf.constant(structure, dtype="float32")
    erosion = K.tf.nn.erosion2d(y, strides=[1, 1, 1, 1], rates=[1, 5, 5, 1], kernel=filter, padding='SAME')
    return erosion

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def mae_vgg(y_true, y_pred):
    y_true = K.permute_dimensions(y_true, (0, 3, 1, 2))
    y_pred = K.permute_dimensions(y_pred, (0, 3, 1, 2))

    y_true = K.reshape(y_true, (K.shape(y_true)[0], K.shape(y_true)[1], K.shape(y_true)[2] * K.shape(y_true)[3]))
    y_pred = K.reshape(y_pred, (K.shape(y_pred)[0], K.shape(y_pred)[1], K.shape(y_pred)[2] * K.shape(y_pred)[3]))

    return K.mean(mean_squared_error(y_true, y_pred))


def make_loss(loss_name):
    if loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5, bootstrapping='soft', alpha=1)

        return loss
    if loss_name == 'bce':
        return binary_crossentropy
    else:
        ValueError("Unknown loss.")
