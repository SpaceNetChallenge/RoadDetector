from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Concatenate, UpSampling2D, Activation, SpatialDropout2D, RepeatVector, Reshape
from resnet50_padding_same import ResNet50, identity_block
from resnet50_padding_same import conv_block as resnet_conv_block
from keras.losses import binary_crossentropy
from inception_resnet_v2_padding_same import InceptionResNetV2, inception_resnet_block, conv2d_bn
#from keras.applications.densenet import DenseNet169
from inception_v3_padding_same import InceptionV3, inc_conv2d_bn
bn_axis = 3
channel_axis = bn_axis

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_rounded(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))

def dice_logloss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true, y_pred) * 0.5

def dice_logloss2(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.75 + dice_coef_loss(y_true, y_pred) * 0.25

def dice_logloss3(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.15 + dice_coef_loss(y_true, y_pred) * 0.85

#from https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2
# weight: weighted tensor(same shape with mask image)
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
    weighted_dice_loss(y_true, y_pred, weight)
    return loss

def conv_block(prev, num_filters, kernel=(3, 3), strides=(1, 1), act='relu', prefix=None):
    name = None
    if prefix is not None:
        name = prefix + '_conv'
    conv = Conv2D(num_filters, kernel, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(prev)
    if prefix is not None:
        name = prefix + '_norm'
    conv = BatchNormalization(name=name, axis=bn_axis)(conv)
    if prefix is not None:
        name = prefix + '_act'
    conv = Activation(act, name=name)(conv)
    return conv

def get_resnet_unet(input_shape, weights='imagenet'):
    inp = Input(input_shape + (9,))
    
    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inp)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    conv1 = x
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    enc1 = x
    
    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    enc2 = x
    
    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    enc3 = x
    
    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    enc4 = x
        
    up6 = concatenate([UpSampling2D()(enc4), enc3], axis=-1)
    conv6 = conv_block(up6, 128)
    conv6 = conv_block(conv6, 128)

    up7 = concatenate([UpSampling2D()(conv6), enc2], axis=-1)
    conv7 = conv_block(up7, 96)
    conv7 = conv_block(conv7, 96)

    up8 = concatenate([UpSampling2D()(conv7), enc1], axis=-1)
    conv8 = conv_block(up8, 64)
    conv8 = conv_block(conv8, 64)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, 48)
    conv9 = conv_block(conv9, 48)

    up10 = concatenate([UpSampling2D()(conv9), inp], axis=-1)
    conv10 = conv_block(up10, 32)
    conv10 = conv_block(conv10, 32)
#    conv10 = SpatialDropout2D(0.33)(conv10)
    res = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    model = Model(inp, res)
    
    if weights == 'imagenet':
        resnet = ResNet50(input_shape=input_shape + (3,), include_top=False, weights=weights)
        for i in range(2, len(resnet.layers)-1):
            model.layers[i].set_weights(resnet.layers[i].get_weights())
            model.layers[i].trainable = False
        
    return model

#def get_resnet_unet(input_shape, weights='imagenet'):
#    resnet50 = ResNet50(input_shape=input_shape + (3,), weights=weights, include_top=False)
#    
#    conv_idxs = []
#    for i in range(len(resnet50.layers) - 1):
#        resnet50.layers[i].trainable = False
#        if isinstance(resnet50.layers[i], Activation) and resnet50.layers[i].output_shape[1:3] != resnet50.layers[i+1].output_shape[1:3]:
#            conv_idxs.append(i)
#    
#    resnet_conv1 = resnet50.layers[conv_idxs[0]].output
#    resnet_conv2 = resnet50.layers[conv_idxs[1]].output
#    resnet_conv3 = resnet50.layers[conv_idxs[2]].output
#    resnet_conv4 = resnet50.layers[conv_idxs[3]].output
#    resnet_conv5 = resnet50.layers[conv_idxs[4]].output
#
#    input2 = Input(input_shape + (1,))
#    
#    conv0 = conv_block(input2, 8)
#    conv0 = conv_block(conv0, 8)
#    pool0 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv0)
#    
#    conv1 = conv_block(pool0, 12)
#    conv1 = conv_block(conv1, 12)
#    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
#
#    conv2 = conv_block(pool1, 16)
#    conv2 = conv_block(conv2, 16)
#    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
#
#    conv3 = conv_block(pool2, 32)
#    conv3 = conv_block(conv3, 32)
#    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
#
#    conv4 = conv_block(pool3, 48)
#    conv4 = conv_block(conv4, 48)
#    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)
#    
#    conv5 = conv_block(pool4, 64)
#    conv5 = conv_block(conv5, 64)
#
#    up6 = concatenate([UpSampling2D()(resnet_conv5), resnet_conv4, UpSampling2D()(conv5), conv4], axis=-1)
#    conv6 = conv_block(up6, 192)
#    conv6 = conv_block(conv6, 192)
#
#    up7 = concatenate([UpSampling2D()(conv6), resnet_conv3, conv3], axis=-1)
#    conv7 = conv_block(up7, 128)
#    conv7 = conv_block(conv7, 128)
#
#    up8 = concatenate([UpSampling2D()(conv7), resnet_conv2, conv2], axis=-1)
#    conv8 = conv_block(up8, 96)
#    conv8 = conv_block(conv8, 96)
#
#    up9 = concatenate([UpSampling2D()(conv8), resnet_conv1, conv1], axis=-1)
#    conv9 = conv_block(up9, 64)
#    conv9 = conv_block(conv9, 64)
#
#    up10 = concatenate([UpSampling2D()(conv9), resnet50.input, conv0, input2], axis=-1)
#    conv10 = conv_block(up10, 32)
#    conv10 = conv_block(conv10, 32)
#    conv10 = SpatialDropout2D(0.33)(conv10)
#    res = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
#    model = Model([resnet50.input, input2], res)
#    return model


def get_vgg_unet(input_shape, weights='imagenet'):  
    input1 = Input(input_shape + (8,))
    
    input2 = Input((4,))
    
    conv1 = conv_block(input1, 64, prefix='conv1')
    conv1 = conv_block(conv1, 64, prefix='conv1_2')
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = conv_block(pool1, 128, prefix='conv2')
    conv2 = conv_block(conv2, 128, prefix='conv2_2')
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = conv_block(pool2, 256, prefix='conv3')
    conv3 = conv_block(conv3, 256, prefix='conv3_2')
    conv3 = conv_block(conv3, 256, prefix='conv3_3')
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = conv_block(pool3, 512, prefix='conv4')
    conv4 = conv_block(conv4, 512, prefix='conv4_2')
    conv4 = conv_block(conv4, 512, prefix='conv4_3')
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)
    
    conv5 = conv_block(pool4, 512, prefix='conv5')
    conv5 = conv_block(conv5, 512, prefix='conv5_2')
    conv5 = conv_block(conv5, 512, prefix='conv5_3')
    pool5 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv5)

#    rp0 = RepeatVector(pool5.shape[1].value * pool5.shape[2].value)(input2)
#    rp0 = Reshape((pool5.shape[1].value, pool5.shape[2].value, 4))(rp0)
#    rp0 = concatenate([pool5, rp0], axis=-1)
#    rp0
    conv5_2 = conv_block(pool5, 512, prefix='conv5_2_0')
    conv5_2 = conv_block(conv5_2, 512, prefix='conv5_2_1')
    conv5_2 = conv_block(conv5_2, 512, prefix='conv5_2_2')
    
    up6_0 = concatenate([UpSampling2D()(conv5_2), conv5], axis=-1)
    conv6_0 = conv_block(up6_0, 128)
    conv6_0 = conv_block(conv6_0, 128)
    
    up6 = concatenate([UpSampling2D()(conv6_0), conv4], axis=-1)
    conv6 = conv_block(up6, 96)
    conv6 = conv_block(conv6, 96)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block(up7, 64)
    conv7 = conv_block(conv7, 64)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block(up8, 48)
    conv8 = conv_block(conv8, 48)

    rp1 = RepeatVector(input_shape[0] * input_shape[1])(input2)
    rp1 = Reshape(input_shape + (4,))(rp1)
    
    up9 = concatenate([UpSampling2D()(conv8), conv1, rp1], axis=-1)
    conv9 = conv_block(up9, 32)
    conv9 = conv_block(conv9, 32)
#    conv9 = SpatialDropout2D(0.33)(conv9)
    res = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model([input1, input2], res)
    
    if weights == 'imagenet':
        vgg16 = VGG16(input_shape=input_shape + (3,), weights=weights, include_top=False)
        vgg_l = vgg16.get_layer('block1_conv1')
        l = model.get_layer('conv1_conv')
        w0 = vgg_l.get_weights()
        w = l.get_weights()
        w[0][:, :, [1, 2, 4], :] = 0.8 * w0[0]
        w[0][:, :, [0, 3, 5], :] = 0.12 * w0[0]
        w[0][:, :, [0, 6, 7], :] = 0.12 * w0[0]
        w[1] = w0[1]
        l.set_weights(w)
        vgg_l = vgg16.get_layer('block1_conv2')
        l = model.get_layer('conv1_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block2_conv1')
        l = model.get_layer('conv2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block2_conv2')
        l = model.get_layer('conv2_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block3_conv1')
        l = model.get_layer('conv3_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block3_conv2')
        l = model.get_layer('conv3_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block3_conv3')
        l = model.get_layer('conv3_3_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block4_conv1')
        l = model.get_layer('conv4_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block4_conv2')
        l = model.get_layer('conv4_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block4_conv3')
        l = model.get_layer('conv4_3_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block5_conv1')
        l = model.get_layer('conv5_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block5_conv2')
        l = model.get_layer('conv5_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block5_conv3')
        l = model.get_layer('conv5_3_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l.trainable = False
        
    return model


def get_vgg_unet2(input_shape, weights='imagenet'):  
    input1 = Input(input_shape + (8,))
    
    conv1 = conv_block(input1, 64, prefix='conv1')
    conv1 = conv_block(conv1, 64, prefix='conv1_2')
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = conv_block(pool1, 128, prefix='conv2')
    conv2 = conv_block(conv2, 128, prefix='conv2_2')
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = conv_block(pool2, 256, prefix='conv3')
    conv3 = conv_block(conv3, 256, prefix='conv3_2')
    conv3 = conv_block(conv3, 256, prefix='conv3_3')
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = conv_block(pool3, 512, prefix='conv4')
    conv4 = conv_block(conv4, 512, prefix='conv4_2')
    conv4 = conv_block(conv4, 512, prefix='conv4_3')
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)
    
    conv5 = conv_block(pool4, 512, prefix='conv5')
    conv5 = conv_block(conv5, 512, prefix='conv5_2')
    conv5 = conv_block(conv5, 512, prefix='conv5_3')
    pool5 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv5)

#    rp0 = RepeatVector(pool5.shape[1].value * pool5.shape[2].value)(input2)
#    rp0 = Reshape((pool5.shape[1].value, pool5.shape[2].value, 4))(rp0)
#    rp0 = concatenate([pool5, rp0], axis=-1)
#    rp0
#    conv5_2 = conv_block(pool5, 128, kernel=(3, 3), prefix='conv5_2_0')
#    conv5_2 = conv_block(conv5_2, 128, kernel=(3, 3), prefix='conv5_2_1')
#    pool5_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv5_2)
#    
#    conv5_3 = conv_block(pool5_2, 128, kernel=(3, 3), prefix='conv5_3_0')
#    conv5_3 = conv_block(conv5_3, 128, kernel=(3, 3), prefix='conv5_3_1')
    
#    conv5_2 = conv_block(conv5_2, 128, prefix='conv5_2_2')
    
#    up6_00 = concatenate([UpSampling2D()(conv5_3), conv5_2], axis=-1)
#    conv6_00 = conv_block(up6_00, 128)
#    conv6_00 = conv_block(conv6_00, 128)
#    
#    up6_0 = concatenate([UpSampling2D()(conv6_00), conv5], axis=-1)
#    conv6_0 = conv_block(up6_0, 128)
#    conv6_0 = conv_block(conv6_0, 128)
    
    conv5_2 = conv_block(pool5, 512, prefix='conv5_2_0')
    conv5_2 = conv_block(conv5_2, 512, prefix='conv5_2_1')
    conv5_2 = conv_block(conv5_2, 512, prefix='conv5_2_2')
    
    up6_0 = concatenate([UpSampling2D()(conv5_2), conv5], axis=-1)
    conv6_0 = conv_block(up6_0, 128)
    conv6_0 = conv_block(conv6_0, 128)
    
    up6 = concatenate([UpSampling2D()(conv6_0), conv4], axis=-1) #conv6_0 conv5
    conv6 = conv_block(up6, 96)
    conv6 = conv_block(conv6, 96)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block(up7, 64)
    conv7 = conv_block(conv7, 64)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block(up8, 48)
    conv8 = conv_block(conv8, 48)
    
    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, 32)
    conv9 = conv_block(conv9, 32)
#    conv9 = SpatialDropout2D(0.33)(conv9)
    res = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(input1, res)
    
    if weights == 'imagenet':
        vgg16 = VGG16(input_shape=input_shape + (3,), weights=weights, include_top=False)
        vgg_l = vgg16.get_layer('block1_conv1')
        l = model.get_layer('conv1_conv')
        w0 = vgg_l.get_weights()
        w = l.get_weights()
        w[0][:, :, [1, 2, 4], :] = 0.8 * w0[0]
        w[0][:, :, [0, 3, 5], :] = 0.1 * w0[0]
        w[0][:, :, [0, 6, 7], :] = 0.1 * w0[0]
        w[1] = w0[1]
        l.set_weights(w)
        vgg_l = vgg16.get_layer('block1_conv2')
        l = model.get_layer('conv1_2_conv')
        l.set_weights(vgg_l.get_weights())
        
        vgg_l = vgg16.get_layer('block2_conv1')
        l = model.get_layer('conv2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block2_conv2')
        l = model.get_layer('conv2_2_conv')
        l.set_weights(vgg_l.get_weights())
        
        vgg_l = vgg16.get_layer('block3_conv1')
        l = model.get_layer('conv3_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block3_conv2')
        l = model.get_layer('conv3_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block3_conv3')
        l = model.get_layer('conv3_3_conv')
        l.set_weights(vgg_l.get_weights())
        
        vgg_l = vgg16.get_layer('block4_conv1')
        l = model.get_layer('conv4_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block4_conv2')
        l = model.get_layer('conv4_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block4_conv3')
        l = model.get_layer('conv4_3_conv')
        l.set_weights(vgg_l.get_weights())
        
        vgg_l = vgg16.get_layer('block5_conv1')
        l = model.get_layer('conv5_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block5_conv2')
        l = model.get_layer('conv5_2_conv')
        l.set_weights(vgg_l.get_weights())
        vgg_l = vgg16.get_layer('block5_conv3')
        l = model.get_layer('conv5_3_conv')
        l.set_weights(vgg_l.get_weights())
        
    return model


def get_vgg_unet_small(input_shape, weights='imagenet', freeze=False):  
    input1 = Input(input_shape + (9,))
    
    conv1 = conv_block(input1, 64, prefix='conv1')
    conv1 = conv_block(conv1, 64, prefix='conv1_2')
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = conv_block(pool1, 128, prefix='conv2')
    conv2 = conv_block(conv2, 128, prefix='conv2_2')
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = conv_block(pool2, 256, prefix='conv3')
    conv3 = conv_block(conv3, 256, prefix='conv3_2')
    conv3 = conv_block(conv3, 256, prefix='conv3_3')
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = conv_block(pool3, 512, prefix='conv4')
    conv4 = conv_block(conv4, 512, prefix='conv4_2')
    conv4 = conv_block(conv4, 512, prefix='conv4_3')
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)
    
    conv5 = conv_block(pool4, 512, prefix='conv5')
    conv5 = conv_block(conv5, 512, prefix='conv5_2')
    conv5 = conv_block(conv5, 512, prefix='conv5_3')
    
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block(up6, 96)
    conv6 = conv_block(conv6, 96)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block(up7, 64)
    conv7 = conv_block(conv7, 64)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block(up8, 48)
    conv8 = conv_block(conv8, 48)
    
    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, 32)
    conv9 = conv_block(conv9, 32)
    res = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(input1, res)
    
    if weights == 'imagenet':
        vgg16 = VGG16(input_shape=input_shape + (3,), weights=weights, include_top=False)
        vgg_l = vgg16.get_layer('block1_conv1')
        l = model.get_layer('conv1_conv')
        w0 = vgg_l.get_weights()
        w = l.get_weights()
        w[0][:, :, [1, 2, 4], :] = 0.8 * w0[0]
        w[0][:, :, [0, 3, 5], :] = 0.1 * w0[0]
        w[0][:, :, [6, 7, 8], :] = 0.1 * w0[0]
        w[1] = w0[1]
        l.set_weights(w)
        vgg_l = vgg16.get_layer('block1_conv2')
        l = model.get_layer('conv1_2_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
            
        vgg_l = vgg16.get_layer('block2_conv1')
        l = model.get_layer('conv2_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block2_conv2')
        l = model.get_layer('conv2_2_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block3_conv1')
        l = model.get_layer('conv3_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block3_conv2')
        l = model.get_layer('conv3_2_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block3_conv3')
        l = model.get_layer('conv3_3_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block4_conv1')
        l = model.get_layer('conv4_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block4_conv2')
        l = model.get_layer('conv4_2_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block4_conv3')
        l = model.get_layer('conv4_3_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        
        vgg_l = vgg16.get_layer('block5_conv1')
        l = model.get_layer('conv5_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block5_conv2')
        l = model.get_layer('conv5_2_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        vgg_l = vgg16.get_layer('block5_conv3')
        l = model.get_layer('conv5_3_conv')
        l.set_weights(vgg_l.get_weights())
        if freeze:
            vgg_l.trainable = False
        
    return model

def get_inception_resnet_v2_unet(input_shape, weights='imagenet'):
    inp = Input(input_shape + (9,))
    
    # Stem block: 35 x 35 x 192
    x = conv2d_bn(inp, 32, 3, strides=2, padding='same')
    x = conv2d_bn(x, 32, 3, padding='same')
    x = conv2d_bn(x, 64, 3)
    conv1 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = conv2d_bn(x, 80, 1, padding='same')
    x = conv2d_bn(x, 192, 3, padding='same')
    conv2 = x
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)
    conv3 = x
    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)
    conv4 = x
    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='same')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')
    conv5 = x
    
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block(up6, 128)
    conv6 = conv_block(conv6, 128)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block(up7, 96)
    conv7 = conv_block(conv7, 96)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block(up8, 64)
    conv8 = conv_block(conv8, 64)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, 48)
    conv9 = conv_block(conv9, 48)

    up10 = concatenate([UpSampling2D()(conv9), inp], axis=-1)
    conv10 = conv_block(up10, 32)
    conv10 = conv_block(conv10, 32)
#    conv10 = SpatialDropout2D(0.33)(conv10)
    res = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    model = Model(inp, res)
    
    if weights == 'imagenet':
        inception_resnet_v2 = InceptionResNetV2(weights=weights, include_top=False, input_shape=input_shape + (3,))
        for i in range(2, len(inception_resnet_v2.layers)-1):
            model.layers[i].set_weights(inception_resnet_v2.layers[i].get_weights())
            model.layers[i].trainable = False
        
    return model

def get_inception_v3_unet(input_shape, weights='imagenet'):
    inp = Input(input_shape + (9,))
    
    x = inc_conv2d_bn(inp, 32, 3, 3, strides=(2, 2), padding='same')
    x = inc_conv2d_bn(x, 32, 3, 3, padding='same')
    x = inc_conv2d_bn(x, 64, 3, 3)
    conv1 = x
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inc_conv2d_bn(x, 80, 1, 1, padding='same')
    x = inc_conv2d_bn(x, 192, 3, 3, padding='same')
    conv2 = x
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = inc_conv2d_bn(x, 64, 1, 1)

    branch5x5 = inc_conv2d_bn(x, 48, 1, 1)
    branch5x5 = inc_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = inc_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = inc_conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = inc_conv2d_bn(x, 64, 1, 1)

    branch5x5 = inc_conv2d_bn(x, 48, 1, 1)
    branch5x5 = inc_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = inc_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = inc_conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = inc_conv2d_bn(x, 64, 1, 1)

    branch5x5 = inc_conv2d_bn(x, 48, 1, 1)
    branch5x5 = inc_conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = inc_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = inc_conv2d_bn(branch_pool, 64, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    conv3 = x
    # mixed 3: 17 x 17 x 768
    branch3x3 = inc_conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='same')

    branch3x3dbl = inc_conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = inc_conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='same')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = inc_conv2d_bn(x, 192, 1, 1)

    branch7x7 = inc_conv2d_bn(x, 128, 1, 1)
    branch7x7 = inc_conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = inc_conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = inc_conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = inc_conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = inc_conv2d_bn(x, 192, 1, 1)

        branch7x7 = inc_conv2d_bn(x, 160, 1, 1)
        branch7x7 = inc_conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = inc_conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = inc_conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = inc_conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = inc_conv2d_bn(x, 192, 1, 1)

    branch7x7 = inc_conv2d_bn(x, 192, 1, 1)
    branch7x7 = inc_conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = inc_conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = inc_conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = inc_conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = inc_conv2d_bn(branch_pool, 192, 1, 1)
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    conv4 = x
    # mixed 8: 8 x 8 x 1280
    branch3x3 = inc_conv2d_bn(x, 192, 1, 1)
    branch3x3 = inc_conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='same')

    branch7x7x3 = inc_conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = inc_conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = inc_conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = inc_conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='same')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = inc_conv2d_bn(x, 320, 1, 1)

        branch3x3 = inc_conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = inc_conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = inc_conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = inc_conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = inc_conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = inc_conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = inc_conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = inc_conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
        
    conv5 = x
    
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block(up6, 160)
    conv6 = conv_block(conv6, 160)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block(up7, 128)
    conv7 = conv_block(conv7, 128)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block(up8, 96)
    conv8 = conv_block(conv8, 96)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block(up9, 64)
    conv9 = conv_block(conv9, 64)

    up10 = concatenate([UpSampling2D()(conv9), inp], axis=-1)
    conv10 = conv_block(up10, 48)
    conv10 = conv_block(conv10, 48)
    res = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    model = Model(inp, res)
    
    if weights == 'imagenet':
        inception_v3 = InceptionV3(weights=weights, include_top=False, input_shape=input_shape + (3,))
        for i in range(2, len(inception_v3.layers)):
            model.layers[i].set_weights(inception_v3.layers[i].get_weights())
            model.layers[i].trainable = False
        
    return model