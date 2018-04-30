import numpy as np
from keras import Model, Input
from keras.layers import UpSampling2D, concatenate, Activation, BatchNormalization, Conv2D, SpatialDropout2D, \
    MaxPooling2D, add
from keras.utils import get_file

import linknet
import resnet50_padding
from inceptionv3_padding import InceptionV3Same
from inceptionv3_padding_swish import InceptionV3SameSwish
from linknet import LinkNet, decoder, deconv_bn_relu
from resnet50_padding import identity_block, conv_block, ResNet50


def conv_bn_relu(prevlayer, filters, prefix, strides=(1, 1), kernel_size=(3, 3)):
    conv = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def resnet_50(input_shape):
    img_input = Input(input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    if input_shape[-1] > 3:
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1_changed')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    print("Loading pretrained weights for Resnet50...")
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            resnet50_padding.WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')
    model = Model(img_input, x)
    model.load_weights(weights_path, by_name=True)
    if input_shape[-1] > 3:
        print("Loading weights for conv1 layer separately for the first 3 channels")
        conv1_weights = np.zeros((7, 7, input_shape[-1], 64), dtype="float32")
        resnet_ori = ResNet50(include_top=False, input_shape=(224, 224, 3))
        conv1_weights[:, :, :3, :] = resnet_ori.get_layer("conv1").get_weights()[0][:, :, :, :]
        # random init
        conv1_weights[:, :, 3:, :] = model.get_layer('conv1_changed').get_weights()[0][:, :, 3:, :]
        bias = resnet_ori.get_layer("conv1").get_weights()[1]
        model.get_layer('conv1_changed').set_weights((conv1_weights, bias))
        model.get_layer('conv1_changed').name = 'conv1'

    return model


def resnet_50_unet(input_shape):
    resnet_base = resnet_50(input_shape=input_shape)

    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_bn_relu(up6, 256, "conv6_1")
    conv6 = conv_bn_relu(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_bn_relu(up7, 192, "conv7_1")
    conv7 = conv_bn_relu(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_bn_relu(up8, 128, "conv8_1")
    conv8 = conv_bn_relu(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_bn_relu(up9, 64, "conv9_1")
    conv9 = conv_bn_relu(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input], axis=-1)
    conv10 = conv_bn_relu(up10, 32, "conv10_1")
    conv10 = conv_bn_relu(conv10, 32, "conv10_2")
    x = SpatialDropout2D(0.5)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    model = Model(resnet_base.input, x)
    return model

def inception_unet(input_shape):
    inception_base = InceptionV3Same(input_shape=input_shape)

    conv1 = inception_base.get_layer("activation_3").output
    conv2 = inception_base.get_layer("activation_5").output
    conv3 = inception_base.get_layer("activation_29").output
    conv4 = inception_base.get_layer("activation_75").output
    conv5 = inception_base.get_layer("mixed10").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_bn_relu(up6, 256, "conv6_1")
    conv6 = conv_bn_relu(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_bn_relu(up7, 192, "conv7_1")
    conv7 = conv_bn_relu(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_bn_relu(up8, 128, "conv8_1")
    conv8 = conv_bn_relu(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_bn_relu(up9, 64, "conv9_1")
    conv9 = conv_bn_relu(conv9, 64, "conv9_2")

    up10 = UpSampling2D()(conv9)
    conv10 = conv_bn_relu(up10, 32, "conv10_1")
    conv10 = conv_bn_relu(conv10, 32, "conv10_2")
    x = SpatialDropout2D(0.5)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="mask")(x)
    model = Model(inception_base.input, x)
    return model


def linknet_resnet_50(input_shape):
    resnet_base = resnet_50(input_shape=input_shape)

    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output
    filters = [256, 512, 1024, 2048]
    feature_scale = 6
    decoder5 = decoder(conv5, filters[3], filters[2], name='decoder5', feature_scale=feature_scale, transposed_conv=True)
    decoder5 = add([decoder5, conv4])
    decoder4 = decoder(decoder5, filters[2], filters[1], name='decoder4', feature_scale=feature_scale, transposed_conv=True)
    decoder4 = add([decoder4, conv3])
    decoder3 = decoder(decoder4, filters[1], filters[0], name='decoder3', feature_scale=feature_scale, transposed_conv=True)
    decoder3 = add([decoder3, conv2])
    decoder2 = decoder(decoder3, filters[0], filters[0], name='decoder2', feature_scale=feature_scale, transposed_conv=True)
    decoder1 = concatenate([conv1, decoder2], axis=-1)
    x = deconv_bn_relu(decoder1, num_channels=32, kernel_size=3, name="decoder1", transposed_conv=True)
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_1')
    x = SpatialDropout2D(0.5)(x)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    model = Model(resnet_base.input, x)
    return model

def linknet_inception(input_shape):
    inception_base = InceptionV3Same(input_shape=input_shape)
    conv1 = inception_base.get_layer("activation_3").output
    conv2 = inception_base.get_layer("activation_5").output
    conv3 = inception_base.get_layer("mixed2").output
    conv4 = inception_base.get_layer("mixed7").output
    conv5 = inception_base.get_layer("mixed10").output
    filters = [192, 288, 768, 2048]
    feature_scale = 4
    decoder5 = decoder(conv5, filters[3], filters[2], name='decoder5', feature_scale=feature_scale, transposed_conv=False)
    decoder5 = add([decoder5, conv4])
    decoder4 = decoder(decoder5, filters[2], filters[1], name='decoder4', feature_scale=feature_scale, transposed_conv=False)
    decoder4 = add([decoder4, conv3])
    decoder3 = decoder(decoder4, filters[1], filters[0], name='decoder3', feature_scale=feature_scale, transposed_conv=False)
    decoder3 = add([decoder3, conv2])
    decoder2 = decoder(decoder3, filters[0], filters[0], name='decoder2', feature_scale=feature_scale, transposed_conv=False)
    decoder1 = concatenate([conv1, decoder2], axis=-1)
    x = deconv_bn_relu(decoder1, num_channels=64, kernel_size=3, name="decoder1", transposed_conv=False)
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_1')
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_2')
    x = SpatialDropout2D(0.5)(x)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    model = Model(inception_base.input, x)
    return model


def swish_linknet_inception(input_shape):
    inception_base = InceptionV3SameSwish(input_shape=input_shape)
    conv1 = inception_base.get_layer("activation_3").output
    conv2 = inception_base.get_layer("activation_5").output
    conv3 = inception_base.get_layer("mixed2").output
    conv4 = inception_base.get_layer("mixed7").output
    conv5 = inception_base.get_layer("mixed10").output
    filters = [192, 288, 768, 2048]
    feature_scale = 6
    decoder5 = decoder(conv5, filters[3], filters[2], name='decoder5', feature_scale=feature_scale, transposed_conv=False, activation='swish')
    decoder5 = add([decoder5, conv4])
    decoder4 = decoder(decoder5, filters[2], filters[1], name='decoder4', feature_scale=feature_scale, transposed_conv=False, activation='swish')
    decoder4 = add([decoder4, conv3])
    decoder3 = decoder(decoder4, filters[1], filters[0], name='decoder3', feature_scale=feature_scale, transposed_conv=False, activation='swish')
    decoder3 = add([decoder3, conv2])
    decoder2 = decoder(decoder3, filters[0], filters[0], name='decoder2', feature_scale=feature_scale, transposed_conv=False, activation='swish')
    decoder1 = concatenate([conv1, decoder2], axis=-1)
    x = deconv_bn_relu(decoder1, num_channels=32, kernel_size=3, name="decoder1", transposed_conv=False, activation='swish')
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_1', activation='swish')
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_2', activation='swish')
    x = SpatialDropout2D(0.5)(x)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    model = Model(inception_base.input, x)
    return model


def linknet_inception_lite(input_shape):
    inception_base = InceptionV3Same(input_shape=input_shape)
    conv1 = inception_base.get_layer("activation_3").output
    conv2 = inception_base.get_layer("activation_5").output
    conv3 = inception_base.get_layer("mixed2").output
    conv4 = inception_base.get_layer("mixed7").output
    conv5 = inception_base.get_layer("mixed10").output
    filters = [192, 288, 768, 2048]
    feature_scale = 6
    decoder5 = decoder(conv5, filters[3], filters[2], name='decoder5', feature_scale=feature_scale, transposed_conv=True)
    decoder5 = add([decoder5, conv4])
    decoder4 = decoder(decoder5, filters[2], filters[1], name='decoder4', feature_scale=feature_scale, transposed_conv=True)
    decoder4 = add([decoder4, conv3])
    decoder3 = decoder(decoder4, filters[1], filters[0], name='decoder3', feature_scale=feature_scale, transposed_conv=True)
    decoder3 = add([decoder3, conv2])
    decoder2 = decoder(decoder3, filters[0], filters[0], name='decoder2', feature_scale=feature_scale, transposed_conv=True)
    decoder1 = concatenate([conv1, decoder2], axis=-1)
    x = deconv_bn_relu(decoder1, num_channels=32, kernel_size=3, name="decoder1", transposed_conv=True)
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_1')
    x = linknet.conv_bn_relu(x, 32, 3, stride=1, padding='same', name='fc_2')
    x = SpatialDropout2D(0.5)(x)
    x = Conv2D(1, (1, 1), activation="sigmoid")(x)
    model = Model(inception_base.input, x)
    return model


def make_model(network, input_shape):
    if network == 'linknet':
        return LinkNet(input_shape, skipConnectionConv1=True)
    if network == 'linknet_resnet50':
        return linknet_resnet_50(input_shape)
    if network == 'resnet-unet':
        return resnet_50_unet(input_shape)
    if network == 'inception-unet':
        return inception_unet(input_shape)
    if network == 'inception-swish':
        return swish_linknet_inception(input_shape)
    if network == 'linknet_inception':
        return linknet_inception(input_shape)
    if network == 'linknet_inception_lite':
        return linknet_inception_lite(input_shape)
    else:
        raise ValueError('unknown network ' + network)


if __name__ == '__main__':
    linknet_inception(input_shape=(1280, 1280, 8)).summary()
