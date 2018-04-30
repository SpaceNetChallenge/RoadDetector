import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, add, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import SpatialDropout2D
from keras.layers.merge import concatenate
from keras.models import Model


def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', activation='relu'):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation(activation, name=name + '_relu')(x)
    return x


def deconv_bn_relu(input, num_channels, kernel_size, name, transposed_conv, activation='relu'):
    if transposed_conv:
        x = Conv2DTranspose(num_channels, kernel_size=(4, 4), strides=(2,2), padding="same")(input)
    else:
        x = UpSampling2D()(input)
        x = Conv2D(num_channels, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding='same')(x)
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation(activation, name=name + "_act")(x)
    return x


def residual_block(input, n_filters, stride=1, downsample=None, name=None):
    if downsample is None:
        shortcut = input
    else:
        shortcut = downsample

    x = conv_bn_relu(input, n_filters, kernel_size=3, stride=stride, name=name + '/cvbnrelu')
    x = Conv2D(n_filters, (3, 3), name=name + '_conv2', padding='same')(x)
    x = BatchNormalization(name=name + '_batchnorm2')(x)

    x = add([x, shortcut], name=name + "_add")
    x = Activation('relu', name=name + '_relu2')(x)
    return x


def encoder(inputs, m, n, blocks, stride, name='encoder'):
    downsample = None
    if stride != 1 or m != n:
        downsample = Conv2D(n, (1, 1), strides=(stride, stride), name=name + '_conv_downsample')(inputs)
        downsample = BatchNormalization(name=name + '_batchnorm_downsample')(downsample)

    output = residual_block(inputs, n, stride, downsample, name=name + '/residualBlock0')
    for i in range(1, blocks):
        output = residual_block(output, n, stride=1, name=name + '/residualBlock{}'.format(i))
    return output


def decoder(inputs, n_filters, planes, name='decoder', feature_scale=2, transposed_conv=False, activation='relu'):
    x = conv_bn_relu(inputs, num_channel=n_filters // feature_scale, kernel_size=1, stride=1, padding='same', name=name + '/c1', activation=activation)
    x = deconv_bn_relu(x, num_channels=n_filters // feature_scale, kernel_size=3, name=name + '/dc1', transposed_conv=transposed_conv, activation=activation)
    x = conv_bn_relu(x, num_channel=planes, kernel_size=1, stride=1, padding='same', name=name + '/c2', activation=activation)
    return x


def LinkNet(input_shape=(256, 256, 3), classes=1, dropout=0.5, feature_scale=4, pretrained_weights=None, skipConnectionConv1=False):
    """
    Architecture is similar to linknet, but there are a lot of enhancements
     - initial block is changed to strides 1, 2*32 convs to better handle borders.
     - additional encoder block with filter size=512 and corresponding decoder block.
     - ability to use subpixel deconvolution ( not optimized may lead to artifacts)
     - ability to have skip connection from conv1
     - ability to use pretrained RGB weights for network with more channels, additional channels are initialised with zeros the same way as in Deep Image Matting paper

    Overall this approach helps to better deal with very small objects and boundaries between them.

    The network has around 20m parameters in default configuration, for a small dataset like Urban3D it is better to use dropout rate = 0.5
    As SpatialDropout2D (aka feature map dropout) is used instead of ordinary Dropout it improves semantic segmetation performance and helps to reduce overfitting

    See:
        LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation (https://arxiv.org/abs/1707.03718)
        Understanding Convolution for Semantic Segmentation https://arxiv.org/abs/1702.08502
        Deep Image Matting https://arxiv.org/abs/1703.03872
    """
    layers = [2, 2, 2, 2, 2]
    filters = [64, 128, 256, 512, 512]

    inputs = Input(shape=input_shape)
    if pretrained_weights:
        print("Using pretrained weights {}".format(pretrained_weights))
    if pretrained_weights and input_shape[-1] > 3:
        x = conv_bn_relu(inputs, 32, 3, stride=1, name="block1_conv1_changed")
    else:
        x = conv_bn_relu(inputs, 32, 3, stride=1, name="block1_conv1")
    x = conv_bn_relu(x, 32, 3, stride=1, name="block1_conv2")
    conv1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="block1_pool")(x)

    enc1 = encoder(x, m=32, n=filters[0], blocks=layers[0], stride=1, name='encoder1')
    enc2 = encoder(enc1, m=filters[0], n=filters[1], blocks=layers[1], stride=2, name='encoder2')
    enc3 = encoder(enc2, m=filters[1], n=filters[2], blocks=layers[2], stride=2, name='encoder3')
    enc4 = encoder(enc3, m=filters[2], n=filters[3], blocks=layers[3], stride=2, name='encoder4')
    enc5 = encoder(enc4, m=filters[3], n=filters[4], blocks=layers[4], stride=2, name='encoder5')

    x = linknet_decoder(conv1, enc1, enc2, enc3, enc4, enc5, filters, feature_scale)
    x = SpatialDropout2D(dropout)(x)
    x = Conv2D(filters=classes, kernel_size=(1, 1), padding="same", name="prediction")(x)
    x = Activation("sigmoid", name="mask")(x)

    model = Model(inputs=inputs, outputs=x)
    if pretrained_weights:
        print("Loading pretrained weights {}".format(pretrained_weights))
        model.load_weights(pretrained_weights, by_name=True)

    if pretrained_weights and input_shape[-1] > 3:
        conv1_weights = np.zeros((3, 3, input_shape[-1], 32), dtype="float32")
        three_channels_net = LinkNet(input_shape=(224, 224, 3))
        three_channels_net.load_weights(pretrained_weights, by_name=True)
        conv1_weights[:, :, :3, :] = three_channels_net.get_layer("block1_conv1_conv").get_weights()[0][:, :, :, :]
        bias = three_channels_net.get_layer("block1_conv1_conv").get_weights()[1]
        model.get_layer('block1_conv1_changed_conv').set_weights((conv1_weights, bias))
        model.get_layer('block1_conv1_changed_conv').name = 'block1_conv1_conv'

    return model


def linknet_decoder(conv1, enc1, enc2, enc3, enc4, enc5, filters=[64, 128, 256, 512, 512], feature_scale=4, skipFirst=False, transposed_conv=False):
    decoder5 = decoder(enc5, filters[4], filters[3], name='decoder5', feature_scale=feature_scale, transposed_conv=transposed_conv)
    decoder5 = add([decoder5, enc4])
    decoder4 = decoder(decoder5, filters[3], filters[2], name='decoder4', feature_scale=feature_scale, transposed_conv=transposed_conv)
    decoder4 = add([decoder4, enc3])
    decoder3 = decoder(decoder4, filters[2], filters[1], name='decoder3', feature_scale=feature_scale, transposed_conv=transposed_conv)
    decoder3 = add([decoder3, enc2])
    decoder2 = decoder(decoder3, filters[1], filters[0], name='decoder2', feature_scale=feature_scale, transposed_conv=transposed_conv)
    decoder2 = add([decoder2, enc1])
    decoder1 = decoder(decoder2, filters[0], filters[0], name='decoder1', feature_scale=feature_scale, transposed_conv=transposed_conv)
    if skipFirst:
        x = concatenate([conv1, decoder1])
        x = conv_bn_relu(x, 32, 3, stride=1, padding='same', name='f2_skip_1')
        x = conv_bn_relu(x, 32, 3, stride=1, padding='same', name='f2_skip_2')
    else:
        x = conv_bn_relu(decoder1, 32, 3, stride=1, padding='same', name='f2')
    return x


if __name__ == '__main__':
    LinkNet((384, 384, 3)).summary()
