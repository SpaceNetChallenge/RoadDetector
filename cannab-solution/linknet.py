from keras import layers
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, concatenate, UpSampling2D
from keras.applications.vgg16 import VGG16
from models import conv_block
from resnet50_padding_same import ResNet50, identity_block
from resnet50_padding_same import conv_block as resnet_conv_block

bn_axis = 3

def linknet_residual_block(input_tensor, filters, shortcut=None):
    if shortcut is None:
        shortcut = input_tensor
    
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def linknet_conv_block(input_tensor, filters, res_blocks, stride=2):
    if stride == 1:
        x = input_tensor
    else:
        x = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(input_tensor)
        x = BatchNormalization(axis=bn_axis)(x)
    
    for i in range(res_blocks):
        x = linknet_residual_block(x, filters)
        
    return x

def linknet_deconv_block(input_tensor, filters_in, filters_out):
    x = Conv2D(int(filters_in/4), (1, 1), padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(int(filters_in/4), (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters_out, (1, 1), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    return x

def get_linknet(input_shape=(512, 512)):
    inp = Input(input_shape + (9,))
    
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2))(inp) #, strides=(2, 2)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
#    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    enc1 = linknet_conv_block(x, 64, 3, stride=1) #, stride=1
    enc2 = linknet_conv_block(enc1, 128, 4)
    enc3 = linknet_conv_block(enc2, 256, 6)
    enc4 = linknet_conv_block(enc3, 384, 3)
    enc5 = linknet_conv_block(enc4, 512, 3)
    
    dec5 = linknet_deconv_block(enc5, 512, 384)
    dec5 = layers.add([dec5, enc4])
    dec4 = linknet_deconv_block(dec5, 384, 256)
    dec4 = layers.add([dec4, enc3])
    dec3 = linknet_deconv_block(dec4, 256, 128)
    dec3 = layers.add([dec3, enc2])
    dec2 = linknet_deconv_block(dec3, 128, 64)
    dec2 = layers.add([dec2, enc1])
    dec1 = linknet_deconv_block(dec2, 64, 64)
    
#    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(dec1)
#    x = BatchNormalization(axis=bn_axis)(x)
#    x = Activation('relu')(x)
    
    x = Conv2D(48, (3, 3), padding='same')(dec1)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = Model(inp, x)
    
    return model

def get_resnet50_linknet(input_shape, weights='imagenet'):
    inp = Input(input_shape + (9,))
    
    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inp)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
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
        
    dec4 = linknet_deconv_block(enc4, 2048, 1024)
    dec4 = layers.add([dec4, enc3])
    dec3 = linknet_deconv_block(dec4, 1024, 512)
    dec3 = layers.add([dec3, enc2])
    dec2 = linknet_deconv_block(dec3, 512, 256)
    dec2 = layers.add([dec2, enc1])
    dec1 = linknet_deconv_block(dec2, 256, 64)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(dec1)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(48, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = Model(inp, x)
    if weights == 'imagenet':
        resnet = ResNet50(input_shape=input_shape + (3,), include_top=False, weights=weights)
        for i in range(2, len(resnet.layers)-1):
            model.layers[i].set_weights(resnet.layers[i].get_weights())
            model.layers[i].trainable = False
        
    return model

def get_vgg_linknet_small(input_shape, weights='imagenet', freeze=False):  
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
    pool5 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv5)
    
    conv6 = conv_block(pool5, 512, prefix='conv6')
    conv6 = conv_block(conv6, 512, prefix='conv6_2')
    conv6 = conv_block(conv6, 512, prefix='conv6_3')
    
    dec5 = linknet_deconv_block(conv6, 512, 512)
    dec5 = layers.add([dec5, conv5])
    dec4 = linknet_deconv_block(dec5, 512, 512)
    dec4 = layers.add([dec4, conv4])
    dec3 = linknet_deconv_block(dec4, 512, 256)
    dec3 = layers.add([dec3, conv3])
    dec2 = linknet_deconv_block(dec3, 256, 128)
    dec2 = layers.add([dec2, conv2])
    dec1 = linknet_deconv_block(dec2, 128, 64)
    dec1 = layers.add([dec1, conv1])
    
    x = Conv2D(48, (3, 3), padding='same')(dec1)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = Model(input1, x)
    
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