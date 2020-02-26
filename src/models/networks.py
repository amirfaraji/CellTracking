from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose
from keras.layers import Input, MaxPooling2D, SeparableConv2D, UpSampling2D
from keras.layers import Concatenate, Flatten, Dense, Dropout, ZeroPadding2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


 


def unet_xception(img_shape):
    """
    U-Net architecture with xception encoder 
    :param img_shape: image shape
    :return: Keras model
    """
    img_input = Input(img_shape)
    xcept = Xception(include_top=False, weights='imagenet')

    x = UpSampling2D((2,2))(xcept(img_input))
    residual = Conv2DTranspose(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(xcept(img_input))
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block15_sepconv1')(x)
    x = BatchNormalization(name='block15_sepconv1_bn')(x)
    x = Activation('relu', name='block15_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block15_sepconv2')(x)
    x = BatchNormalization(name='block15_sepconv2_bn')(x)

    x = Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 16)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3),
                                    padding='same',
                                    use_bias=False,
                                    name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = Add()([x, residual])




    residual = Conv2DTranspose(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = UpSampling2D((2,2))(x)

    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block24_sepconv1')(x)
    x = BatchNormalization(name='block24_sepconv1_bn')(x)
    x = Activation('relu', name='block24_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block24_sepconv2')(x)
    x = BatchNormalization(name='block24_sepconv2_bn')(x)

    x = Add()([x, residual])


    residual = Conv2DTranspose(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = UpSampling2D((2,2))(x)

    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block25_sepconv1')(x)
    x = BatchNormalization(name='block25_sepconv1_bn')(x)
    x = Activation('relu', name='block25_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block25_sepconv2')(x)
    x = BatchNormalization(name='block25_sepconv2_bn')(x)


    x = Add()([x, residual])



    residual = Conv2DTranspose(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = UpSampling2D((2,2))(x)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block26_sepconv1')(x)
    x = BatchNormalization(name='block26_sepconv1_bn')(x)
    x = Activation('relu', name='block26_sepconv1_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block26_sepconv2')(x)
    x = BatchNormalization(name='block26_sepconv2_bn')(x)


    x = Add()([x, residual])

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='block27_conv1')(x)
    x = BatchNormalization( name='block27_conv1_bn')(x)
    x = Activation('relu', name='block27_conv1_act')(x)

    x = Conv2DTranspose(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization( name='block27_conv2_bn')(x)
    x = Activation('relu', name='block27_conv2_act')(x)

    x = Conv2D(3, (1, 1), name='output_conv1')(x)
    output = Activation('sigmoid')(x)
    model = Model(img_input, output)

    return model


def conv_bn_relu(inputs, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
    """Order of Common Layers"""
    x = Conv2D(nb_outputs, kernel_size=kernel, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def conv_relu_bn(inputs, nb_outputs, kernel, strides=(1,1), padding='same', activation='relu'):
    """Order of Common Layers"""
    x = Conv2D(nb_outputs, kernel_size=kernel, strides=strides, padding=padding)(inputs)
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    return x

def unet(img_shape, num_filts):
    """U-Net Architecture"""
    inputs = Input(img_shape)
    conv1 = conv_bn_relu(inputs, 4*num_filts, 1)
    conv1 = conv_bn_relu(conv1, 4*num_filts, 3)
    conv1 = conv_bn_relu(conv1, 4*num_filts, 1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, 8*num_filts, 1)
    conv2 = conv_bn_relu(conv2, 8*num_filts, 3)
    conv2 = conv_bn_relu(conv2, 8*num_filts, 1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, 16*num_filts, 1)
    conv3 = conv_bn_relu(conv3, 16*num_filts, 3)
    conv3 = conv_bn_relu(conv3, 16*num_filts, 1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(pool3, 32*num_filts, 1)
    conv4 = conv_bn_relu(conv4, 32*num_filts, 3)
    conv4 = conv_bn_relu(conv4, 32*num_filts, 1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(pool4, 64*num_filts, 1)
    conv5 = conv_bn_relu(conv5, 64*num_filts, 3)
    conv5 = conv_bn_relu(conv5, 64*num_filts, 3)
    conv5 = conv_bn_relu(conv5, 64*num_filts, 3)
    conv5 = conv_bn_relu(conv5, 64*num_filts, 1)
    
    up4 = conv_bn_relu(
        UpSampling2D(size=(2, 2))(conv5),
        16*num_filts, 
        1
    )
    merge4 = Concatenate()([conv4, up4])
    convup4 = conv_bn_relu(merge4, 16*num_filts, 3)
    convup4 = conv_bn_relu(convup4, 16*num_filts, 1)

    up3 = conv_bn_relu(
        UpSampling2D(size=(2, 2))(convup4),
        16*num_filts, 
        1
    )
    merge3 = Concatenate()([conv3, up3])
    convup3 = conv_bn_relu(merge3, 16*num_filts, 3)
    convup3 = conv_bn_relu(convup3, 16*num_filts, 1)



    up2 = conv_bn_relu(
        UpSampling2D(size=(2, 2))(convup3),
        8*num_filts, 
        1
    )
    merge2 = Concatenate()([conv2, up2])
    convup2 = conv_bn_relu(merge2, 8*num_filts, 3)
    convup2 = conv_bn_relu(convup2, 8*num_filts, 1)

    up1 = conv_bn_relu(
        UpSampling2D(size=(2, 2))(convup2),
        4*num_filts, 
        1
    )
    merge1 = Concatenate()([conv1, up1])
    convup1 = conv_bn_relu(merge1, 4*num_filts, 3)
    convup1 = conv_bn_relu(convup1, 4*num_filts, 1)

    outputs = conv_bn_relu(convup1, 1, 1, activation='sigmoid')
    model = Model(inputs, outputs)
    return model

def unet_VGG16(img_shape):
    img_input = Input (shape = (img_shape))
	
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    f1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    f2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	
	# Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x) 
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    f3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)


	# Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    f4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
	
	# Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    f5 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	

	
    VGG16PreTrained = VGG16(include_top=False, weights='imagenet')
    #VGG16PreTrained.summary()

    vgg  = Model(img_input , x  )
    #vgg.summary()
    #vgg16.load_weights(vgg_weight_path, by_name=True)
    vgg.set_weights(VGG16PreTrained.get_weights()) 

	
    #levels = [f1 , f2 , f3 , f4,  f5]
	
    o = f5
	
    o = Conv2D(512, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.2)(o)

    o = UpSampling2D( (2,2))(o)
    o = Conv2D( 512, (3, 3), padding='same')(o)
    o = Dropout(0.2)(o)
    o = Concatenate()([o, f4]) 
    o = Conv2D( 512, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.2)(o)

    o = UpSampling2D( (2,2))(o)
    o = Conv2D( 256, (3, 3), padding='same')(o)
    o = Dropout(0.2)(o)
    o = Concatenate()([o,f3]) 
    o = Conv2D( 256, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.2)(o)

    o = UpSampling2D((2,2))(o)
    o = Conv2D( 128, (3, 3), padding='same')(o)
    o = Dropout(0.2)(o)	
    o = Concatenate()([o,f2])
    o = Conv2D( 128 , (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.2)(o)	

    o = UpSampling2D( (2,2))(o)
    o = Conv2D( 64, (3, 3), padding='same')(o)
    o = Dropout(0.2)(o)	
    o = Concatenate()([o,f1]) 
    o = Conv2D( 64 , (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.2)(o)	

    density_pred =  Conv2D(3, (1, 1), bias = False, activation='sigmoid', init='orthogonal',name='pred')(o)
	
    model = Model(img_input, density_pred)

    #model.summary()

    return model