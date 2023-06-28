import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate, Input
from tensorflow.keras.models import Model


def conv_block(input_, n_filters):
    x = Conv2D(n_filters, 3, padding="same")(input_)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input_, n_filters):
    x = conv_block(input_, n_filters)
    p = MaxPool2D([2, 2])(x)

    return x, p


def decoder_block(input_, skip, n_filters):
    x = Conv2DTranspose(n_filters, [2, 2], strides=2, padding="same")(input_)
    x = Concatenate()([x, skip])
    x = conv_block(x, n_filters)

    return x


def UNet(shape=(768, 768, 3)):
    # input tensor
    input_ = Input(shape)

    # reduce the image size
    x = MaxPool2D([3, 3])(input_)

    # 4 encoder blocks
    s1, p1 = encoder_block(x, 4)
    s2, p2 = encoder_block(p1, 8)
    s3, p3 = encoder_block(p2, 16)
    s4, p4 = encoder_block(p3, 32)

    # bridge between encoder and decoder blocks
    b1 = conv_block(p4, 64)

    # 4 encoder blocks
    d1 = decoder_block(b1, s4, 32)
    d2 = decoder_block(d1, s3, 16)
    d3 = decoder_block(d2, s2, 8)
    d4 = decoder_block(d3, s1, 4)

    # reduce image channels to 1
    x = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    # increase the image size and get output tensor
    output_ = UpSampling2D([3, 3])(x)

    # create model object and return it
    model = Model(input_, output_, name="UNet")
    return model
