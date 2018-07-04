from keras import backend as K
from keras.layers import (
    Input,
    concatenate,
    Convolution2D,
    Conv2DTranspose,
    BatchNormalization,
    Dropout
)
from keras.models import Model


class Unet(object):
    @staticmethod
    def model(img_rows, img_cols):
        inputs = Input(shape=(img_rows, img_cols, 3))

        conv1 = Convolution2D(filters=8, kernel_size=2, strides=(2, 2),
                              activation='relu',
                              padding='same')(inputs)

        conv2 = Convolution2D(filters=16, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv1)

        conv3 = Convolution2D(filters=32, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv2)

        conv4 = Convolution2D(filters=64, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv3)

        conv5 = Convolution2D(filters=128, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv4)

        conv6 = Convolution2D(filters=256, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv5)

        conv7 = Convolution2D(filters=512, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv6)

        conv8 = Convolution2D(filters=512, kernel_size=2, strides=(2, 2),
                              activation='relu', padding='same')(conv7)

        up9 = Conv2DTranspose(filters=512, kernel_size=4, strides=(2, 2),
                              activation='relu', padding='same')(conv8)
        up9 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up9)
        up9 = Dropout(rate=0.5)(up9)

        up10 = concatenate([up9, conv7])
        up10 = Conv2DTranspose(filters=256, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up10)
        up10 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up10)
        up10 = Dropout(rate=0.5)(up10)

        up11 = concatenate([up10, conv6])
        up11 = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up11)
        up11 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up11)

        up12 = concatenate([up11, conv5])
        up12 = Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up12)
        up12 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up12)

        up13 = concatenate([up12, conv4])
        up13 = Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up13)
        up13 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up13)

        up14 = concatenate([up13, conv3])
        up14 = Conv2DTranspose(filters=16, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up14)
        up14 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up14)

        up15 = concatenate([up14, conv2])
        up15 = Conv2DTranspose(filters=8, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up15)
        up15 = BatchNormalization(momentum=0.1, epsilon=1e-5)(up15)

        up16 = concatenate([up15, conv1])
        up16 = Conv2DTranspose(filters=8, kernel_size=4, strides=(2, 2),
                               activation='relu', padding='same')(up16)

        conv17 = Convolution2D(filters=1, kernel_size=(1, 1),
                               activation='sigmoid')(up16)

        model = Model(inputs=inputs, outputs=conv17)

        return model

    @staticmethod
    def metric(y_true, y_false, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_false_f = K.flatten(y_false)
        intersection = K.sum(y_true_f * y_false_f)
        return 2 * (intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_false_f) + smooth)

    @staticmethod
    def loss(y_true, y_false):
        return -1 * Unet.metric(y_true, y_false)