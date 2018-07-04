from keras import backend as K
from keras.layers import (
    Input,
    concatenate,
    Convolution2D,
    MaxPooling2D,
    UpSampling2D
)
from keras.models import Model


class Unet(object):
    @staticmethod
    def model(img_rows, img_cols):
        inputs = Input(shape=(img_rows, img_cols, 3))

        conv1 = Convolution2D(filters=8, kernel_size=(3, 3), activation='relu',
                              padding='same')(inputs)
        conv1 = Convolution2D(filters=8, kernel_size=(3, 3), activation='relu',
                              padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(filters=16, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool1)
        conv2 = Convolution2D(filters=16, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(filters=32, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool2)
        conv3 = Convolution2D(filters=32, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(filters=64, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool3)
        conv4 = Convolution2D(filters=64, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(filters=128, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool4)
        conv5 = Convolution2D(filters=128, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
        conv6 = Convolution2D(filters=64, kernel_size=(3, 3),
                              activation='relu', padding='same')(up6)
        conv6 = Convolution2D(filters=64, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Convolution2D(filters=32, kernel_size=(3, 3),
                              activation='relu', padding='same')(up7)
        conv7 = Convolution2D(filters=32, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = Convolution2D(filters=16, kernel_size=(3, 3),
                              activation='relu', padding='same')(up8)
        conv8 = Convolution2D(filters=16, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = Convolution2D(filters=8, kernel_size=(3, 3), activation='relu',
                              padding='same')(up9)
        conv9 = Convolution2D(filters=8, kernel_size=(3, 3), activation='relu',
                              padding='same')(conv9)

        conv10 = Convolution2D(filters=3, kernel_size=(1, 1),
                               activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model

    @staticmethod
    def metric(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def loss(y_true, y_pred):
        return 1 - Unet.metric(y_true, y_pred)
