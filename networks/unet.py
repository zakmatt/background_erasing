from keras import backend as K
from keras.layers import (
    Activation,
    Input,
    concatenate,
    Convolution2D,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
from keras.models import Model


class Unet(object):
    @staticmethod
    def model(img_rows, img_cols):
        inputs = Input(shape=(img_rows, img_cols, 3))

        # 256x256x3 -> 128x128x8
        conv1 = Convolution2D(filters=8, kernel_size=(3, 3), activation='relu',
                              padding='same')(inputs)
        conv1 = Convolution2D(filters=8, kernel_size=(3, 3), activation='relu',
                              padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # 128x128x8 -> 64x64x16
        conv2 = Convolution2D(filters=16, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool1)
        conv2 = Convolution2D(filters=16, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # 64x64x16 -> 32x32x32
        conv3 = Convolution2D(filters=32, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool2)
        conv3 = Convolution2D(filters=32, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # 32x32x32 -> 16x16x64
        conv4 = Convolution2D(filters=64, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool3)
        conv4 = Convolution2D(filters=64, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # 16x16x64 -> 8x8x128
        conv5 = Convolution2D(filters=128, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool4)
        conv5 = Convolution2D(filters=128, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # 8x8x128 -> 4x4x256
        conv6 = Convolution2D(filters=256, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool5)
        conv6 = Convolution2D(filters=256, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv6)
        pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

        # 4x4x256 -> 2x2x512
        conv7 = Convolution2D(filters=512, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool6)
        conv7 = Convolution2D(filters=512, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv7)
        pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

        # 2x2x512 -> 1x1x1024
        conv8 = Convolution2D(filters=1024, kernel_size=(3, 3),
                              activation='relu', padding='same')(pool7)
        conv8 = Convolution2D(filters=1024, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv8)

        # 1x1x1024 -> 2x2x512
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv7], axis=-1)
        conv9 = Convolution2D(filters=512, kernel_size=(3, 3),
                              activation='relu', padding='same')(up9)
        conv9 = Convolution2D(filters=512, kernel_size=(3, 3),
                              activation='relu', padding='same')(conv9)

        # 2x2x512 -> 4x4x256
        up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv6], axis=-1)
        conv10 = Convolution2D(filters=256, kernel_size=(3, 3),
                               activation='relu', padding='same')(up10)
        conv10 = Convolution2D(filters=256, kernel_size=(3, 3),
                               activation='relu', padding='same')(conv10)

        # 4x4x256 -> 8x8x128
        up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv5], axis=-1)
        conv11 = Convolution2D(filters=128, kernel_size=(3, 3),
                               activation='relu', padding='same')(up11)
        conv11 = Convolution2D(filters=128, kernel_size=(3, 3),
                               activation='relu', padding='same')(conv11)

        # 8x8x128 -> 16x16x64
        up12 = concatenate([UpSampling2D(size=(2, 2))(conv11), conv4], axis=-1)
        conv12 = Convolution2D(filters=64, kernel_size=(3, 3),
                               activation='relu', padding='same')(up12)
        conv12 = Convolution2D(filters=64, kernel_size=(3, 3),
                               activation='relu', padding='same')(conv12)

        # 16x16x64 -> 32x32x32
        up13 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv3], axis=-1)
        conv13 = Convolution2D(filters=32, kernel_size=(3, 3),
                               activation='relu', padding='same')(up13)
        conv13 = Convolution2D(filters=32, kernel_size=(3, 3),
                               activation='relu', padding='same')(conv13)

        # 32x32x32 -> 64x64x16
        up14 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv2], axis=-1)
        conv14 = Convolution2D(filters=16, kernel_size=(3, 3),
                               activation='relu', padding='same')(up14)
        conv14 = Convolution2D(filters=16, kernel_size=(3, 3),
                               activation='relu', padding='same')(conv14)

        # 64x64x16 -> 128x128x8
        up15 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv1], axis=-1)
        conv15 = Convolution2D(filters=8, kernel_size=(3, 3),
                               activation='relu', padding='same')(up15)
        conv15 = Convolution2D(filters=8, kernel_size=(3, 3),
                               activation='relu', padding='same')(conv15)

        conv16 = Convolution2D(filters=2, kernel_size=(1, 1),
                               activation='linear')(conv15)

        flat_output = Reshape((img_cols*img_rows, 2))(conv16)
        softmax_output = Activation('softmax')(flat_output)

        model = Model(inputs=inputs, outputs=softmax_output)

        return model

    @staticmethod
    def metric(y_true, y_false, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_false_f = K.flatten(y_false)
        intersection = K.sum(y_true_f * y_false_f)
        union = K.sum(y_true_f) + K.sum(y_false_f) - intersection
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def loss(y_true, y_pred):
        return 1.0 - Unet.metric(y_true, y_pred)
