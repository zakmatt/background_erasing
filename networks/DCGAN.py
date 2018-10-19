import logging
import numpy as np
import os

from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import (
    BatchNormalization,
    Conv2DTranspose,
    Concatenate,
    Convolution2D,
    Dropout,
    Input,
    LeakyReLU,
    ReLU,
    ZeroPadding2D
)
from keras.models import Model
from keras.optimizers import Adam

from utils.loss_validate import LossValidate

EPS = 1e-12

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DCGAN(object):

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file, val_batch_size):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = 3
        self.mask_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.mask_shape = (self.img_rows, self.img_cols, self.mask_channels)
        self.discriminator_input_shape = (
            self.img_rows, self.img_cols,
            self.img_channels + self.mask_channels
        )

        self.batch_generator = batch_gen

        self.generator_weights_path = str(
            save_model_dir) + '/DCGAN_batch_1_epoch_{}.hdf5'
        self.dcgan_weights_path = str(
            save_model_dir) + '/generator_batch_1_epoch_{}.hdf5'
        self.discriminator_weights_path = str(
            save_model_dir) + '/discriminator_batch_1_epoch_{}.hdf5'

        results_file = os.path.join(save_model_dir, results_file)

        # build a generator
        self.generator = self._generator(64)
        self.generator.compile(
            loss=DCGAN._generator_l1_loss,
            optimizer=Adam(0.0002, 0.5)
        )

        # build discriminator
        self.discriminator = self._discriminator(64)
        self.discriminator.compile(
            loss=DCGAN._discriminator_loss,
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy']
        )

        # generator takes an image as an input and returns a mask
        image = Input(shape=self.img_shape)
        generated_mask = self.generator(image)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated masks with images
        #  as input and determines validity
        combined_inputs_fake = Concatenate(axis=-1)([image, generated_mask])

        # return patch o 1's and 0's
        discrim_fake = self.discriminator(combined_inputs_fake)
        self.combined_model = Model(inputs=image, outputs=combined_inputs_fake)
        self.combined_model.compile(
            optimizer=Adam(0.0002, 0.5),
            loss=DCGAN._generator_loss(discrim_fake)
        )

        # build logs
        self.loss_validate = LossValidate(
            self.generator,
            batch_gen.generate_test_batch,
            results_file,
            val_batch_size=val_batch_size
        )

    def _generator(self, n_filters):
        """Generator method"""
        inputs = Input(shape=self.img_shape)

        layers = []

        output = Convolution2D(
            filters=n_filters, kernel_size=(2, 2),
            strides=(2, 2), padding='same'
        )(inputs)
        layers.append(output)

        layers_specs = [
            n_filters * 2,
            n_filters * 4,
            n_filters * 8,
            n_filters * 8,
            n_filters * 8,
            n_filters * 8,
            n_filters * 8
        ]

        for output_channels in layers_specs:
            rectified_inputs = LeakyReLU(alpha=0.2)(layers[-1])
            convolved = Convolution2D(
                filters=output_channels, kernel_size=4,
                strides=(2, 2), padding='same'
            )(rectified_inputs)
            output = BatchNormalization(
                axis=-1, momentum=0.1, epsilon=1e-5,
                gamma_initializer=RandomNormal(1.0, 0.02)
            )(convolved)
            layers.append(output)

        layers_specs = [
            (n_filters * 8, 0.5),
            (n_filters * 8, 0.5),
            (n_filters * 8, 0.5),
            (n_filters * 8, 0.0),
            (n_filters * 4, 0.0),
            (n_filters * 2, 0.0),
            (n_filters, 0.0),
        ]

        num_encoder_layers = len(layers)
        for dec_layer, (output_channels, dropout) in enumerate(layers_specs):
            skip_layer = num_encoder_layers - dec_layer - 1
            if dec_layer == 0:
                # no skip connection for the first decoding layer
                layer_inputs = layers[-1]
            else:
                layer_inputs = Concatenate(axis=-1)(
                    [layers[-1], layers[skip_layer]]
                )
            rectified_output = ReLU()(layer_inputs)
            output = Conv2DTranspose(
                filters=output_channels, kernel_size=4, strides=(2, 2),
                padding='same', kernel_initializer=RandomNormal(0.0, 0.02)
            )(rectified_output)
            output = BatchNormalization(
                axis=-1, momentum=0.1,
                epsilon=1e-5, gamma_initializer=RandomNormal(1.0, 0.02)
            )(output)

            if dropout > 0:
                output = Dropout(dropout)(output)

            layers.append(output)

        layer_inputs = Concatenate(axis=-1)([layers[-1], layers[0]])
        rectified_inputs = ReLU()(layer_inputs)
        output = Conv2DTranspose(
            filters=self.mask_channels, kernel_size=4, strides=(2, 2),
            padding='same', kernel_initializer=RandomNormal(0.0, 0.02),
            activation='sigmoid'
        )(rectified_inputs)
        layers.append(output)
        return Model(inputs=inputs, outputs=output)

    @staticmethod
    def _generator_l1_loss(targets, generated):
        gen_loss_l1 = K.mean(K.abs(targets - generated))
        return gen_loss_l1

    @staticmethod
    def _generator_loss(predict_fake,
                        gan_weight=1.0,
                        l1_weight=100):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_gan = K.mean(-K.log(predict_fake + EPS))

        def loss(targets, generated):
            gen_loss_l1 = K.mean(K.abs(targets - generated))
            gen_loss = gen_loss_gan * gan_weight + gen_loss_l1 * l1_weight
            return gen_loss
        return loss

    def _discriminator(self, n_filters):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] =>
        #    [batch, height, width, in_channels * 2]
        combined_inputs = Input(shape=self.discriminator_input_shape)
        padded = ZeroPadding2D(
            padding=((1, 1), (1, 1)), data_format='channels_last'
        )(combined_inputs)
        output = Convolution2D(
            filters=n_filters, kernel_size=4,
            strides=(2, 2), padding='valid',
            kernel_initializer=RandomNormal(0, 0.02)
        )(padded)
        output = LeakyReLU(alpha=0.2)(output)
        layers.append(output)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]

        for i in range(n_layers):
            output_channels = n_filters * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2
            output = Convolution2D(
                filters=output_channels, kernel_size=4,
                strides=(stride, stride), padding='valid',
                kernel_initializer=RandomNormal(0, 0.02)
            )(layers[-1])
            output = BatchNormalization(
                axis=-1, epsilon=1e-5, momentum=0.1,
                gamma_initializer=RandomNormal(1.0, 0.02)
            )(output)
            output = LeakyReLU(alpha=0.2)(output)
            layers.append(output)

        output = Convolution2D(
            filters=1, kernel_size=4,
            strides=(1, 1), padding='valid',
            kernel_initializer=RandomNormal(0, 0.02),
            activation='sigmoid'
        )(layers[-1])
        layers.append(output)

        return Model(inputs=combined_inputs, outputs=layers[-1])

    @staticmethod
    def _discriminator_loss(predict_real, predict_fake):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        return K.mean(
            -(K.log(predict_real + EPS) + K.log(1 - predict_fake + EPS))
        )

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def train(self, initial_epoch, nb_epochs, steps_per_epoch=int(1e3)):
        """Model training method"""

        # Adversarial ground truths
        valid = np.ones((self.batch_generator.batch_size, 1))
        fake = np.zeros((self.batch_generator.batch_size, 1))

        for epoch in range(initial_epoch + 1, initial_epoch + nb_epochs):
            for step in range(steps_per_epoch):
                # -------------------
                # Train discriminator
                # -------------------
                img, mask = next(self.batch_generator.train_batches)
                generated = self.generator.predict(img)

                # train discriminator where real-like classified images are 1's
                # and fake-like ones are 0's
                discrim_loss_real = self.discriminator.train_on_batch(
                    np.concatenate([img, mask], axis=-1), valid
                )
                d_loss_fake = self.discriminator.train_on_batch(
                    np.concatenate([img, generated], axis=-1), fake
                )

                # ---------------
                # Train generator
                # ---------------
                self.combined_model.train_on_batch(
                    img, valid
                )

            logging.info('One image IOU: {:2f}'.format(
                DCGAN.metric(generated, mask))
            )
            # log error value
            self.loss_validate.error_log(epoch)

            if epoch % 1 == 0:
                self.combined_model.save_weights(self.dcgan_weights_path)
                self.generator.save_weights(self.generator_weights_path)
                self.discriminator.save_weights(
                    self.discriminator_weights_path
                )

    @staticmethod
    def metric(y_true, y_false, smooth=1.):
        y_true_f = y_true.flatten()
        y_false_f = y_false.flatten()
        intersection = np.sum(y_true_f * y_false_f)
        union = np.sum(y_true_f) + np.sum(y_false_f) - intersection
        return (intersection + smooth) / (union + smooth)
