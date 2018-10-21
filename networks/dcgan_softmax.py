import logging
import numpy as np

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
    Activation,
    Reshape
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from networks.DCGAN import DCGAN


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DCGAN_softmax(DCGAN):

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file, val_batch_size):

        self.mask_channels = 2
        super().__init__(
            img_rows, img_cols, batch_gen,
            save_model_dir, results_file, val_batch_size
        )

        self.generator_weights_path = str(
            save_model_dir) + '/DCGAN_softmax_batch_1_epoch_{}.hdf5'
        self.dcgan_weights_path = str(
            save_model_dir) + '/generator_softmax_batch_1_epoch_{}.hdf5'
        self.discriminator_weights_path = str(
            save_model_dir) + '/discriminator_softmax_batch_1_epoch_{}.hdf5'

        # build a generator
        self.generator = self._generator(64)
        self.generator.compile(
            loss=DCGAN_softmax._generator_crossentropy,
            optimizer=Adam(0.0002, 0.5)
        )

        # change batch masks to have 2. channels for loss validate class
        self.loss_validate.to_categ = True

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
            activation='relu'
        )(rectified_inputs)

        output = Convolution2D(filters=2, kernel_size=(1, 1),
                               activation='linear')(output)

        flat_output = Reshape((self.img_cols * self.img_rows, 2))(output)
        softmax_output = Activation('softmax')(flat_output)

        layers.append(softmax_output)
        return Model(inputs=inputs, outputs=softmax_output)

    @staticmethod
    def _generator_crossentropy(predicted, target):
        return K.binary_crossentropy(predicted, target)

    def _get_batch(self):
        while True:
            x, mask = next(self.batch_generator.train_batches)
            mask = to_categorical(mask, 2)
            mask = np.reshape(
                mask,
                (
                    self.batch_generator.batch_size,
                    self.img_cols, self.img_rows, 2
                )
            )
            yield x, mask

    def train(self, initial_epoch, nb_epochs, steps_per_epoch=int(1e3)):
        """Model training method"""

        # Adversarial ground truths
        valid = np.ones((self.batch_generator.batch_size, 1))
        fake = np.zeros((self.batch_generator.batch_size, 1))

        for epoch in range(initial_epoch + 1, initial_epoch + nb_epochs):
            discrim_losses = []
            for step in range(steps_per_epoch):
                # -------------------
                # Train discriminator
                # -------------------
                img, mask = next(self._get_batch())
                generated = self.generator.predict(img)
                generated = np.reshape(
                    generated,
                    (
                        self.batch_generator.batch_size,
                        *self.mask_shape
                    )
                )

                # train discriminator where real-like classified images are 1's
                # and fake-like ones are 0's
                discrim_loss_real = self.discriminator.train_on_batch(
                    np.concatenate([img, mask], axis=-1), valid
                )
                discrim_loss_fake = self.discriminator.train_on_batch(
                    np.concatenate([img, generated], axis=-1), fake
                )
                discrim_losses.append(discrim_loss_real)
                discrim_losses.append(discrim_loss_fake)

                # ---------------
                # Train generator
                # ---------------
                self.combined_model.train_on_batch(
                    img, valid
                )

            logging.info('One image IOU: {:2f}'.format(
                DCGAN.metric(generated, mask))
            )
            discrim_loss = np.sum(discrim_losses, axis=0)/len(discrim_losses)
            logging.info('Discrimination loss: {:2f}, accucacy: {:2f}'.format(
                discrim_loss[0], 100 * discrim_loss[1]
            ))
            # log error value
            self.loss_validate.error_log(epoch)

            if epoch % 50 == 0:
                self.combined_model.save_weights(
                    self.dcgan_weights_path.format(epoch)
                )
                self.generator.save_weights(
                    self.generator_weights_path.format(epoch)
                )
                self.discriminator.save_weights(
                    self.discriminator_weights_path.format(epoch)
                )
