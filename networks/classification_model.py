import os
import tensorflow as tf

from abc import ABCMeta

from utils.loss_validate_callback import LossValidateCallback


# noinspection PyPep8Naming
class ClassModel(metaclass=ABCMeta):
    """

    """

    name = 'class_model'

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_gen = batch_gen

        f_path = str(
            save_model_dir
        ) + '/' + self.name + '_batch_' + str(
            batch_gen.batch_size
        ) + '_epoch_{epoch:02d}.hdf5'

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f_path,
            mode='auto',
            period=1
        )
        self.callbacks = [checkpoint]

        reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.8, patience=10, verbose=1,
            mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001
        )

        results_file = os.path.join(save_model_dir, results_file)
        self.callbacks = [
            checkpoint,
            reduce_on_plateau,
            LossValidateCallback(
                batch_gen.generate_test_batch,
                results_file
            )
        ]

    def create_model(self, img_rows, img_cols, model_architecture):
        """Create a fine tuned model based on the provided architecture

        :param img_rows: number of rows in an image
        :type img_rows: int
        :param img_cols: number of columns in an image
        :type img_cols: int
        :param model_architecture: model architecture
        :type model_architecture: keras.applications.vgg16.VGG16
        """

        base_model = model_architecture(
            input_shape=(img_rows, img_cols, 3),
            weights='imagenet',
            include_top=False
        )
        # base_model.trainable = False
        for layer in base_model.layers:
            layer.trainable = False

        # add a global spatial average pooling layer
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        predictions = tf.keras.layers.Dense(
            3, activation='softmax', name='activations'
        )(x)

        self.model = tf.keras.Model(
            inputs=base_model.input,
            outputs=predictions
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, initial_epoch, nb_epochs):

        (_, _), (val_x, val_y) = self.batch_gen.generate_test_batch()
        if hasattr(self, 'model'):
            self.model.fit_generator(
                self.batch_gen.train_batches,
                steps_per_epoch=self.batch_gen.num_batches,
                epochs=nb_epochs,
                callbacks=self.callbacks,
                initial_epoch=initial_epoch,
                validation_data=(val_x, val_y)
            )

    def load_weights(self, weights_path):
        if hasattr(self, 'model'):
            self.model.load_weights(weights_path)
