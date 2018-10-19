import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import os

from collections import namedtuple
from utils.batch_generator import BatchGenerator


class DirectoryNotExisting(Exception):
    pass


class NoSuchArchitectureImpemented(Exception):
    pass


LossRecord = namedtuple(
    'LossRecord',
    [
        'epoch', 'eval_train_loss', 'eval_val_loss', 'batch_train_loss',
        'batch_val_loss', 'train_avg_loss', 'train_std_loss',
        'val_avg_loss', 'val_std_loss'
    ]
)


class Verifier(object):
    def __init__(self, module_path, model, model_weights, data_dir):
        self.module_path = module_path
        self.model = model
        self.model_weights = model_weights
        self.data_dir = data_dir

    @property
    def module_path(self):
        return self._module_path

    @property
    def model(self):
        return self._model

    @property
    def model_weights(self):
        return self._model_weights

    @module_path.setter
    def module_path(self, value):
        file_path = '/'.join(value.split('.'))
        file_path = "./{}.py".format(file_path)
        if not os.path.isfile(file_path):
            raise DirectoryNotExisting()

        self._module_path = file_path

    @model.setter
    def model(self, value):
        try:
            spec = importlib.util.spec_from_file_location(
                value, self.module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, value)
            self._model = model_class.model(256, 256)

        except NoSuchArchitectureImpemented as e:
            raise e

    @model_weights.setter
    def model_weights(self, value):
        if not os.path.isfile(value):
            print('Weights file does not exist')
            raise FileNotFoundError()

        self._model_weights = value

    def load_data(self, data_dir, num_of_imgs):
        batch_gen = BatchGenerator(
            data_dir=data_dir,
            batch_size=num_of_imgs
        )
        batch_gen.load_data()

        self.samples_x, self.samples_y = batch_gen.generate_test_batch(
            num_of_imgs
        )

    def plot_images(self):
        """A method creating a figure with test images containing
        actual images and their masks
        """

        imgs_pos = [i for i in range(1, 13, 2)]
        masks_pos = [i for i in range(2, 13, 2)]

        plt.figure(figsize=(12, 8))
        for (pos, img_pos, mask_pos) in zip(range(6), imgs_pos, masks_pos):
            plt.subplot(3, 4, img_pos)
            img = self.samples_x[pos].astype(np.uint8)
            plt.imshow(img)
            plt.subplot(3, 4, mask_pos)
            mask = self.samples_y[pos].astype(np.uint8)
            mask = mask.reshape(mask.shape[:2])
            plt.imshow(mask)
        plt.axis('off')

        plt.savefig(
            os.path.join(
                self.data_dir, 'test_images.png'
            )
        )

    def load_model_weights(self):
        self.model.load_weights(self.model_weights)

    def predict_images(self):
        self._predicted = self.model.predict(self.samples_x)

    def visualize_prediction_on_test(self):
        """A method creating a figure with test images containing
        actual images and their masks
        """

        imgs_pos = [i for i in range(1, 19, 3)]
        masks_pos = [i for i in range(2, 19, 3)]
        masks_gen_pos = [i for i in range(3, 19, 3)]

        plt.figure(figsize=(12, 8))
        for (pos, img_pos, mask_pos, mask_gen_pos) in zip(
                range(6), imgs_pos, masks_pos, masks_gen_pos
        ):
            plt.subplot(3, 6, img_pos)
            img = self.samples_x[pos].astype(np.uint8)
            plt.imshow(img)

            plt.subplot(3, 6, mask_pos)
            mask = self.samples_y[pos].astype(np.uint8)
            mask = mask.reshape(mask.shape[:2])
            plt.imshow(mask)

            plt.subplot(3, 6, mask_gen_pos)
            mask_gen = self._predicted[pos].reshape(256, 256, 2) * 255
            mask_gen = mask_gen.astype(np.uint8)
            mask_gen[mask_gen < 127] = 0
            mask_gen[mask_gen >= 127] = 255
            mask_gen = mask_gen[..., 1]
            plt.imshow(mask_gen)
        plt.axis('off')

        plt.savefig(
            os.path.join(
                self.data_dir, 'test_images_prediction_result.png'
            )
        )

    def _plot_error(self, train_error_dict, val_error_dict):
        """Error plotting method

        :param error_dict: Error dictionary. keys: epochs, values: error values
        :return:
        """
        train_error_change = sorted(
            (key, val) for key, val in train_error_dict.items())
        val_error_change = sorted(
            (key, val) for key, val in val_error_dict.items())
        plt.plot(*zip(*[(int(x[0]), float(x[1])) for x in train_error_change]),
                 'r', label='Trainin error')
        plt.plot(*zip(*[(int(x[0]), float(x[1])) for x in val_error_change]),
                 'b', label='Validation error')
        plt.legend()
        plt.xlabel('Epoch #')
        plt.ylabel('Error Value')
        plt.grid()
        plt.title('Error across epochs')
        plt.savefig(
            os.path.join(self.data_dir, 'error_change.png')
        )

    def visualize_error(self):
        """Visualize training and validation error curves"""

        files = sorted([
            file for file in os.listdir(self.data_dir)
            if file.endswith('.txt') and 'loss' in file
        ])

        error = []
        for file in files:
            with open(os.path.join(self.data_dir, file), 'r') as f:
                next(f)
                error.append(f.readlines())
        error = [
            sample.split(',') for file_sample in error
            for sample in file_sample
        ]

        loss_data = [
            LossRecord(*record) for record in error if record
        ]

        train_error_change = dict(
            (int(x.epoch), float(x.batch_train_loss)) for x in loss_data)
        val_error_change = dict(
            (int(x.epoch), float(x.batch_val_loss)) for x in loss_data)

        self._plot_error(train_error_change, val_error_change)



