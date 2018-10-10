import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.batch_generator import BatchGenerator


class DirectoryNotExisting(Exception):
    pass


class NoSuchArchitectureImpemented(Exception):
    pass


class Verifier(object):
    def __init__(self, module_path, architecture, model_weights):
        self.module_path = module_path
        self.architecture = architecture
        self.model_weights = model_weights

    @property
    def module_path(self):
        return self._architecture_file

    @property
    def architecture(self):
        return self._architecture

    @property
    def model_weights(self):
        return self._model_weights

    @module_path.setter
    def module_path(self, value):
        file_path = '/'.join(value.split('.'))
        file_path = "./{}".format(file_path)
        if not os.path.isfile(file_path):
            raise DirectoryNotExisting()

        self._module_path = value

    @architecture.setter
    def architecture(self, value):
        try:
            spec = importlib.util.spec_from_file_location(
                value, self.module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, value)
            self.model = model_class.model(256, 256)

        except NoSuchArchitectureImpemented() as e:
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
            val_dir=data_dir,
            batch_size=num_of_imgs
        )
        batch_gen.load_data()

        self.samples_x, self.samples_y = next(batch_gen.train_batches)

    def plot_images(self, save_dir):
        """A method creating a figure with test images containing
        actual images and their masks

        :param save_dir: directory where the result is saved
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
                save_dir, 'test_images.png'
            )
        )

    def load_model_weights(self):
        self.model.load_weights(self.model_weights)

    def predict_images(self):
        self._predicted = self.model.predict(self.samples_x)

    def visualize_prediction_on_test(self, save_dir):
        """A method creating a figure with test images containing
        actual images and their masks

        :param save_dir: directory where the result is saved
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
            mask_gen = self._predicted[pos].reshape(224, 224)
            mask_gen = mask_gen.astype(np.uint8)
            plt.imshow(mask_gen)
        plt.axis('off')

        plt.savefig(
            os.path.join(
                save_dir, 'test_images_prediction_result.png'
            )
        )