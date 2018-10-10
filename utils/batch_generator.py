import glob
import cv2
import numpy as np
import os

from math import ceil
from numpy.random import randint, normal

from utils.augmentation_methods import Augmentor


class NoDataPath(Exception):
    pass


class DirecotryNotExisting(Exception):
    pass


class BatchGenerator(object):
    """Class generating image batches"""

    def __init__(self, data_dir, val_dir, batch_size=1):
        self._train_batch_pos = 0
        self._test_batch_pos = 0

        if not isinstance(batch_size, int):
            raise TypeError()

        self._batch_size = batch_size

        if not data_dir:
            raise NoDataPath()

        if not os.path.isdir(data_dir):
            raise DirecotryNotExisting()

        self._data_dir = data_dir
        self._val_dir = val_dir

    @staticmethod
    def _get_files_names(data_dir):
        files = glob.glob(os.path.join(data_dir, '*.jpg'))
        files = [f for f in files if '_mask' not in f]
        files = sorted(
            map(
                lambda f_name:
                (f_name, '{}_mask.jpg'.format(f_name.split('.')[0])),
                files
            ),
            key=lambda pair: pair[0]
        )
        return files

    def load_data(self):
        """Load files names from a given directory

        Images are loaded in pairs; image - mask

        """

        train_files = BatchGenerator._get_files_names(self._data_dir)
        val_files = BatchGenerator._get_files_names(self._val_data_dir)
        self._images_pairs = np.array(train_files)
        self._val_images_pairs = np.array(val_files)
        self._dataset_size = len(train_files)
        self._batch_size = (
            self._batch_size if self.batch_size < len(train_files)
            else len(train_files)
        )

        self._num_batches = int(ceil(len(train_files) / self.batch_size))

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def val_dir(self):
        return self._val_data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        if not data_dir:
            raise NoDataPath()

        self._data_dir = data_dir

    @val_dir.setter
    def val_dir(self, val_data_dir):
        if not val_data_dir:
            raise NoDataPath()

        self._val_data_dir = val_data_dir

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if not isinstance(batch_size, int):
            raise TypeError()

        self._batch_size = batch_size

    @property
    def images_pairs(self):
        return self._images_pairs

    @property
    def num_batches(self):
        return self._num_batches

    @staticmethod
    def _augment(image, mask):
        # generate options from 0 to 5
        # 0 means do nothing
        rnd_selection = randint(6)

        if rnd_selection == 1:
            # brightness
            image = Augmentor.augment_brightness(image)
        if rnd_selection == 2:
            # translation
            image, mask = Augmentor.augment_translate(
                image, mask, normal(50, 25)
            )
        if rnd_selection == 3:
            image, mask = Augmentor.augment_stretch(
                image, mask, normal(30, 10)
            )
        if rnd_selection == 4:
            image = Augmentor.blur(image)
        if rnd_selection == 5:
            image = Augmentor.mirror(image)
            mask = Augmentor.mirror(mask)

        return image, mask

    @staticmethod
    def read_images(img_pair):
        img_path, mask_path = img_pair

        # any of files does not exist
        if not os.path.isfile(img_path) or \
                not os.path.isfile(mask_path):
            return None, None

        img = cv2.imread(img_path)
        # change channels from B,R,G to R,G,B
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        # augment pair
        img, mask = BatchGenerator._augment(img, mask)

        # change types from uint8 to float32
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        mask[mask < 128] = 0.
        mask[mask >= 128] = 1.
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        return img, mask

    @staticmethod
    def _read_batch_pairs(pairs):
        x_data, y_data = [], []
        for pair in pairs:
            img, mask = BatchGenerator.read_images(pair)

            if img is None or mask is None:
                continue

            x_data.append(img)
            y_data.append(mask)

        x_data = np.array(x_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.float32)

        return x_data, y_data

    @property
    def train_batches(self):

        while True:
            idx = np.random.choice(
                range(len(self._images_pairs)),
                self._batch_size,
                replace=False
            )
            pairs = self._images_pairs[idx]
            x_data, y_data = BatchGenerator._read_batch_pairs(pairs)

            yield x_data, y_data

    def generate_test_batch(self, batch_size):
        def idxs(imgs_pair):
            return np.random.choice(
                range(len(imgs_pair)),
                batch_size,
                replace=False
            )
        train_idx = idxs(self._images_pairs)
        train_pairs = self._images_pairs[train_idx]
        val_idx = idxs(self._val_images_pairs)
        val_pairs = self._val_images_pairs[val_idx]

        train_batch = BatchGenerator._read_batch_pairs(train_pairs)
        val_batch = BatchGenerator._read_batch_pairs(val_pairs)

        return train_batch, val_batch
