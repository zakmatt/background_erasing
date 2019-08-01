import cv2
import numpy as np
import os

from math import ceil
from numpy.random import randint, normal
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from utils.augmentation_methods import Augmentor


class NoDataFrame(Exception):
    pass


class DirecotryNotExisting(Exception):
    pass


class BatchGenerator(object):
    """Class generating image batches"""

    def __init__(self, data, validate=0.1,
                 batch_size=1, preprocess=None, segmentation=True, shape=None):
        """

        :param data:
        :type data: DataFrame
        :param validate:
        :param batch_size:
        :param segmentation:
        """

        self.batch_size = batch_size

        if data is None or len(data) == 0:
            raise NoDataFrame()

        if not isinstance(validate, float):
            raise TypeError()

        train_data, validation_data = train_test_split(
            data, test_size=validate
        )
        self.data = train_data
        self.validate = validation_data
        self._num_batches = int(ceil(len(train_data) / self.batch_size))
        self._preprocess = preprocess
        self._segmentation = segmentation
        self._shape = shape

    @property
    def data(self):
        return self._data

    @property
    def validate(self):
        return self._validate

    @data.setter
    def data(self, data):
        if hasattr(self, '_data'):
            print('data attribute already set')
            return

        self._data = data

    @validate.setter
    def validate(self, validate):
        if hasattr(self, '_validate'):
            print('validate attribute already set')
            return

        self._validate = validate

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if not isinstance(batch_size, int):
            raise TypeError()

        self._batch_size = batch_size

    @property
    def num_batches(self):
        return self._num_batches

    @staticmethod
    def _augment(image, mask=None):
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
            if mask is not None:
                mask = Augmentor.mirror(mask)

        return image, mask

    @staticmethod
    def read_images(image_path, preprocess=None, mask_path=None, shape=None):

        # any of files does not exist
        if not os.path.isfile(image_path):
            return None, None

        if mask_path is not None:
            if not os.path.isfile(mask_path):
                return None, None

        img = cv2.imread(image_path)
        # change channels from B,R,G to R,G,B
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if shape is not None:
            img = cv2.resize(img, shape)

        if mask_path is not None:
            mask = cv2.imread(mask_path, 0)
            if shape is not None:
                mask = cv2.resize(mask, shape)
        else:
            mask = None

        # augment pair
        # img, mask = BatchGenerator._augment(img, mask)

        # change types from uint8 to float32
        img = img.astype(np.float32)

        if preprocess:
            img = preprocess(img)

        if mask is not None:
            mask = mask.astype(np.float32)
            mask[mask < 128] = 0.
            mask[mask >= 128] = 1.
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)

        return img, mask

    def _read_batch_pairs(self, data):
        """

        :param data:
        :type data: DataFrame
        :return:
        """

        x_data, y_data = [], []
        for _, row in data.iterrows():
            image_path = row['image_path']

            if self._segmentation:
                mask_path = row['mask_path']
                img, mask = self.read_images(
                    image_path, self._preprocess, mask_path, self._shape
                )
                if mask is None:
                    continue
                y_data.append(mask)
            else:
                img, _ = self.read_images(
                    image_path, preprocess=self._preprocess, shape=self._shape
                )

                y_data.append(row['class'])

            if img is None:
                continue

            x_data.append(img)

        x_data = np.array(x_data, dtype=np.float32)
        y_data = np.array(y_data, dtype=np.float32)

        return x_data, y_data

    @property
    def train_batches(self):

        while True:
            rows = self.data.sample(self._batch_size)
            x_data, y_data = self._read_batch_pairs(rows)
            yield x_data, y_data

    @property
    def validation_batches(self):

        while True:
            rows = self.validate.sample(self._batch_size)
            x_data, y_data = self._read_batch_pairs(rows)
            yield x_data, y_data

    def generate_test_batch(self):

        validation_batch = self.validate.shape[0]
        rows = self.data.sample(validation_batch)
        validate_rows = self.validate.sample(validation_batch)
        x_train_data, y_train_data = self._read_batch_pairs(rows)
        x_val_data, y_val_data = self._read_batch_pairs(validate_rows)

        return (x_train_data, y_train_data), (x_val_data, y_val_data)
