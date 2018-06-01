import cv2 as cv
import glob
import numpy as np
import os

from math import ceil


class NoDataPath(Exception):
    pass


class DirecotryNotExisting(Exception):
    pass


class BatchGenerator(object):
    """Class generating image batches"""

    def __init__(self, data_dir, batch_size=1):
        self._train_batch_pos = 0
        self._test_batch_pos = 0

        if type(batch_size) is not int:
            raise TypeError()

        self._batch_size = batch_size

        if not data_dir:
            raise NoDataPath()

        if not os.path.isdir(data_dir):
            raise DirecotryNotExisting()

        self._data_dir = data_dir

    def load_data(self):
        files = glob.glob(os.path.join(self._data_dir, '*.jpg'))
        files = [f for f in files if '_mask' not in f]
        files = sorted(
            map(
                lambda f_name:
                    (f_name, '{}_mask.jpg'.format(f_name.split('.')[0])),
                files
            ),
            key=lambda pair: pair[0]
        )

        self._images_pairs = files

        self._num_batches = int(ceil(len(files)/self.batch_size))

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        if not data_dir:
            raise NoDataPath()

        self._data_dir = data_dir

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if type(batch_size) is not int:
            raise TypeError()

        self._batch_size = batch_size

    @property
    def images_pairs(self):
        return self._images_pairs

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def train_batches(self):
        self._train_batch_pos += 1
        x_data = []
        y_data = []
        for batch_pos in range(self._num_batches):
            for pair in self._images_pairs[
                        batch_pos * self._batch_size:(
                                    batch_pos + 1) * self._batch_size]:
                img_path, mask_path = pair
                img = cv.imread(img_path).astype(np.float32)
                mask = cv.imread(mask_path).astype(np.float32)
                x_data.append(img)
                y_data.append(mask)

                self._train_batch_pos += 1

            x_data = np.array(x_data, dtype=np.float32)
            y_data = np.array(y_data, dtype=np.float32)

            yield x_data, y_data

    @property
    def test_batch(self):
        pass