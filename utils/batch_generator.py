import glob
import imageio
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
        """Load files names from a given directory

        Images are loaded in pairs; image - mask

        """

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

        self._images_pairs = np.array(files)
        self._dataset_size = len(files)
        self._batch_size = (self._batch_size if self.batch_size < len(files)
                            else len(files))

        self._num_batches = int(ceil(len(files) / self.batch_size))

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

    @staticmethod
    def read_images(img_pair):
        img_path, mask_path = img_pair

        # any of files does not exist
        if not os.path.isfile(img_path) or \
                not os.path.isfile(mask_path):
            return None, None

        img = imageio.imread(img_path).astype(np.float32)
        mask = imageio.imread(mask_path).astype(np.float32)
        mask[mask < 128] = 0.
        mask[mask >= 128] = 1.
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        return img, mask

    @property
    def train_batches(self):

        for batch_pos in range(self._num_batches):

            start_range = batch_pos * self._batch_size
            end_range = (batch_pos + 1) * self._batch_size
            pairs = self._images_pairs[start_range:end_range]
            x_data, y_data = [], []
            for pair in pairs:
                img, mask = BatchGenerator.read_images(pair)

                if img is None or mask is None:
                    continue

                x_data.append(img)
                y_data.append(mask)

            x_data = np.array(x_data, dtype=np.float32)
            y_data = np.array(y_data, dtype=np.float32)

            yield x_data, y_data

    @property
    def test_batch(self):
        pass
