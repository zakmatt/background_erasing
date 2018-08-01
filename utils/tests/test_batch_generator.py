import numpy as np

from unittest import TestCase
from batch_generator import BatchGenerator, NoDataPath

DATA_PATH = 'test_data/'
VAL_DATA_PATH = 'test_val_data/'


class TestBaseGenerator(TestCase):
    """Class testing a batch generator"""

    def setUp(self):
        self.batch_gen = BatchGenerator(DATA_PATH, VAL_DATA_PATH, 2)
        self.batch_gen.load_data()

    def test_check_init(self):
        self.assertEqual(self.batch_gen.data_dir, DATA_PATH)
        self.assertEqual(self.batch_gen.batch_size, 2)

    def test_no_data_path(self):
        with self.assertRaises(NoDataPath):
            self.batch_gen.data_dir = None

        with self.assertRaises(NoDataPath):
            self.batch_gen.data_dir = ''

    def test_load_files_names(self):
        f_names = np.array([
            ['test_data/000000117764.jpg', 'test_data/000000117764_mask.jpg'],
            ['test_data/000000117857.jpg', 'test_data/000000117857_mask.jpg'],
            ['test_data/000000118061.jpg', 'test_data/000000118061_mask.jpg']
        ])
        np.testing.assert_equal(self.batch_gen.images_pairs, f_names)

    def test_num_batches(self):
        self.assertEqual(self.batch_gen.num_batches, 2)

    def test_loaded_dataset_shape(self):
        x_shape = (2, 256, 256, 3)
        y_shape = (2, 256, 256, 3)
        x, y = next(self.batch_gen.train_batches)
        self.assertEqual(x.shape, x_shape)
        self.assertEqual(y.shape, y_shape)
        self.assertEqual(y.max(), 1)
        self.assertEqual(y.min(), 0)

    def test_generate_test_batch(self):
        batch_shape = (1, 256, 256, 3)
        train_batch, val_batch = self.batch_gen.generate_test_batch(
            batch_size=1
        )
        self.assertEqual(train_batch[0].shape, batch_shape)
        self.assertEqual(train_batch[1].shape, batch_shape)
        self.assertEqual(val_batch[0].shape, batch_shape)
        self.assertEqual(val_batch[1].shape, batch_shape)
