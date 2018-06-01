import glob

from unittest import TestCase
from batch_generator import BatchGenerator, NoDataPath

DATA_PATH = 'test_data/'

class TestBaseGenerator(TestCase):
    """Class testing a batch generator"""

    def setUp(self):
        self.batch_gen = BatchGenerator(DATA_PATH)
        self.batch_gen.load_data()

    def test_check_init(self):
        self.assertEqual(self.batch_gen.data_dir, DATA_PATH)
        self.assertEqual(self.batch_gen.batch_size, 1)


    def test_no_data_path(self):
        with self.assertRaises(NoDataPath):
            self.batch_gen.data_dir = None

        with self.assertRaises(NoDataPath):
            self.batch_gen.data_dir = ''

    def test_load_files_names(self):
        f_names = [
            ('test_data/000000000110.jpg', 'test_data/000000000110_mask.jpg'),
            ('test_data/000000000370.jpg', 'test_data/000000000370_mask.jpg')
        ]
        self.assertEqual(self.batch_gen.images_pairs, f_names)

    def test_num_batches(self):
        self.assertEqual(self.batch_gen.num_batches, 2)

    def test_loaded_dataset_shape(self):
        shape = (1, 224, 224, 3)
        x, y = next(self.batch_gen.train_batches)
        self.assertEqual(x.shape, shape)
        self.assertEqual(y.shape, shape)
