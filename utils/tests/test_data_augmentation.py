import cv2
import numpy as np

from unittest import TestCase
from augmentation_methods import Augmentor


class TestAugmentor(TestCase):
    """Class testing data augmentation methods"""

    def setUp(self):
        self.img = cv2.imread('test_data/000000117764.jpg')
        self.mask = cv2.imread('test_data/000000117764_mask.jpg')
        self.augmentator = Augmentor()

    def test_augment_brightness(self):
        bright_img = self.augmentator.augment_brightness(self.img)

        self.assertEqual(bright_img.dtype, np.dtype('uint8'))
        self.assertEqual(bright_img.shape, self.img.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(bright_img, self.img)
            ),
            True
        )

    def test_augment_translationt(self):
        translated_img, translated_mask = self.augmentator.augment_translation(
            self.img,
            self.mask,
            np.random.normal(50, 25)
        )

        self.assertEqual(translated_img.dtype, np.dtype('uint8'))
        self.assertEqual(translated_img.shape, self.img.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(translated_img, self.img)
            ),
            True
        )

        self.assertEqual(translated_mask.dtype, np.dtype('uint8'))
        self.assertEqual(translated_mask.shape, self.mask.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(translated_mask, self.mask)
            ),
            True
        )

    def test_augment_translationt(self):
        translated_img, translated_mask = self.augmentator.augment_translate(
            self.img,
            self.mask,
            np.random.normal(50, 25)
        )

        self.assertEqual(translated_img.dtype, np.dtype('uint8'))
        self.assertEqual(translated_img.shape, self.img.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(translated_img, self.img)
            ),
            True
        )

        self.assertEqual(translated_mask.dtype, np.dtype('uint8'))
        self.assertEqual(translated_mask.shape, self.mask.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(translated_mask, self.mask)
            ),
            True
        )

    def test_blurring(self):
        blurred_img = self.augmentator.blur(self.img)
        np.testing.assert_equal(
            blurred_img,
            cv2.blur(self.img, (5, 5))
        )

    def test_mirror(self):
        mirror_image = self.augmentator.mirror(self.img)
        np.testing.assert_equal(
            mirror_image,
            cv2.flip(self.img, 1)
        )

    def test_elastic_transform(self):
        img_elastic, mask_elastic = self.augmentator.elastic_transform(
            self.img, self.mask, np.random.normal(50, 30),
            np.random.normal(50, 30), np.random.normal(50, 30)
        )

        self.assertEqual(img_elastic.dtype, np.dtype('uint8'))
        self.assertEqual(img_elastic.shape, self.img.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(img_elastic, self.img)
            ),
            True
        )

        self.assertEqual(mask_elastic.dtype, np.dtype('uint8'))
        self.assertEqual(mask_elastic.shape, self.mask.shape)

        # assert that arrays are not equal
        np.testing.assert_equal(
            np.any(
                np.not_equal(mask_elastic, self.mask)
            ),
            True
        )
