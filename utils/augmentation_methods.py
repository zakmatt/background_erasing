import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Augment(object):
    """Data augmnentation class

        A static class containing methods serving for data augmentation;
        - brightness augmentation
        - image stretching
        - image translation
        - mirror reflection

    """

    @staticmethod
    def augment_brightness(image):
        """Change an image brightness

        :return: an image of changed brightness
        """
        rand_brightness = .25 + np.random.uniform()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = image[:, :, 2] * rand_brightness
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

    @staticmethod
    def change_contrast(image):
        """

        :return:
        """
        factor = (255 - 100 * np.random.uniform()) / (
            255 + 100 * np.random.uniform()
        )

        def contrast(c):
            return 128 + factor * (c - 128)

        return image.astype(np.uint8)

    @staticmethod
    def mirror(image):
        """Mirror reflection of an input image

        :return: image
        """

        return image[::-1]


    # using this method is not fully recommended as the transformation time is
    # pretty long this it will slow down the training process

    @staticmethod
    def elastic_transform(image, mask, alpha, sigma, alpha_affine,
                          random_state=None):
        """Elastic deformation of images as described in [Simard2003]_
        (with modifications). [Simard2003] Simard, Steinkraus and Platt,
        "Best Practices for Convolutional Neural Networks applied to
        Visual Document Analysis", in Proc. of the International Conference
        on Document Analysis and Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         Source: https://www.kaggle.com/bguberfain/elastic-transform-for-data-
                                                                augmentation
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size,
                            center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                           size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1],
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, shape_size[::-1],
                              borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                             sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                             sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                              np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,
                                                          (-1, 1)), np.reshape(
            z, (-1, 1))

        image = map_coordinates(image, indices, order=1,
                                mode='reflect').reshape(shape)
        mask = map_coordinates(mask, indices, order=1, mode='reflect').reshape(
            shape)

        return image, mask
