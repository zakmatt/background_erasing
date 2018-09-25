import cv2
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Augmentor(object):
    """Data augmnentation class

        A static class containing methods serving for data augmentation;
        - brightness augmentation
        - image stretching
        - image translation
        - blurring
        - mirror reflection
        - elastic image transformation

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
    def augment_translate(image, mask, trans_range):
        """Translate an image and a mask

        :param image: input image to translate
        :param mask: input mask to translate
        :param trans_range: applied translation range; same for image and mask
        :return: translated image and mask
        """
        assert image.shape == mask.shape

        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        trans_M = np.array([[1, 0, tr_x], [0, 1, tr_y]], dtype=np.float32)
        width, height, _ = image.shape
        image = cv2.warpAffine(image, trans_M, (width, height))
        mask = cv2.warpAffine(mask, trans_M, (width, height))
        return image, mask

    @staticmethod
    def augment_stretch(img, mask, scale_range):
        """Streatch an image and a mask

        :param image: input image to translate
        :param mask: input mask to translate
        :param scale_range: applied streatching range; same for image and mask
        :return: streatched image and mask
        """
        assert img.shape == mask.shape

        tr_x1 = scale_range * np.random.uniform()
        tr_y1 = scale_range * np.random.uniform()
        p1 = (tr_x1, tr_y1)
        tr_x2 = scale_range * np.random.uniform()
        tr_y2 = scale_range * np.random.uniform()
        p2 = (img.shape[0] - tr_x2, tr_y1)

        p3 = (img.shape[0] - tr_x2, img.shape[1] - tr_y2)
        p4 = (tr_x1, img.shape[1] - tr_y2)

        pts1 = np.float32([[p1[0], p1[1]],
                           [p2[0], p2[1]],
                           [p3[0], p3[1]],
                           [p4[0], p4[1]]])
        pts2 = np.float32([[0, 0],
                           [img.shape[1], 0],
                           [img.shape[1], img.shape[0]],
                           [0, img.shape[0]]]
                          )

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (img.shape[0], img.shape[1]))
        img = np.array(img, dtype=np.uint8)
        mask = cv2.warpPerspective(mask, M, (mask.shape[0], mask.shape[1]))
        mask = np.array(mask, dtype=np.uint8)

        return img, mask

    @staticmethod
    def blur(image):
        """Blur an input image

        :return: image
        """

        return cv2.blur(image, (5, 5))

    @staticmethod
    def mirror(image):
        """Get a mirror reflection of an image

        :return: image
        """

        return cv2.flip(image, 1)


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
