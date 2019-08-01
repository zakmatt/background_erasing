from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

from networks.classification_model import ClassModel


# noinspection PyPep8Naming
class VGG16_N(ClassModel):
    """

    """

    name = 'vgg'

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file):
        VGG16_N.__name__ = 'vgg'
        ClassModel.__init__(self, img_rows, img_cols, batch_gen,
                            save_model_dir, results_file)
        self.model = None
        self.create_model(img_rows, img_cols, VGG16)


class ResNet(ClassModel):
    """

    """

    name = 'resnet'

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file):
        ClassModel.__init__(self, img_rows, img_cols, batch_gen,
                            save_model_dir, results_file)
        self.model = None
        self.create_model(img_rows, img_cols, ResNet50)


class Inception(ClassModel):
    """

    """

    name = 'inception'

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file):
        ClassModel.__init__(self, img_rows, img_cols, batch_gen,
                            save_model_dir, results_file)
        self.model = None
        self.create_model(img_rows, img_cols, InceptionV3)
