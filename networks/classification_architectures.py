import tensorflow as tf

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
        self.create_model(
            img_rows,
            img_cols,
            tf.keras.applications.VGG16
        )


class ResNet(ClassModel):
    """

    """

    name = 'resnet'

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file):
        ClassModel.__init__(self, img_rows, img_cols, batch_gen,
                            save_model_dir, results_file)
        self.model = None
        self.create_model(
            img_rows,
            img_cols,
            tf.keras.applications.ResNet50
        )


class Inception(ClassModel):
    """

    """

    name = 'inception'

    def __init__(self, img_rows, img_cols, batch_gen,
                 save_model_dir, results_file):
        ClassModel.__init__(self, img_rows, img_cols, batch_gen,
                            save_model_dir, results_file)
        self.model = None
        self.create_model(
            img_rows,
            img_cols,
            tf.keras.applications.InceptionV3
        )
