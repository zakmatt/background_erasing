import os

from keras.callbacks import Callback

from utils.loss_validate import LossValidate


class LossValidateCallback(LossValidate, Callback):

    def __init__(self, generate_test_batch,
                 results_file, val_batch_size, img_cols=256,
                 img_rows=256, to_categ=False):
        self.generate_test_batch = generate_test_batch
        self.val_batch_size = val_batch_size
        self.to_categ = to_categ
        self.img_rows = img_rows
        self.img_cols = img_cols

        basedir = os.path.dirname(results_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.results_file = results_file
        self.to_categ = to_categ

    def on_epoch_end(self, epoch, logs=None):
        self.error_log(epoch)
