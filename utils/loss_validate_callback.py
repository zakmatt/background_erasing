import os

from keras.callbacks import Callback

from utils.loss_validate import LossValidate


class LossValidateCallback(LossValidate, Callback):

    def __init__(self, generate_test_batch,
                 results_file, img_cols=256,
                 img_rows=256, is_segment=False):
        self.generate_test_batch = generate_test_batch
        self.img_rows = img_rows
        self.img_cols = img_cols

        basedir = os.path.dirname(results_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.results_file = results_file
        self.is_segment = is_segment

    def on_epoch_end(self, epoch, logs=None):
        if self.is_segment:
            self.error_log_segmentation(epoch)
        else:
            self.error_log_classification(epoch)
