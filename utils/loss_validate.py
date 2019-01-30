import numpy as np
import os

from keras.utils import to_categorical


class WrongShapeException(Exception):
    pass


def mask_to_categorical(masks, batch_size=1, img_cols=256, img_rows=256):
    masks = to_categorical(masks, 2)
    masks = np.reshape(
        masks,
        (
            batch_size, img_cols * img_rows, 2
        )
    )
    return masks.astype(np.float32)


class LossValidate(object):
    def __init__(
            self, model, generate_test_batch,
            results_file, val_batch_size, img_cols=256,
            img_rows=256, to_categ=False
    ):
        self.generate_test_batch = generate_test_batch
        self.val_batch_size = val_batch_size
        self.to_categ = to_categ
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model = model

        basedir = os.path.dirname(results_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.results_file = results_file

    @property
    def img_cols(self):
        return self._img_cols

    @img_cols.setter
    def img_cols(self, value):
        if not isinstance(value, int):
            raise TypeError()

        if value <= 0:
            raise WrongShapeException()

        self._img_cols = value

    @property
    def img_rows(self):
        return self._img_rows

    @img_rows.setter
    def img_rows(self, value):
        if not isinstance(value, int):
            raise TypeError()

        if value <= 0:
            raise WrongShapeException()

        self._img_rows = value

    @staticmethod
    def IOU_loss(y_true, y_false):
        def IOU_calc(y_true, y_false, smooth=1.):
            y_true_f = y_true.flatten()
            y_false_f = y_false.flatten()
            intersection = np.sum(y_true_f * y_false_f)
            union = np.sum(y_true_f) + np.sum(y_false_f) - intersection
            return (intersection + smooth) / (union + smooth)

        return 1 - IOU_calc(y_true, y_false)

    def _mask_to_categorical(self, masks, batch_size=1):
        if self.to_categ:
            return mask_to_categorical(
                masks=masks, batch_size=batch_size,
                img_cols=self.img_cols, img_rows=self.img_rows
            ).astype(np.float32)

        return masks.astype(np.float32)

    def error_log(self, epoch):
        train_batch, val_batch = self.generate_test_batch(
            self.val_batch_size
        )
        train_imgs, train_masks = train_batch
        val_imgs, val_masks = val_batch
        train_results = self.model.predict(train_imgs)
        val_results = self.model.predict(val_imgs)

        # change categorical
        train_losses = [
            LossValidate.IOU_loss(
                self._mask_to_categorical(pair[0]), pair[1]
            )
            for pair in zip(train_masks, train_results)
        ]
        average_train_loss = np.average(train_losses)
        std_train_loss = np.std(train_losses)

        # change categorical
        val_losses = [
            LossValidate.IOU_loss(
                self._mask_to_categorical(pair[0]), pair[1]
            )
            for pair in zip(val_masks, val_results)
        ]
        average_val_loss = np.average(val_losses)
        std_val_loss = np.std(val_losses)

        # change categorical if needed
        train_size = len(train_masks)
        train_masks = self._mask_to_categorical(train_masks, train_size)
        batch_train_loss = LossValidate.IOU_loss(
            train_masks, train_results
        )

        # change categorical if needed
        val_size = len(val_masks)
        val_masks = self._mask_to_categorical(val_masks, val_size)
        batch_val_loss = LossValidate.IOU_loss(
            val_masks, val_results
        )

        text = '{0},{1},{2},'.format(epoch, batch_train_loss, batch_val_loss)
        text += '{0},{1},{2},{3}\n'.format(
            average_train_loss, std_train_loss,
            average_val_loss, std_val_loss
        )
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as file:
                columns = 'epoch,batch_train_loss,batch_val_loss,'
                columns += 'batch_stats_train_avg_loss,'
                columns += 'batch_stats_train_std_loss,'
                columns += 'batch_stats_val_avg_loss,'
                columns += 'batch_stats_val_std_loss\n'
                file.writelines(columns)

        with open(self.results_file, 'a') as file:
            file.writelines(text)
