import argparse
import numpy as np

from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam

# from networks.unet_mask_out import Unet
from networks.unet import Unet

from utils.batch_generator import BatchGenerator

BATCH_SIZE = 1
VAL_BATCH = 150
IMG_ROWS, IMG_COLS = 256, 256
NB_EPOCHS = 1001


class LossValidateCallback(Callback):
    def __init__(self, batch_generator, results_file):
        self.batch_generator = batch_generator

        import os
        basedir = os.path.dirname(results_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.results_file = results_file

    @staticmethod
    def IOU_loss(y_true, y_false):
        def IOU_calc(y_true, y_false, smooth=1.):
            y_true_f = y_true.flatten()
            y_false_f = y_false.flatten()
            intersection = np.sum(y_true_f * y_false_f)
            return 2 * (intersection + smooth) / (
                np.sum(y_true_f) + np.sum(y_false_f) + smooth)

        return 1 - IOU_calc(y_true, y_false)

    def on_epoch_end(self, epoch, logs=None):
        # train_losses, val_losses = []
        train_batch, val_batch = self.batch_generator(VAL_BATCH)
        train_imgs, train_masks = train_batch
        val_imgs, val_masks = val_batch
        train_results = self.model.predict(train_imgs)
        val_results = self.model.predict(val_imgs)

        train_losses = [LossValidateCallback.IOU_loss(*pair) for pair in
                        zip(train_masks, train_results)]
        average_train_loss = np.average(train_losses)
        std_train_loss = np.std(train_losses)
        val_losses = [
            LossValidateCallback.IOU_loss(*pair) for pair in
            zip(val_masks, val_results)]
        average_val_loss = np.average(val_losses)
        std_val_loss = np.std(val_losses)

        batch_train_loss = LossValidateCallback.IOU_loss(train_masks,
                                                         train_results)
        batch_val_loss = LossValidateCallback.IOU_loss(val_masks, val_results)

        eval_train_loss, _ = self.model.evaluate(train_imgs, train_masks)
        eval_val_loss, _ = self.model.evaluate(val_imgs, val_masks)

        text = 'epoch: {0} evaluation: train_loss: {1}, ' \
               'validation loss: {2}, '.format(
                   epoch, eval_train_loss, eval_val_loss)
        text += 'batch eval: train loss: {0} validation loss: {1}, '.format(
            batch_train_loss, batch_val_loss
        )
        text += 'batch stats: train avg loss: {0}, train std loss: {1}, ' \
                'val avg loss: {2}, val std loss: {3}\n'.format(
                    average_train_loss, std_train_loss,
                    average_val_loss, std_val_loss
                )
        with open(self.results_file, 'a') as file:
            file.writelines(text)


def train(data_dir, val_data_dir, results_file):
    batch_gen = BatchGenerator(
        data_dir=data_dir, val_data_dir=val_data_dir, batch_size=BATCH_SIZE
    )
    batch_gen.load_data()
    model = Unet.model(IMG_ROWS, IMG_COLS)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=Unet.loss,
        metrics=[Unet.metric]
    )
    checkpoint = ModelCheckpoint(
        filepath='deep_unet_batch_1_epoch_{epoch:02d}.hdf5',
        mode='auto',
        period=50
    )

    model.fit_generator(
        batch_gen.train_batches,
        steps_per_epoch=1e3,
        epochs=NB_EPOCHS,
        callbacks=[
            checkpoint,
            LossValidateCallback(
                batch_gen.generate_test_batch,
                results_file
            )
        ],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_dir",
                        help='Training data directory',
                        required=True)
    parser.add_argument("-v",
                        "--val_data_dir",
                        help='Validation data directory',
                        required=True)
    parser.add_argument("-r",
                        "--results_file",
                        help='Metrics results file path',
                        required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    val_data_dir = args.val_data_dir
    results_file = args.results_file
    train(data_dir, val_data_dir, results_file)
