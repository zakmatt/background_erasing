import argparse


from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam

#from networks.unet_mask_out import Unet
from networks.unet import Unet

from utils.batch_generator import BatchGenerator

BATCH_SIZE = 1
VAL_BATCH = 150
IMG_ROWS, IMG_COLS = 256, 256
NB_EPOCHS = 1001


class LossValidateCallback(Callback):

    def __init__(self, train_batch, val_batch, results_file):
        self.train_batch = train_batch
        self.val_batch = val_batch

        import os
        basedir = os.path.dirname(results_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        self.results_file = results_file

    def on_epoch_end(self, epoch, logs=None):
        train_imgs, train_masks = self.train_batch
        val_imgs, val_masks = self.val_batch
        train_loss, _ = self.model.evaluate(train_imgs, train_masks)
        val_loss, _ = self.model.evaluate(val_imgs, val_masks)
        text = 'epoch: {0} train_loss: {1}, validation loss: {2}\n'.format(
            epoch, train_loss, val_loss
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
                *batch_gen.generate_test_batch(VAL_BATCH),
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
