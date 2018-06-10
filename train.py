import argparse

from bleach import callbacks
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from networks.unet import Unet
from utils.batch_generator import BatchGenerator

BATCH_SIZE = 8
IMG_ROWS, IMG_COLS = 224, 224
NB_EPOCHS=1001


def train(data_dir):
    batch_gen = BatchGenerator(data_dir=data_dir, batch_size=BATCH_SIZE)
    batch_gen.load_data()
    model = Unet.model(IMG_ROWS, IMG_COLS)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=Unet.loss,
        metrics=[Unet.metric]
    )
    checkpoint = ModelCheckpoint(
        filepath='unet_batch_8_out_3_epoch_{epoch:02d}.hdf5',
        mode='auto',
        period=100
    )

    model.fit_generator(
        batch_gen.train_batches,
        steps_per_epoch=1e3,
        epochs=NB_EPOCHS,
        callbacks=[checkpoint],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_dir",
                        help='Training data directory',
                        required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    train(data_dir)
