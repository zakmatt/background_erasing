import argparse


from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

#from networks.unet_mask_out import Unet
from networks.deep_unet import Unet

from utils.batch_generator import BatchGenerator

BATCH_SIZE = 1
IMG_ROWS, IMG_COLS = 256, 256
NB_EPOCHS = 1001


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
        filepath='deep_unet_batch_1_epoch_{epoch:02d}.hdf5',
        mode='auto',
        period=50
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
