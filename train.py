from keras.optimizers import Adam

from networks.unet import Unet
from utils.batch_generator import BatchGenerator

DATA_DIR = '/Users/matt/masters_thesis/resized_data/'
BATCH_SIZE = 8
IMG_ROWS, IMG_COLS = 224, 224


def train():
    batch_gen = BatchGenerator(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    batch_gen.load_data()
    model = Unet.model(IMG_ROWS, IMG_COLS)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=Unet.loss, 
        metrics=[Unet.metric]
    )
    model.fit_generator(batch_gen.train_batches, steps_per_epoch=1e3, epochs=1)
    model.save('./unet_batch_8_out_3.h5')
    model.save_weights("unet_batch_8_out_3_weights.h5", overwrite=True)


if __name__ == '__main__':
    train()
