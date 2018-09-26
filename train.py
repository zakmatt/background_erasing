import argparse
import os
import numpy as np

from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from keras.utils import to_categorical

# from networks.unet_mask_out import Unet
from networks.unet import Unet

from utils.batch_generator import BatchGenerator

BATCH_SIZE = 1
VAL_BATCH = 150
IMG_ROWS, IMG_COLS = 256, 256
NB_EPOCHS = 10#00


def mask_to_categorical(masks, batch_size=1):
    masks = to_categorical(masks, 2)
    masks = np.reshape(
        masks,
        (
            batch_size, IMG_COLS * IMG_ROWS, 2
        )
    )
    return masks.astype(np.float32)


class LossValidateCallback(Callback):

    def __init__(self, batch_generator, results_file):
        self.batch_generator = batch_generator

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
            union = np.sum(y_true_f) + np.sum(y_false_f) - intersection
            return (intersection + smooth) / (union + smooth)

        return 1 - IOU_calc(y_true, y_false)

    def on_epoch_end(self, epoch, logs=None):
        train_batch, val_batch = self.batch_generator(VAL_BATCH)
        train_imgs, train_masks = train_batch
        val_imgs, val_masks = val_batch
        train_results = self.model.predict(train_imgs)
        val_results = self.model.predict(val_imgs)

        # change categorical
        train_losses = [
            LossValidateCallback.IOU_loss(
                mask_to_categorical(pair[0]), pair[1]
            )
            for pair in zip(train_masks, train_results)
        ]
        average_train_loss = np.average(train_losses)
        std_train_loss = np.std(train_losses)

        # change categorical
        val_losses = [
            LossValidateCallback.IOU_loss(
                mask_to_categorical(pair[0]), pair[1]
            )
            for pair in zip(val_masks, val_results)
        ]
        average_val_loss = np.average(val_losses)
        std_val_loss = np.std(val_losses)

        # change categorical
        train_size = len(train_masks)
        batch_train_loss = LossValidateCallback.IOU_loss(
            mask_to_categorical(train_masks, train_size), train_results
        )

        # change categorical
        val_size = len(val_masks)
        batch_val_loss = LossValidateCallback.IOU_loss(
            mask_to_categorical(val_masks, val_size), val_results
        )

        eval_train_loss, _ = self.model.evaluate(
            train_imgs, mask_to_categorical(train_masks, train_size)
        )
        eval_val_loss, _ = self.model.evaluate(
            val_imgs, mask_to_categorical(val_masks, val_size)
        )

        text = '{0}, {1}, {2}, '.format(epoch, eval_train_loss, eval_val_loss)
        text += '{0}, {1}, '.format(batch_train_loss, batch_val_loss)
        text += '{0}, {1}, {2}, {3}\n'.format(
            average_train_loss, std_train_loss,
            average_val_loss, std_val_loss
        )
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as file:
                columns = 'epoch, eval_train_loss, eval_validation_loss, '
                columns += 'batch_train_loss, batch_val_loss, '
                columns += 'batch_stats_train_avg_loss, '
                columns += 'batch_stats_train_std_loss, '
                columns += 'batch_stats_val_avg_loss, '
                columns += 'batch_stats_val_std_loss\n'
                file.writelines(columns)

        with open(self.results_file, 'a') as file:
            file.writelines(text)


def train(
        data_dir, val_data_dir, results_file, save_model_dir, model_info=None
):
    batch_gen = BatchGenerator(
        data_dir=data_dir, val_data_dir=val_data_dir, batch_size=BATCH_SIZE
    )
    batch_gen.load_data()
    model = Unet.model(IMG_ROWS, IMG_COLS)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss='categorical_crossentropy',  # Unet.loss,
        metrics=[Unet.metric]
    )
    initial_epoch = 0
    if model_info:
        weights_path = model_info['weights_path']
        model.load_weights(weights_path)
        if model_info['initial_epoch']:
            initial_epoch = model_info['initial_epoch']

    f_path = str(save_model_dir) + '/deep_unet_batch_1_epoch_{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath=f_path,
        mode='auto',
        period=50
    )

    def get_batch():
        while True:
            x, mask = next(batch_gen.train_batches)
            mask = to_categorical(mask, 2)
            mask = np.reshape(
                mask,
                (
                    batch_gen.batch_size, IMG_COLS*IMG_ROWS, 2
                )
            )
            yield x, mask

    model.fit_generator(
        get_batch(), # batch_gen.train_batches,
        steps_per_epoch=10,#e3,
        epochs=NB_EPOCHS,
        callbacks=[
            checkpoint,
            LossValidateCallback(
                batch_gen.generate_test_batch,
                results_file
            )
        ],
        initial_epoch=initial_epoch
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
    parser.add_argument("-w",
                        "--weights_path",
                        help='Pretrained model weights',
                        required=False)
    parser.add_argument("-e",
                        "--initial_epoch",
                        help='Initial epoch #',
                        required=False)
    parser.add_argument("-s",
                        "--save_model_dir",
                        help='Save model dir',
                        required=False)
    args = parser.parse_args()
    data_dir = args.data_dir
    val_data_dir = args.val_data_dir
    results_file = args.results_file
    weights_path = args.weights_path
    initial_epoch = args.initial_epoch
    save_model_dir = args.save_model_dir

    if not save_model_dir:
        save_model_dir = '.'
    else:
        save_model_dir = str(save_model_dir)
        if save_model_dir[-1] == '/':
            save_model_dir = save_model_dir[:-1]
        if not os.path.isdir(save_model_dir):
            os.makedirs(save_model_dir)

    if weights_path:
        model_info = {}
        if os.path.exists(weights_path):
            model_info['weights_path'] = weights_path
        if initial_epoch:
            initial_epoch = int(initial_epoch)
            model_info['initial_epoch'] = initial_epoch
        train(data_dir, val_data_dir, results_file, save_model_dir, model_info)
    else:
        train(data_dir, val_data_dir, results_file, save_model_dir)
