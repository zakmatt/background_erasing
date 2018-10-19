import argparse
import os

from networks.DCGAN import DCGAN

from utils.batch_generator import BatchGenerator

BATCH_SIZE = 1
VAL_BATCH = 150
IMG_ROWS, IMG_COLS = 256, 256
NB_EPOCHS = 1000


def train(
        data_dir, val_dir, results_file, save_model_dir, model_info=None
):
    """Model training main script

    :param data_dir: Directory to a training dataset
    :type data_dir: str
    :param val_dir: Directory to a validation dataset
    :param results_file: name of a file where the results are to be stored
    :param save_model_dir: path to a place where the results are to be saved
    :param model_info: information whether we start training from a particular
    epoch and with pretrained weights
    :type model_info: dict
    :return:
    """

    batch_gen = BatchGenerator(
        data_dir=data_dir, val_dir=val_dir, batch_size=BATCH_SIZE
    )
    batch_gen.load_data()
    model = DCGAN(IMG_ROWS, IMG_COLS, batch_gen,
                  save_model_dir, results_file, VAL_BATCH)

    initial_epoch = 0
    if model_info:
        weights_path = model_info['weights_path']
        model.load_weights(weights_path)
        if model_info['initial_epoch']:
            initial_epoch = model_info['initial_epoch']

    model.train(initial_epoch, NB_EPOCHS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_dir",
                        help='Training data directory',
                        required=True)
    parser.add_argument("-v",
                        "--val_dir",
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
    val_dir = args.val_dir
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
        train(data_dir, val_dir, results_file, save_model_dir, model_info)
    else:
        train(data_dir, val_dir, results_file, save_model_dir)
