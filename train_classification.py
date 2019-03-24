import argparse
import os
import pandas as pd

from keras.utils import to_categorical

from networks.classification_architectures import (
    Inception,
    ResNet,
    VGG16_N
)
from utils.batch_generator import BatchGenerator

BATCH_SIZE = 32
IMG_SHAPE = (256, 256, 3)
NB_EPOCHS = 150
STEPS_PER_EPOCH = 200

LUNGS_DIR = './dataset/final_classification/'


def train(data_path, validation, results_file,
          save_model_dir, steps_per_epoch=STEPS_PER_EPOCH, initial_epoch=0):
    """Model training main script

    :param data: data_path
    :param validation: Dataset split portion for validation
    :type validation: float
    :param results_file: name of a file where the results are to be stored
    :param save_model_dir: path to a place where the results are to be saved
    :param model_info: information whether we start training from a particular
    epoch and with pretrained weights
    :type model_info: dict
    :return:
    """

    data = pd.read_csv(data_path)
    data['image_path'] = data.file.apply(
        lambda x: os.path.join(LUNGS_DIR, x))
    data['class'] = list(to_categorical(data['class']))

    batch_gen = BatchGenerator(
        data=data, validate=validation,
        batch_size=BATCH_SIZE, segmentation=False,
        shape=IMG_SHAPE[:-1]
    )
    model = VGG16_N(
        IMG_SHAPE[0], IMG_SHAPE[1], batch_gen,
        save_model_dir, results_file
    )

    if initial_epoch > 0:
        model.load_weights(initial_epoch)

    model.train(initial_epoch, NB_EPOCHS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_path",
                        help='Training data info csv file',
                        required=True)
    parser.add_argument("-v",
                        "--val",
                        help='Validation split',
                        required=True)
    parser.add_argument("-r",
                        "--results_file",
                        help='Metrics results file path',
                        required=True)
    parser.add_argument("-e",
                        "--initial_epoch",
                        help='Initial epoch #',
                        required=False)
    parser.add_argument("-s",
                        "--save_model_dir",
                        help='Save model dir',
                        required=False)
    args = parser.parse_args()
    data_path = args.data_path
    val = float(args.val)
    results_file = args.results_file
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

    if initial_epoch:
        initial_epoch = int(initial_epoch)
        train(data_path, val, results_file, save_model_dir, initial_epoch)
    else:
        train(data_path, val, results_file, save_model_dir)
