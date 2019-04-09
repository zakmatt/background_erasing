import argparse
import os
import pandas as pd
import tensorflow as tf

from networks.classification_architectures import (
    Inception,
    ResNet,
    VGG16_N
)
from utils.batch_generator import BatchGenerator

BATCH_SIZE = 32
IMG_SHAPE = (256, 256, 3)
NB_EPOCHS = 1#50
STEPS_PER_EPOCH = 1#200

SEGMENTED_LUNGS_DIR = './dataset/segmented_classification_jpg/'
NON_SEGMENTED_LUNGS_DIR = './dataset/classification_jpg/'


def train(data_path, validation, results_file,
          save_model_dir, initial_epoch=0):
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

    for img_path, is_segmented in [
        (SEGMENTED_LUNGS_DIR, 'segmented'),
        (NON_SEGMENTED_LUNGS_DIR, 'non_segmented')
    ]:
        data = pd.read_csv(data_path)
        data['image_path'] = data.file.apply(
            lambda x: os.path.join(img_path, x))
        data['class'] = list(tf.keras.utils.to_categorical(data['class']))

        models_result_path = [
            # (model, save path, rescale)
            # (VGG16_N, 'vgg/{}/'.format(is_segmented), False),
            (Inception, 'inception/{}/'.format(is_segmented), True),
            (ResNet, 'resnet/{}/'.format(is_segmented), True),
        ]
        for model_arch, path, rescale in models_result_path:

            batch_gen = BatchGenerator(
                data=data, validate=validation,
                batch_size=BATCH_SIZE, segmentation=False,
                shape=IMG_SHAPE[:-1], rescale=rescale
            )

            results_save_dir = os.path.join(
                save_model_dir, path
            )
            results_file_name = '{}_{}'.format(
                model_arch.name, results_file
            )
            model = model_arch(
                IMG_SHAPE[0], IMG_SHAPE[1], batch_gen,
                results_save_dir, results_file_name
            )

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
