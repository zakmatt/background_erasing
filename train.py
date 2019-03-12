import argparse
import os
import pandas as pd

from networks.unet import Unet

from utils.batch_generator import BatchGenerator

BATCH_SIZE = 1
IMG_ROWS, IMG_COLS = 256, 256
NB_EPOCHS = 500
STEPS_PER_EPOCH = 500

LUNGS_IMGS = './dataset/lungs_segmentation/lungs_resized/'
LUNGS_MASKS = './dataset/lungs_segmentation/masks_dilated_resized/'


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
    data['image_path'] = [
        '{}{}'.format(LUNGS_IMGS, file_name) for
        file_name in data.file_name.values
    ]
    data['mask_path'] = [
        '{}{}'.format(LUNGS_MASKS, file_name) for
        file_name in data.file_name.values
    ]
    val_batch_size = int(len(data.shape[0]) * validation)

    batch_gen = BatchGenerator(
        data=data, validate=validation, batch_size=BATCH_SIZE
    )
    model = Unet(IMG_ROWS, IMG_COLS, batch_gen,
                 save_model_dir, results_file)

    if initial_epoch > 0:
        model.load_weights(initial_epoch)

    model.train(initial_epoch, NB_EPOCHS, steps_per_epoch)


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
