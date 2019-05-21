import os
import ssl
from argparse import ArgumentParser
from keras.applications import ResNet50, VGG19, VGG16, InceptionV3, MobileNetV2, InceptionResNetV2, Xception, NASNetLarge, MobileNet, DenseNet201, NASNetMobile
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
BATCH_SIZE = 16

MODELS = {
    'DenseNet201': {'IMG_SIZE': 224, 'PROCESSING': 'torch', 'TRANSFER_LEARNING': DenseNet201},
    'MobileNetV2': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': MobileNetV2},
    'VGG19': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': VGG19},
    'NASNetMobile': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': NASNetMobile},
    'InceptionResNetV2': {'IMG_SIZE': 299, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionResNetV2},
    'InceptionV3': {'IMG_SIZE': 299, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionV3},
    'ResNet50': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': ResNet50},
    'Xception': {'IMG_SIZE': 299, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': Xception},
}


def extract_features(generator, model):
    input_shape = (model['IMG_SIZE'], model['IMG_SIZE'], 3)
    pretrained_model = model['TRANSFER_LEARNING'](
        weights='imagenet', include_top=False, input_shape=input_shape)
    # Due to horizontal flipping we have 2x samples
    sample_count = generator.samples
    print('Sample count: {}'.format(sample_count))
    features = np.zeros(shape=(sample_count, pretrained_model.output_shape[1] *
                               pretrained_model.output_shape[2] *
                               pretrained_model.output_shape[3]))
    labels = np.zeros(shape=(sample_count))

    i = 0
    for inputs_batch, labels_batch in tqdm(generator):
        features_batch = pretrained_model.predict(
            preprocess_input(inputs_batch, mode=model['PROCESSING']))
        op_shape = features_batch.shape

        features_batch = np.reshape(features_batch, (
            inputs_batch.shape[0], op_shape[-3] * op_shape[-2] * op_shape[-1]))

        features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break

        features = np.reshape(features, (
            sample_count, pretrained_model.output_shape[1] *
            pretrained_model.output_shape[2] *
            pretrained_model.output_shape[3]))
    return features, labels


def save_features(train_data, data_type):
    for model_key, model_values in MODELS.items():
        training_file = './features/{}_{}_training_features.npz'.format(
            model_key, data_type)
        validation_file = training_file.replace('training', 'validation')
        if not os.path.exists(training_file):
            print('Creating features file for the first time: {}'.format(training_file))
            x_train, y_train, train_filenames = generate_from_dir(train_data, model_values)
            x_valid, y_valid, validation_filenames = generate_from_dir(train_data.replace('training', 'validation'), model_values)
            np.savez(training_file, x_train=x_train, y_train=y_train, train_filenames=train_filenames)
            np.savez(validation_file, x_valid=x_valid, y_valid=y_valid, validation_filenames=validation_filenames)
            break
        else:
            print('Features file already exist: {}'.format(training_file))


def generate_from_dir(train_dir, model):
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(model['IMG_SIZE'], model['IMG_SIZE']),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    train_filenames = train_generator.filenames
    print(train_generator.samples)

    x_train, y_train = extract_features(train_generator, model)
    return x_train, y_train, train_filenames


def main(type):
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="directory for features extraction")
    args = parser.parse_args()
    train_data = os.path.relpath('./data/{}/training/'.format(type))
    if args.train_data:
        train_data = args.train_data

    real_count = len(os.listdir(os.path.join(train_data, 'gnr_real')))
    fake_count = len(os.listdir(os.path.join(train_data, 'gnr_fake')))
    train_sample_count = real_count + fake_count
    print('Data sample count: {}'.format(train_sample_count))

    for i in range(0, len(MODELS)):
        save_features(train_data, type)


if __name__ == "__main__":
    main('data_flipped')
    # main('data_gan')
    # main('data_photoshop')
