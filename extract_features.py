import os
import ssl
from argparse import ArgumentParser
from keras.applications import ResNet50, VGG19, VGG16, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
ssl._create_default_https_context = ssl._create_unverified_context

IMG_SIZE = [224, 299, 600, 8, 96, 255, 150][0]
POOLING = ['avg', 'max', None][0]
TRANSFER_LEARNING = [ResNet50, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception][-3]
NAME = ['ResNet50', 'VGG16', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception'][-3]
# Models and respective images size and preprocessing
# ['ResNet50', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception']
# [224, 224, 224, 299, 224, 299, 299]
# [caffe, caffe, caffe,tf, tf, tf, tf]
PROCESSING = ['caffe', 'tf', 'torch'][1]
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
BATCH_SIZE = 32


def extract_features(generator):
    pretrained_model = TRANSFER_LEARNING(
        weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    sample_count = generator.samples
    print('Sample count: {}'.format(sample_count))
    features = np.zeros(shape=(sample_count, pretrained_model.output_shape[1] *
                               pretrained_model.output_shape[2] *
                               pretrained_model.output_shape[3]))
    labels = np.zeros(shape=(sample_count))

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = pretrained_model.predict(
            preprocess_input(inputs_batch, mode=PROCESSING))
        op_shape = features_batch.shape

        features_batch = np.reshape(features_batch, (
            inputs_batch.shape[0], op_shape[-3] * op_shape[-2] * op_shape[-1]))

        features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
        if i % 10 == 0:
            print("Extracting features: {} out of {}".format(
                BATCH_SIZE * i, sample_count))
        labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break

        features = np.reshape(features, (
            sample_count, pretrained_model.output_shape[1] *
            pretrained_model.output_shape[2] *
            pretrained_model.output_shape[3]))
    return features, labels


def generate_from_dir(train_dir):

    datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=False)

    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False)

    train_filenames = train_generator.filenames
    test_filenames = validation_generator.filenames


    print(validation_generator.samples)
    print(train_generator.samples)

    x_train, y_train = extract_features(train_generator)
    x_test, y_test = extract_features(validation_generator)

    return x_train, y_train, x_test, y_test, train_filenames, test_filenames


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="directory for features extraction")

    args = parser.parse_args()

    train_dir = os.path.relpath('./data/training')

    if args.train_data:
        train_dir = args.train_data

    real_count = len(os.listdir(os.path.join(train_dir, 'gnr_real')))
    fake_count = len(os.listdir(os.path.join(train_dir, 'gnr_fake')))
    train_sample_count = real_count + fake_count

    features_file = './features/{}_features_IMG-{}_Pre-{}_SAMPLE-{}.npz'.format(
        NAME, IMG_SIZE, PROCESSING, train_sample_count)
    if not os.path.exists(features_file):
        print('Creating features for first time')
        x_train, y_train, x_test, y_test, train_filenames, test_filenames =\
            generate_from_dir(train_dir)
        np.savez(features_file, x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test, train_filenames=train_filenames,
                 test_filenames=test_filenames)
    else:
        print('Features file already exist: {}'.format(features_file))


if __name__ == "__main__":
    main()
