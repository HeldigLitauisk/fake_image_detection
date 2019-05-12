import os
import ssl
from argparse import ArgumentParser
from keras.applications import ResNet50, VGG19, VGG16, InceptionV3, MobileNetV2, InceptionResNetV2, Xception, NASNetLarge, MobileNet, DenseNet201, NASNetMobile
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

LR = [1e-3, 0.01, 2e-5, 2e-4, 1e-4, 5e-4, 146e-5][2]
SGD2 = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# CHANGE HERE
OPTIMIZER = [Adam, SGD, RMSprop][0]
DROPOUT = [0.3, 0.4, 0.5, 0.2][0]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][1]
LOSS = ['binary_crossentropy', 'categorical_crossentropy'][0]
METRIC = ['acc']
MODELS = {
    'DenseNet201': {'IMG_SIZE': 224, 'PROCESSING': 'torch', 'TRANSFER_LEARNING': DenseNet201},
    'MobileNetV2': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': MobileNetV2},
    'VGG19': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': VGG19},
    'NASNetMobile': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': NASNetMobile},
    'InceptionResNetV2': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionResNetV2},
    'InceptionV3': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionV3},
    'ResNet50': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': ResNet50},
    'Xception': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': Xception},
}
CHANNELS = 3
NUMBER_OF_CLASSES = 1
NUM_EPOCHS = 50
BATCH_SIZE = 16
# Change FULLY CONNECTED layers setup here
# LAYERS = [10, 'DROPOUT', 10, 'DROPOUT', 10]
LAYERS = [10]
ssl._create_default_https_context = ssl._create_unverified_context


def train_model(train_generator, val_generator, model):
    input_shape = (model['IMG_SIZE'], model['IMG_SIZE'], 3)
    model = model['TRANSFER_LEARNING'](
        weights=None, include_top=True, input_shape=input_shape, classes=1)
    # Due to horizontal flipping we have 2x samples
    train_count = train_generator.samples
    valid_count = val_generator.samples
    print('Train sample count: {}'.format(train_count))
    print('Val Sample count: {}'.format(valid_count))

    model.compile(optimizer=OPTIMIZER(lr=LR),
                  # model.compile(optimizer=SGD2,
                  loss=LOSS,
                  metrics=METRIC)
    print(model.summary())

    # model.fit_generator(train_generator,
    #                     steps_per_epoch=steps,
    #                     epochs=10,
    #                     validation_data=validation_generator,
    #                     validation_steps=num_val_samples / batch_size)

    model.fit_generator(
        train_generator, steps_per_epoch=train_count / BATCH_SIZE,
        epochs=NUM_EPOCHS, validation_data=val_generator,
        validation_steps=valid_count / BATCH_SIZE)#, callbacks=[tb_call_back])

    return model


def save_features(train_data):
    for model_key, model_values in MODELS.items():
        training_file = "./models/{}_model.h5".format(model_key)
        if not os.path.exists(training_file):
            print('Creating features file for the first time: {}'.format(training_file))
            model = generate_from_dir(train_data, model_values)
            model.save(os.path.relpath("./models/{}_model.h5".format(model_key)))
            model.save_weights(os.path.relpath('./weights/{}_weights.h5'.format(model_key)))
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

    validation_dir = train_dir.replace('training', 'validation')
    print(train_dir)
    print(validation_dir)

    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(model['IMG_SIZE'], model['IMG_SIZE']),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    train_filenames = train_generator.filenames
    print(train_generator.samples)

    new_model = train_model(train_generator, validation_generator, model)
    return new_model


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="directory for features extraction")
    args = parser.parse_args()
    train_data = os.path.relpath('./data/data_photoshop/training/')
    if args.train_data:
        train_data = args.train_data

    real_count = len(os.listdir(os.path.join(train_data, 'gnr_real')))
    fake_count = len(os.listdir(os.path.join(train_data, 'gnr_fake')))
    train_sample_count = real_count + fake_count
    print('Data sample count: {}'.format(train_sample_count))

    for i in range(0, len(MODELS)):
        save_features(train_data)


if __name__ == "__main__":
    main()
