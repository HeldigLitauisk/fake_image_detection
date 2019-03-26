import os
from argparse import ArgumentParser
from random import shuffle

import cv2
from keras import Model, Input
from keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.optimizers import SGD
from keras_preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# CHANGE HERE
OPTIMIZER = ['adam', sgd][0]
IMG_SIZE = [224, 299, 600, 96][1]
POOLING = ['avg', 'max', None][0]
DROPOUT = [0.3, 0.4, 0.5][0]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][0]
OBJECTIVE_FUNCTION = ['binary_crossentropy', 'categorical_crossentropy'][0]
LOSS_METRIC = ['accuracy']
TRANSFER_LEARNING = [ResNet50, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception][-1]

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
LR = 1e-3
NUMBER_OF_CLASSES = 2
CHANNELS = 3
NUM_EPOCHS = 3
BATCH_SIZE = 32
MODEL_NAME = 'model_{}_{}_{}.model'.format(
    IMG_SIZE, DENSE_LAYER_ACTIVATION, NUM_EPOCHS)
CONV_LAYERS = []
DENSE_LAYERS = []


def create_full_transfer_learning_model():
    # Best performance: XX %

    tf.keras.backend.clear_session()
    input_tensor = Input(shape=INPUT_SHAPE)
    base_model = TRANSFER_LEARNING(
        include_top=True,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=INPUT_SHAPE,
        pooling=POOLING)
    op = Dense(128, activation='relu')(base_model.output)
    op = Dropout(DROPOUT)(op)
    output_tensor = Dense(NUMBER_OF_CLASSES,
                          activation=DENSE_LAYER_ACTIVATION)(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def create_transfer_learning_model():
    # Best performance: XX %

    tf.keras.backend.clear_session()
    input_tensor = Input(shape=INPUT_SHAPE)
    base_model = TRANSFER_LEARNING(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=INPUT_SHAPE,
        pooling=POOLING)

    for layer in base_model.layers:
        layer.trainable = False
    op = Dense(120, activation='relu')(base_model.output)
    op = Dropout(DROPOUT)(op)
    op = Dense(84, activation='relu')(op)
    op = Dropout(DROPOUT)(op)
    op = Dense(10, activation='relu')(op)
    op = Dropout(DROPOUT)(op)

    output_tensor = Dense(NUMBER_OF_CLASSES,
                          activation=DENSE_LAYER_ACTIVATION)(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def label_img(img):
    if img.split('/')[-1].startswith('real'):
        return np_utils.to_categorical(1, num_classes=NUMBER_OF_CLASSES)
    else:
        return np_utils.to_categorical(0, num_classes=NUMBER_OF_CLASSES)


def create_train_data(train_dir, data_name='train'):
    npy_data_data = './data/{}_{}_data.npy'.format(IMG_SIZE, data_name)
    if os.path.exists(npy_data_data):
        print('Loaded from disk: {}'.format(npy_data_data))
        return np.load(npy_data_data)
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        image = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        training_data.append([image, label, path])
    shuffle(training_data)
    np.save(npy_data_data, training_data)
    return training_data


def plot_data(data, model):
    fig = plt.figure()
    for num, data in enumerate(data[:24]):
        img_num = data[1]
        print(img_num)
        truth = 'Real' if img_num[0] == 1 else 'Fake'

        y = fig.add_subplot(12, 2, num + 1)
        orig = cv2.imread(data[2])
        data = data[0].reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict(np.array([data, ]))

        if np.argmax(model_out) == 1:
            str_label = 'P:Fake vs T:{}'.format(truth)
        else:
            str_label = 'P:Real vs T:{}'.format(truth)

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data(data_dir, data_type='train'):
    train_data = create_train_data(data_dir, data_type)
    train = np.array([i[0] for i in train_data]).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test = [i[1] for i in train_data]
    return train, test


def evaluate_group(data, model, group_name='all'):
    grouped_train = []
    grouped_test = []
    for element in data:
        path = element[2]
        if group_name in path:
            grouped_train.append(element[0])
            grouped_test.append(element[1])
        elif group_name == 'fake' and 'real' not in path:
            grouped_train.append(element[0])
            grouped_test.append(element[1])
        elif group_name == 'all':
            grouped_train.append(element[0])
            grouped_test.append(element[1])
    train = np.array(grouped_train).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test = grouped_test
    scores = model.evaluate(train, test, verbose=1)
    print("Accuracy %s: %.2f%%" % (group_name, scores[1] * 100))


def data_generator(x_data, y_data, training_data=True):
    if training_data:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        datagen.fit(x_data)
        return datagen.flow(
            x_data, y_data, batch_size=BATCH_SIZE,
            save_to_dir='./augmentations/')
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow(x_data, y_data, batch_size=BATCH_SIZE)


def main():
    tf.keras.backend.clear_session()

    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="directory for training data")
    parser.add_argument("--test_data", required=False,
                        help="directory for testing data")
    parser.add_argument("--validation_data", required=False,
                        help="directory for validation data")
    args = parser.parse_args()

    train_dir = './data/training'
    test_dir = './data/test'
    validation_dir = './data/validation'

    if args.train_data:
        train_dir = args.train_data
    if args.test_data:
        test_dir = args.test_data
    if args.validation_data:
        validation_dir = args.validation_data

    tb_call_back = TensorBoard(log_dir='./graphs/{}/'.format(MODEL_NAME),
                               histogram_freq=0, write_graph=True,
                               write_images=True)

    x_train, y_train = load_data(train_dir, 'train')
    x_valid, y_valid = load_data(validation_dir, 'validation')
    # x_test, y_test = load_data(test_dir, 'test')

    model = create_transfer_learning_model()
    model.compile(
        loss=OBJECTIVE_FUNCTION, optimizer=OPTIMIZER, metrics=LOSS_METRIC)
    print(model.summary())

    train_generator = data_generator(x_train, y_train)
    valid_generator = data_generator(x_valid, y_valid, training_data=False)
    model.fit_generator(
        train_generator, steps_per_epoch=len(x_train) / BATCH_SIZE,
        epochs=NUM_EPOCHS, validation_data=valid_generator,
        validation_steps=len(x_valid) / BATCH_SIZE, callbacks=[tb_call_back])

    validation_data = create_train_data(validation_dir, 'validation')
    evaluate_group(validation_data, model)
    evaluate_group(validation_data, model, 'real')
    evaluate_group(validation_data, model, 'fake')
    evaluate_group(validation_data, model, 'easy')
    evaluate_group(validation_data, model, 'mid')
    evaluate_group(validation_data, model, 'hard')

    model.save("./models/{}_model.h5".format(MODEL_NAME))
    model.save_weights('./weights/{}_weights.h5'.format(MODEL_NAME))

    plot_data(validation_data, model)


if __name__ == "__main__":
    main()
