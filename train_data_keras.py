import os
from argparse import ArgumentParser
from random import shuffle

import cv2
from keras import Model, Input
from keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.optimizers import SGD
from keras_preprocessing.image import load_img, img_to_array
from matplotlib.image import imread
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
POOLING = ['avg', 'max', None][2]
DROPOUT = [0.3, 0.4, 0.5][0]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][0]
OBJECTIVE_FUNCTION = ['binary_crossentropy', 'categorical_crossentropy'][1]
LOSS_METRIC = ['accuracy']
TRANSFER_LEARNING = [ResNet50, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception][-1]
NAME = ['ResNet50', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception'][-1]
PROCESSING = ['caffe', 'tf', 'torch'][1]

INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
LR = 1e-3
NUMBER_OF_CLASSES = 2
CHANNELS = 3
NUM_EPOCHS = 5
BATCH_SIZE = 5
MODEL_NAME = 'model_{}_{}_{}_{}.model'.format(
    NAME, IMG_SIZE, DENSE_LAYER_ACTIVATION, NUM_EPOCHS)
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
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(DROPOUT)(op)
    # op = Dense(84, activation='relu')(op)
    # op = Dropout(DROPOUT)(op)
    # op = Dense(10, activation='relu')(op)
    # op = Dropout(DROPOUT)(op)

    output_tensor = Dense(NUMBER_OF_CLASSES,
                          activation=DENSE_LAYER_ACTIVATION)(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model, base_model


def label_img(img):
    if img.split('/')[-1].startswith('real'):
        return np_utils.to_categorical(1, num_classes=NUMBER_OF_CLASSES)
    else:
        return np_utils.to_categorical(0, num_classes=NUMBER_OF_CLASSES)


def create_train_data(train_dir, data_name='train', preprocess=True):
    npy_data_data = './data/{}_{}_{}_data.npy'.format(
        NAME, IMG_SIZE, data_name)
    if os.path.exists(npy_data_data) and preprocess:
        print('Loaded from disk: {}'.format(npy_data_data))
        return np.load(npy_data_data)
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        if preprocess:
            image = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image)
            image = image.reshape(
                (1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image, mode=PROCESSING)
            # image = np.array(cv2.resize(
            #     imread(path), (IMG_SIZE, IMG_SIZE)))
        else:
            image = np.array(cv2.resize(
                imread(path), (IMG_SIZE, IMG_SIZE)))
        training_data.append([image, label, path])
    shuffle(training_data)
    if preprocess:
        np.save(npy_data_data, training_data)
    return training_data


def plot_data(data, model):
    fig = plt.figure()
    for num, data in enumerate(data[:24]):
        img_num = data[1]
        truth = 'Real' if img_num[0] == 1 else 'Fake'

        y = fig.add_subplot(12, 2, num + 1)
        orig = imread(data[2])
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


def load_data(data_dir, data_type='train', preprocess=True):
    train_data = create_train_data(data_dir, data_type, preprocess)
    train = np.array([i[0] for i in train_data]).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test = np.array([i[1] for i in train_data])
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
    test = np.array(grouped_test)
    scores = model.evaluate(train, test, verbose=1)
    print("Accuracy %s: %.2f%%" % (group_name, scores[1] * 100))


def create_group(data, group_name='real'):
    grouped_train = []
    for element in data:
        path = element[2]
        if group_name in path:
            grouped_train.append(element[0])
        else:
            grouped_train.append(element[0])
    return np.array(grouped_train).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])


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


def extract_features(x_data, y_data, base_model):
    sample_count = len(x_data)

    features = np.zeros(shape=(
        sample_count, base_model.output_shape[1], base_model.output_shape[2],
        base_model.output_shape[3]))
    labels = np.zeros(shape=(sample_count, 2))

    data_generator = ImageDataGenerator(rescale=1./255)
    generator = data_generator.flow(x_data, y_data, batch_size=BATCH_SIZE)

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
        if i % 10 == 0:
            print("Extracting features: %.2d%% out of %.2d%%" % (
                i, sample_count))
        labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    return features, labels


def data_augmentation(dir, data_type='real', count=10):
    gnr_data = create_train_data(dir, 'train', False)
    data = create_group(gnr_data, data_type)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    datagen.fit(data)
    for _ in range(0, count):
        datagen.flow(
            data, batch_size=BATCH_SIZE, save_to_dir='./data/train/',
            save_prefix=data_type).next()


def main():
    tf.keras.backend.clear_session()

    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="directory for training data")
    parser.add_argument("--test_data", required=False,
                        help="directory for testing data")
    parser.add_argument("--validation_data", required=False,
                        help="directory for validation data")
    parser.add_argument("--augmentation", required=False,
                        help="whether should we do data augmentation")
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
    if args.augmentation:
        data_augmentation(train_dir, 'real')
        data_augmentation(train_dir, 'fake')

    tb_call_back = TensorBoard(log_dir='./graphs/{}/'.format(MODEL_NAME),
                               histogram_freq=0, write_graph=True,
                               write_images=True)

    x_train, y_train = load_data(train_dir, 'train')
    x_valid, y_valid = load_data(validation_dir, 'validation')
    x_test, y_test = load_data(test_dir, 'test')

    full_model, base_model = create_transfer_learning_model()
    output_shape = base_model.output_shape

    model = Sequential()
    # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    model.add(Dense(4096, activation='relu',
                    input_dim=output_shape[1] * output_shape[2] * output_shape[3]))
    # model.add(Dropout(DROPOUT))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(NUMBER_OF_CLASSES, activation=DENSE_LAYER_ACTIVATION))

    model.compile(
        loss=OBJECTIVE_FUNCTION, optimizer=OPTIMIZER, metrics=LOSS_METRIC)
    print(model.summary())

    features_file = './features/{}_features.npz'.format(NAME)
    if not os.path.exists(features_file):
        print('Creating features for first time')
        train_features, train_labels = extract_features(
            x_train, y_train, base_model)
        validation_features, validation_labels = extract_features(
            x_valid, y_valid, base_model)
        test_features, test_labels = extract_features(
            x_test, y_test, base_model)

        np.savez(features_file, train_features=train_features,
                 train_labels=train_labels,
                 validation_features=validation_features,
                 validation_labels=validation_labels,
                 test_features=test_features, test_labels=test_labels)
    else:
        bottleneck_features = np.load(features_file)
        train_features = bottleneck_features['train_features']
        train_labels = bottleneck_features['train_labels']
        validation_features = bottleneck_features['validation_features']
        validation_labels = bottleneck_features['validation_labels']
        test_features = bottleneck_features['test_features']
        test_labels = bottleneck_features['test_labels']
        print('Loaded {} features from disk'.format(features_file))

    train_features = np.reshape(train_features, (len(x_train), output_shape[1] * output_shape[2] * output_shape[3]))
    validation_features = np.reshape(validation_features, (len(x_valid), output_shape[1] * output_shape[2] * output_shape[3]))
    test_features = np.reshape(test_features, (len(x_test), output_shape[1] * output_shape[2] * output_shape[3]))

    history = model.fit(
        train_features, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(validation_features, validation_labels),
        callbacks=[tb_call_back])

    scores = model.evaluate(x_valid, y_valid, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # model.fit_generator(
    #     train_generator, steps_per_epoch=len(x_train) / BATCH_SIZE,
    #     epochs=NUM_EPOCHS, validation_data=valid_generator,
    #     validation_steps=len(x_valid) / BATCH_SIZE, callbacks=[tb_call_back])

    validation_data = create_train_data(validation_dir, 'validation')
    # evaluate_group(validation_data, model)
    # evaluate_group(validation_data, model, 'real')
    # evaluate_group(validation_data, model, 'fake')
    # evaluate_group(validation_data, model, 'easy')
    # evaluate_group(validation_data, model, 'mid')
    # evaluate_group(validation_data, model, 'hard')

    model.save("./models/{}_model.h5".format(MODEL_NAME))
    model.save_weights('./weights/{}_weights.h5'.format(MODEL_NAME))

    plot_data(validation_data, model)


if __name__ == "__main__":
    main()
