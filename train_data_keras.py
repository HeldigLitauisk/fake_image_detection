import os
from argparse import ArgumentParser
from random import shuffle

import cv2
from keras import Model, Input, optimizers
from keras.applications import ResNet50, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.optimizers import SGD, RMSprop, Adam
from keras_preprocessing.image import load_img, img_to_array
from matplotlib.image import imread
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, \
    BatchNormalization, LeakyReLU, Activation, K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
import ssl

LR = [1e-3, 0.01, 2e-5, 2e-4][0]
# sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# sgd2 = optimizers.sgd(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# rms = RMSprop(lr=LR)
# adam = Adam(lr=LR)
# CHANGE HERE
OPTIMIZER = [Adam, SGD, RMSprop][0]
IMG_SIZE = [224, 299, 600, 96, 255, 150][-3]
POOLING = ['avg', 'max', None][0]
DROPOUT = [0.3, 0.4, 0.5][0]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][1]
LOSS = ['binary_crossentropy', 'categorical_crossentropy'][0]
METRIC = ['acc']
TRANSFER_LEARNING = [ResNet50, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception][-2]
NAME = ['ResNet50', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception'][-2]
PROCESSING = ['caffe', 'tf', 'torch'][1]

CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
NUMBER_OF_CLASSES = 2
NUM_EPOCHS = 50
BATCH_SIZE = 32
MODEL_NAME = 'model_{}_{}_{}_{}.model'.format(
    NAME, IMG_SIZE, DENSE_LAYER_ACTIVATION, NUM_EPOCHS)
CONV = []
# Change FULLY CONNECTED layers setup here
LAYERS = [124, 'DROPOUT', 124, 10]


def create_full_transfer_learning_model():
    # Best performance: XX %

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
    # input_tensor = Input(shape=INPUT_SHAPE)
    base_model = TRANSFER_LEARNING(
        include_top=False,
        weights='imagenet',
        # input_tensor=input_tensor,
        input_shape=INPUT_SHAPE)
        # pooling=POOLING)

    # for layer in base_model.layers:
    #     layer.trainable = False
    # op = Dense(256, activation='relu')(base_model.output)
    # op = Dropout(DROPOUT)(op)
    # op = Dense(84, activation='relu')(op)
    # op = Dropout(DROPOUT)(op)
    # op = Dense(10, activation='relu')(op)
    # op = Dropout(DROPOUT)(op)

    # output_tensor = Dense(NUMBER_OF_CLASSES,
    #                       activation=DENSE_LAYER_ACTIVATION)(op)
    # model = Model(inputs=input_tensor, outputs=output_tensor)
    return base_model


def label_img(img):
    if 'real' in img:
        return to_categorical(1, num_classes=NUMBER_OF_CLASSES)
    else:
        return to_categorical(0, num_classes=NUMBER_OF_CLASSES)


def create_train_data(train_dir, data_name='train'):
    npy_data_data = './data/{}_{}_{}_data.npy'.format(
        NAME, IMG_SIZE, data_name)
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
        image = preprocess_input(image, mode=PROCESSING)
        training_data.append([image, label, path])
    shuffle(training_data)
    np.save(npy_data_data, training_data)
    return training_data


def plot_data(x, y, model):
    fig = plt.figure()
    for num, data in enumerate(x[:12]):
        truth = 'Real' if np.argmax(data[1]) == 1 else 'Fake'

        y = fig.add_subplot(4, 3, num + 1)
        orig = imread(data[2])
        data = data[0].reshape(IMG_SIZE, IMG_SIZE, CHANNELS)
        if model:
            model_out = model.predict(np.array([data, ]))

            if np.argmax(model_out) == 1:
                str_label = 'P:Fake vs T:{}'.format(truth)
            else:
                str_label = 'P:Real vs T:{}'.format(truth)

        y.imshow(orig, cmap='gray')
        if model:
            plt.title(str_label)
        else:
            plt.title(truth)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data(data_dir, data_type='train'):
    train_data = create_train_data(data_dir, data_type)
    train = np.array([i[0] for i in train_data]).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test = np.array([i[1] for i in train_data])
    return train, test


def evaluate_group(x_valid, y_valid, model, group_name='real'):
    grouped_train = []
    grouped_test = []
    for x, y in x_valid, y_valid:
        path = x[2]
        if group_name == 'real' and 'real' in path:
            grouped_train.append(x)
            grouped_test.append(y)
        elif group_name == 'fake' and 'real' not in path:
            grouped_train.append(x)
            grouped_test.append(y)
    scores = model.evaluate(grouped_train, grouped_test, verbose=1, batch_size=BATCH_SIZE)
    print("Accuracy %s: %.2f%%" % (group_name, scores[1] * 100))


def create_group(data, group_name):
    grouped_train = []
    for element in data:
        path = element[2]
        if group_name == 'real' and 'real' in path:
            grouped_train.append(element[0])
        elif group_name == 'fake' and 'real' not in path:
            grouped_train.append(element[0])
    print("There are {} {} images".format(len(grouped_train), group_name))
    return np.array(grouped_train).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])


def data_generator(x_data, y_data, training_data=True, name='fake'):
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
            save_to_dir='./augmentations/{}/'.format(name))
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow(x_data, y_data, batch_size=BATCH_SIZE)


def extract_features(train_dir):
    pretrained_model = TRANSFER_LEARNING(
        weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )
    sample_count = generator.samples
    print('Sample count: {}'.format(sample_count))
    features = np.zeros(shape=(sample_count, pretrained_model.output_shape[1] *
                               pretrained_model.output_shape[2] *
                               pretrained_model.output_shape[3]))
    labels = np.zeros(shape=(sample_count, NUMBER_OF_CLASSES))

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = pretrained_model.predict(inputs_batch)
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


# def data_augmentation(dir, data_type, count=2000):
#     gnr_data = create_train_data(dir, 'train', False)
#     data = create_group(gnr_data, data_type)
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#     datagen.fit(data)
#     datagen.flow(
#         data, batch_size=count, save_to_dir='./data/training/',
#         save_prefix='{}_gnr_'.format(data_type)).next()

def data_augmentation(fake_dir, count=1000, name='fakes'):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2
        )
    datagen.flow_from_directory(
        fake_dir, batch_size=count, save_to_dir='./augmentations/{}/'.format(name),
        save_prefix='{}_gnr_'.format(name), target_size=(600, 600),
        save_format='jpg').next()


def get_feat_count(output_shape):
    count = 1
    for i in range(1, len(output_shape)):
        count = count * output_shape[i]
    return count


def count_labels(labels):
    fakes, reals = [], []
    for label in labels:
        reals.append(label) if np.argmax(label) == 1 else fakes.append(label)
    print("{} fakes labels created for training dataset".format(len(fakes)))
    print("{} reals labels created for training dataset".format(len(reals)))


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

    train_dir = os.path.relpath('./data/training')
    test_dir = os.path.relpath('./data/test')
    validation_dir = os.path.relpath('./data/validation')

    if args.train_data:
        train_dir = args.train_data
    if args.test_data:
        test_dir = args.test_data
    if args.validation_data:
        validation_dir = args.validation_datareal
    if args.augmentation:
        data_augmentation(os.path.join(train_dir, 'reals'), name='real')
        data_augmentation(os.path.join(train_dir, 'fakes'), name='fake')
    # else:
    #     gnr_data = create_train_data(train_dir, 'train', True)
    #     create_group(gnr_data, 'fake')
    #     create_group(gnr_data, 'real')

    #
    # x_train, y_train = load_data(train_dir, 'train')
    # x_valid, y_valid = load_data(validation_dir, 'validation')
    # x_test, y_test = load_data(test_dir, 'test')
    #
    # count_labels(y_train)
    # count_labels(y_valid)

    base_model = create_transfer_learning_model()
    output_shape = base_model.output_shape

    features_file = './features/{}_features_IMG-{}_BATCH-{}.npz'.format(
        NAME, IMG_SIZE, BATCH_SIZE)
    if not os.path.exists(features_file):
        print('Creating features for first time')
        train_features, train_labels = extract_features(
            train_dir)
        validation_features, validation_labels = extract_features(
           validation_dir)
        test_features, test_labels = extract_features(
            test_dir)

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

    # train_features = np.reshape(train_features, (len(x_train), dimensions))
    # validation_features = np.reshape(validation_features, (len(x_valid), dimensions))
    # test_features = np.reshape(test_features, (len(x_test), dimensions))
    #
    # count_labels(train_labels)
    # count_labels(validation_labels)
    # print(base_model.summary())
    #
    # tf.keras.backend.clear_session()
    # tb_call_back = TensorBoard(log_dir='./graphs/{}/'.format(MODEL_NAME),
    #                            histogram_freq=0, write_graph=True,
    #                            write_images=True)
    real_count = len(os.listdir(os.path.join(train_dir, 'real')))
    fake_count = len(os.listdir(os.path.join(train_dir, 'fake')))
    train_sample_count = real_count + fake_count

    dimensions = get_feat_count(base_model.output_shape)
    print(base_model.output_shape)
    print('dimensions: {}'.format(dimensions))

    model = Sequential()
    ################# FC #######################
    if len(LAYERS):
        model.add(Dense(LAYERS[0], activation='relu', input_dim=dimensions,
                        name='fc_input'))
    for layer in LAYERS:
        if 'DROPOUT' in str(layer):
            model.add(Dropout(DROPOUT))
        else:
            model.add(Dense(int(layer), activation='relu'))
    model.add(Dense(NUMBER_OF_CLASSES, activation=DENSE_LAYER_ACTIVATION,
                    name='fc_output'))
    ################# FC #######################
    model.compile(optimizer=OPTIMIZER(lr=LR),
                  loss=LOSS,
                  metrics=METRIC)
    print(model.summary())
    model.fit(
        train_features, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(validation_features, validation_labels))

    scores = model.evaluate(validation_features, validation_labels, verbose=1,
                            batch_size=BATCH_SIZE)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    scores = model.evaluate(test_features, test_labels, verbose=1,
                            batch_size=BATCH_SIZE)
    print("Accuracy test: %.2f%%" % (scores[1] * 100))

    # plot_data(gnr_data, model)

    datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)
    sample_count = validation_generator.samples
    print('Sample count: {}'.format(sample_count))


    fnames = validation_generator.filenames
    ground_truth = validation_generator.classes
    label2index = validation_generator.class_indices
    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())
    predictions = model.predict_classes(validation_features)
    prob = model.predict(validation_features)
    errors = np.where(predictions != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), sample_count))

    print('Fake % count: {}%'.format(round(fake_count / train_sample_count * 100), 2))
    print('Real % count: {}%'.format(round(real_count / train_sample_count * 100), 2))

    wrong_fakes, wrong_real = [], []
    for i in range(0, len(errors)):
        pred_class = np.argmax(prob[errors[i]])
        pred_label = idx2label[pred_class]
        #################################
        if pred_label == 'fake':
            wrong_fakes.append(errors[i])
        elif pred_label == 'real':
            wrong_real.append(errors[i])
        #################################
        if i % 50 == 0:
            print('Filename: {}, Original label:{}, Prediction:{}, confidence : {:.3f}'.format(
                fnames[errors[i]].split('/')[-1],
                fnames[errors[i]].split('/')[0],
                pred_label,
                prob[errors[i]][pred_class]))
            original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
            plt.imshow(original)
            plt.show()
    print('Fake % predicted incorrectly: {}%'.format(round(len(wrong_fakes) / len(errors) * 100), 2))
    print('Real % predicted incorrectly: {}%'.format(round(len(wrong_real) / len(errors) * 100), 2))


    # model.fit_generator(
    #     train_generator, steps_per_epoch=len(x_train) / BATCH_SIZE,
    #     epochs=NUM_EPOCHS, validation_data=valid_generator,
    #     validation_steps=len(x_valid) / BATCH_SIZE, callbacks=[tb_call_back])

    # evaluate_group(x_valid, y_valid, model, 'real')
    # evaluate_group(x_valid, y_valid, model, 'fake')
    # print(validation_features)

    model.save("./models/{}_model.h5".format(MODEL_NAME))
    model.save_weights('./weights/{}_weights.h5'.format(MODEL_NAME))

    # gnr_data = create_train_data(validation_dir, 'validation')
    # plot_data(gnr_data, model)


if __name__ == "__main__":
    main()
