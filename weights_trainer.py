import os
import random
import shutil
import ssl
import time
from argparse import ArgumentParser
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3, \
    MobileNetV2, InceptionResNetV2, Xception, \
    NASNetMobile, DenseNet201, NASNetLarge
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalMaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras_preprocessing.image import load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
ssl._create_default_https_context = ssl._create_unverified_context

LR = [1e-3, 0.01, 2e-5, 2e-4, 1e-4, 5e-4, 146e-5]
SGD2 = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# CHANGE HERE
OPTIMIZER = [Adam, SGD, RMSprop]
DROPOUT = [0.3, 0.4, 0.5, 0.2][0]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][1]
LOSS = ['binary_crossentropy', 'categorical_crossentropy'][0]
METRIC = ['acc']
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
CHANNELS = 3
NUMBER_OF_CLASSES = 1
NUM_EPOCHS = 100
BATCH_SIZE = 16
# Change FULLY CONNECTED layers setup here
LAYERS_EVOLUTION = [
    [64, 'DROPOUT', 64, 'DROPOUT', 64],
    [32, 'DROPOUT', 32, 'DROPOUT', 32],
    [16, 'DROPOUT', 16, 'DROPOUT', 16],
    [10, 'DROPOUT', 10, 'DROPOUT', 10],
    [64],
    [32],
    [16],
    [10],
    # [1028, 'DROPOUT', 1028, 'DROPOUT', 100]
]


def get_random_layers():
    rnd = random.randint(2, 4096)
    layers = [rnd]
    for layer in range(random.randint(0, 5)):
        if random.randint(2, 4096) % random.randint(1, 5) == 0:
            layers.append('D')
        else:
            rnd = random.randint(2, rnd)
            layers.append(rnd)
    return layers


def create_base_model(model):
    input_shape = (model['IMG_SIZE'], model['IMG_SIZE'], 3)
    return model['TRANSFER_LEARNING'](
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)


def get_feat_count(output_shape):
    count = 1
    for i in range(1, len(output_shape)):
        count = count * output_shape[i]
    return count


def reshape_data(x_train, base_model):
    print(x_train.shape)
    x_train = np.reshape(x_train, (
        len(x_train), base_model.output_shape[1],
        base_model.output_shape[2],
        base_model.output_shape[3]))
    return x_train


def get_indices(x_train):
    a = np.array(x_train)
    indices = np.arange(a.shape[0])
    print('indices: {}'.format(indices))
    np.random.seed(1993)
    np.random.shuffle(indices)
    print('indices: {}'.format(indices))
    return indices


def shuffle_array(np_array, indices):
    return np.array(np_array)[indices]


def load_features(features_file, type='train'):
    if os.path.exists(features_file):
        bottleneck_features = np.load(features_file)
        x_train = bottleneck_features['x_{}'.format(type)]
        y_train = bottleneck_features['y_{}'.format(type)]
        if type == 'valid':
            type = 'validation'
        train_filenames = bottleneck_features['{}_filenames'.format(type)]
        print('Loaded {} features from disk'.format(features_file))
        return x_train, y_train, train_filenames
    else:
        raise Exception('Extracted Features could not be found: {}'.format(features_file))


def get_misclassified(model, x_validation, y_validation, filenames_validation):
    predictions = model.predict(x_validation)
    correct_count = 0
    misclassified = []
    for i in range(0, len(y_validation)):
        if np.round(predictions[i], 2) >= 0.5:
            if np.round(y_validation[i], 2) == 1:
                correct_count += 1
            else:
                misclassified.append(
                    (x_validation[i], y_validation[i], filenames_validation[i],
                     predictions[i])
                )
        elif np.round(predictions[i], 2) < 0.5:
            if np.round(y_validation[i], 2) == 0:
                correct_count += 1
            else:
                misclassified.append(
                    (x_validation[i], y_validation[i], filenames_validation[i],
                     predictions[i]))
    print('manual acc: {}'.format(correct_count / (len(predictions)) * 100))
    return misclassified, predictions


def show_stats(errors, validation_dir, show_images=False):
    shown_count = 0
    wrong_real, wrong_fake = 0, 0
    for pic in errors:
        pred_label = 'Real' if np.round(pic[3], 0) == 1 else 'Fake'
        orig_label = 'Real' if np.round(pic[1], 0) == 1 else 'Fake'
        if orig_label == 'Real':
            wrong_real += 1
        else:
            wrong_fake += 1
        if show_images:
            if shown_count < 10:
                print('Filename: {}, Original label:{}, '
                      'Prediction:{}, confidence : {}'.format(
                        pic[2],
                        orig_label,
                        pred_label,
                        np.round(pic[3], 2)))
                original = load_img('{}/{}'.format(validation_dir, pic[2]))
                plt.imshow(original)
                plt.show()
                shown_count += 1
    print('Fake % predicted incorrectly: {}%'.format(
        round(wrong_fake / len(errors) * 100), 2))
    print('Real % predicted incorrectly: {}%'.format(
        round(wrong_real / len(errors) * 100), 2))


def main(data_type):
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="where training data is located")
    args = parser.parse_args()
    train_dir = os.path.relpath('./data/{}/training'.format(data_type))
    if args.train_data:
        train_dir = args.train_data

    real_count = len(os.listdir(os.path.join(train_dir, 'gnr_real')))
    fake_count = len(os.listdir(os.path.join(train_dir, 'gnr_fake')))
    train_sample_count = real_count + fake_count
    print('Fake % count: {}%'.format(
        round(fake_count / train_sample_count * 100), 2))
    print('Real % count: {}%'.format(
        round(real_count / train_sample_count * 100), 2))

    for key, value in MODELS.items():
        training_features = './features/{}_{}_training_features.npz'.format(
            key, data_type)

        # model_file = './models/{}_{}_model.h5'.format(
        #     key, data_type)
        #
        # if os.path.exists(model_file):
        #     print('File already exists: {}'.format(model_file))
        #     continue

        try:
            x_train, y_train, train_filenames = load_features(training_features, 'train')
            x_valid, y_valid, valid_filenames = load_features(training_features.replace('training', 'validation'), 'valid')
        except Exception as e:
            print(e)
            continue

        print('Original data split: {}'.format(x_train.shape))

        base_model = create_base_model(value)
        dimensions = get_feat_count(base_model.output_shape)
        print('Base model output shape: {}'.format(base_model.output_shape))
        print('Input dimensions: {}'.format(dimensions))

        print('x_train shape: {}'.format(x_train.shape))
        print('y_train shape: {}'.format(y_train.shape))

        if len(x_train) != len(y_train) or len(x_train) != len(train_filenames):
            raise Exception('Loser, you made a mistake somewhere!')

        # indices = get_indices(x_train)
        # x_train = shuffle_array(x_train, indices)
        # y_train = shuffle_array(y_train, indices)
        # train_filenames = shuffle_array(train_filenames, indices)

        for _ in range(1):
            LAYERS = get_random_layers()
            layers_repr = '-'.join(str(e) for e in LAYERS)
            model_name = '{}_[{}]'.format(key, layers_repr)

            tb_call_back = TensorBoard(
                log_dir='./graphs/{}/'.format(model_name),
                histogram_freq=0, write_graph=True,
                write_images=True)

            model = Sequential()
            # ------------------- FC --------------------- #
            print('Random fc: {}'.format(LAYERS))
            if len(LAYERS):
                model.add(Dense(LAYERS[0], activation='relu', input_dim=dimensions,
                                name='fc_input'))
            for layer in LAYERS:
                if 'D' in str(layer):
                    model.add(Dropout(random.uniform(0.1, 5)))
                else:
                    model.add(Dense(int(layer), activation='relu'))
            model.add(Dense(NUMBER_OF_CLASSES, activation=DENSE_LAYER_ACTIVATION,
                            name='fc_output'))
            # ------------------- FC --------------------- #
            model.compile(optimizer=OPTIMIZER[random.randint(0, len(OPTIMIZER))](lr=LR[random.randint(0, len(LR))]),
                          loss=LOSS,
                          metrics=METRIC)
            print(model.summary())

            es = EarlyStopping(monitor='val_acc', mode='auto', verbose=0,
                               patience=10)

            filepath = '{type}_{model}_weights_{layers}'.format(
                type=data_type, model=key, layers=layers_repr)
            checkpoint = ModelCheckpoint(filepath+'.h5', monitor='val_acc',
                                         verbose=0, save_best_only=True,
                                         mode='max', save_weights_only=True)

            history = model.fit(
                x_train, y_train, epochs=NUM_EPOCHS, batch_size=random.randint(1, 100),
                validation_data=(x_valid, y_valid),
                callbacks=[tb_call_back, checkpoint, es])

            scores = model.evaluate(x_valid, y_valid, verbose=1)
            print("Accuracy: %.2f%%" % (scores[1] * 100))

            missclassified, predictions = get_misclassified(
                model, x_valid, y_valid, valid_filenames)

            show_stats(missclassified, train_dir.replace('training', 'validation'),
                       show_images=False)

            model.save_weights(os.path.relpath('./random_weights/{}_score-{}.h5'.format(
                filepath, round(checkpoint.best, 2))))
            model.save(os.path.relpath('./random_models/{}'.format(filepath)))

            c_matrix = confusion_matrix(y_true=np.round(y_valid, 0),
                                        y_pred=np.round(predictions, 0))
            print(c_matrix)


if __name__ == "__main__":
    while True:
        try:
            main('data_flipped')
            # main('data_gan')
            # main('data_photoshop')
            # time.sleep(60)
        except Exception as e:
            print(e)
            continue
