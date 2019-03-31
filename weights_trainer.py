import os
from argparse import ArgumentParser

import keras
import tensorflow as tf
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import TensorBoard
from keras_preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, confusion_matrix

LR = [1e-3, 0.01, 2e-5, 2e-4, 1e-4, 5e-4, 146e-5][2]
SGD2 = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# CHANGE HERE
OPTIMIZER = [Adam, SGD, RMSprop][0]
IMG_SIZE = [224, 299, 600, 8, 96, 255, 150][0]
DROPOUT = [0.3, 0.4, 0.5][-1]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][1]
LOSS = ['binary_crossentropy', 'categorical_crossentropy'][0]
METRIC = ['acc']
TRANSFER_LEARNING = [ResNet50, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception][-4]
NAME = ['ResNet50', 'VGG16', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception'][-4]
# Models and respective images size and preprocessing
# ['ResNet50', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception']
# [224, 224, 224, 299, 224, 299, 299]
# [caffe, caffe, caffe,tf, tf, tf, tf]
PROCESSING = ['caffe', 'tf', 'torch'][1]
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
NUMBER_OF_CLASSES = 1
NUM_EPOCHS = 32
BATCH_SIZE = 32
MODEL_NAME = 'model_{}_{}_{}_{}_L_{}_B-{}'.format(
    NAME, IMG_SIZE, DENSE_LAYER_ACTIVATION, NUM_EPOCHS, LR, LOSS, BATCH_SIZE)
# Change FULLY CONNECTED layers setup here
# LAYERS = [120, 'DROPOUT', 84]
# LAYERS = [120, 'DROPOUT', 84, 10]
LAYERS = [1200, 'DROPOUT', 840, 'DROPOUT', 100]


class MetricsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch:
            # print(self.validation_data[0])
            x_test = self.validation_data[0]
            y_test = self.validation_data[1]
            prediction = self.model.predict(x_test)
            print('r2:', r2_score(prediction, y_test).round(2))


def create_base_model():
    # input_tensor = Inputshape=INPUT_SHAPE)
    return TRANSFER_LEARNING(
        include_top=False,
        weights='imagenet',
        # input_tensor=input_tensor,
        input_shape=INPUT_SHAPE)


def get_feat_count(output_shape):
    count = 1
    for i in range(1, len(output_shape)):
        count = count * output_shape[i]
    return count


def evaluate_validation(model, x_train, y_train, train_dir):
    scores = model.evaluate(x_train, y_train, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    fnames = validation_generator.filenames
    # ground_truth = validation_generator.classes
    # label2index = validation_generator.class_indices
    validation_features = [x_train[i] for i in label2index]
    validation_labels = [y_train[i] for i in label2index]
    print('Validation features count: {}'.format(len(validation_features)))
    print('Validation labels count: {}'.format(len(validation_labels)))
    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())
    predictions = model.predict_classes(validation_features)
    prob = model.predict(validation_features)
    errors = np.where(predictions != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

    wrong_fakes, wrong_real = [], []
    for i in range(0, len(errors)):
        pred_class = np.argmax(prob[errors[i]])
        pred_label = idx2label[pred_class]
        #################################
        if pred_label == 'fake':
            wrong_fakes.append(errors[i])
        elif pred_label == 'real':
            wrong_real.append(errors[i])
        ################################
        if i % 50 == 0:
            print('Filename: {}, Original label:{}, Prediction:{}, confidence : {:.3f}'.format(
                fnames[errors[i]].split('/')[-1],
                fnames[errors[i]].split('/')[0],
                pred_label,
                prob[errors[i]][pred_class]))
            original = load_img('{}/{}'.format(train_dir, fnames[errors[i]]))
            plt.imshow(original)
            plt.show()
    print('Fake % predicted incorrectly: {}%'.format(
        round(len(wrong_fakes) / len(errors) * 100), 2))
    print('Real % predicted incorrectly: {}%'.format(
        round(len(wrong_real) / len(errors) * 100), 2))

    c_matrix = confusion_matrix(validation_labels.argmax(axis=1), predictions)
    print(c_matrix)

    model.save("./models/{}_model_score-{}.h5".format(
        MODEL_NAME, int(scores[1] * 100)))
    model.save_weights('./weights/{}_weights_score-{}.h5'.format(
        MODEL_NAME, int(scores[1] * 100)))


def reshape_data(x_train, base_model):
    print(x_train.shape)
    x_train = np.reshape(x_train, (
        len(x_train), base_model.output_shape[1],
        base_model.output_shape[2],
        base_model.output_shape[3]))
    return x_train


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=False,
                        help="where training data is located")
    args = parser.parse_args()
    train_dir = os.path.relpath('./data/training')
    if args.train_data:
        train_dir = args.train_data

    real_count = len(os.listdir(os.path.join(train_dir, 'gnr_real')))
    fake_count = len(os.listdir(os.path.join(train_dir, 'gnr_fake')))
    train_sample_count = real_count + fake_count
    print('Fake % count: {}%'.format(
        round(fake_count / train_sample_count * 100), 2))
    print('Real % count: {}%'.format(
        round(real_count / train_sample_count * 100), 2))

    features_file = './features/{}_features_IMG-{}_Pre-{}_SAMPLE-{}.npz'.format(
        NAME, IMG_SIZE, PROCESSING, train_sample_count)
    if os.path.exists(features_file):
        bottleneck_features = np.load(features_file)
        x_train = bottleneck_features['x_train']
        x_test = bottleneck_features['x_test']
        y_train = bottleneck_features['y_train']
        y_test = bottleneck_features['y_test']
        train_filenames = bottleneck_features['train_filenames']
        test_filenames = bottleneck_features['test_filenames']
        print('Loaded {} features from disk'.format(features_file))
    else:
        raise Exception('Extracted Features could not be found: {}'.format(
            features_file))

    tb_call_back = TensorBoard(log_dir='./graphs/{}/'.format(MODEL_NAME),
                               histogram_freq=0, write_graph=True,
                               write_images=True)

    print('Original data split: {}'.format(x_train.shape))

    base_model = create_base_model()
    dimensions = get_feat_count(base_model.output_shape)
    print('Base model output shape: {}'.format(base_model.output_shape))
    print('Input dimensions: {}'.format(dimensions))

    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))

    model = Sequential()
    # ------------------- FC --------------------- #
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
    # ------------------- FC --------------------- #
    model.compile(optimizer=OPTIMIZER(lr=LR),
                  loss=LOSS,
                  metrics=METRIC)
    print(model.summary())

    history = model.fit(
        x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tb_call_back, MetricsCallback()], validation_split=0.25, shuffle=False)

    x_validation = history.validation_data[0]
    y_validation = history.validation_data[1]
    # train_count = train_sample_count - len(y_validation)
    filenames_validation = train_filenames[-(len(y_validation)):]

    scores = model.evaluate(x_validation, y_validation, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
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
                     predictions[i])
                )

    print('manual acc: {}'.format(correct_count/(len(predictions)-1)*100))

    shown_count = 0
    wrong_real, wrong_fake = 0, 0
    for pic in misclassified:
        pred_label = 'Real' if np.round(pic[3], 0) == 1 else 'Fake'
        orig_label = 'Real' if np.round(pic[1], 0) == 1 else 'Fake'
        if orig_label == 'Real':
            wrong_real += 1
        else:
            wrong_fake += 1
        if shown_count < 10:
            print('Filename: {}, Original label:{}, '
                  'Prediction:{}, confidence : {}'.format(
                    pic[2],
                    orig_label,
                    pred_label,
                    np.round(pic[3], 2)))
            original = load_img('{}/{}'.format(train_dir, pic[2]))
            plt.imshow(original)
            plt.show()
            shown_count += 1
    print('Fake % predicted incorrectly: {}%'.format(
        round(wrong_fake / len(misclassified) * 100), 2))
    print('Real % predicted incorrectly: {}%'.format(
        round(wrong_real / len(misclassified) * 100), 2))

    c_matrix = confusion_matrix(np.round(y_validation, 0), np.round(predictions, 0))
    print(c_matrix)
    model.save("./models/{}_model_score-{}.h5".format(
        MODEL_NAME, int(scores[1] * 100)))
    model.save_weights('./weights/{}_weights_score-{}.h5'.format(
        MODEL_NAME, int(scores[1] * 100)))


if __name__ == "__main__":
    main()


