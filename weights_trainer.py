import os
from argparse import ArgumentParser
import tensorflow as tf
from keras.applications import ResNet50, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard

LR = [1e-3, 0.01, 2e-5, 2e-4, 1e-4, 5e-4, 146e-5][2]
SGD2 = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# CHANGE HERE
OPTIMIZER = [Adam, SGD, RMSprop][0]
IMG_SIZE = [224, 299, 600, 8, 96, 255, 150][0]
DROPOUT = [0.3, 0.4, 0.5][-1]
DENSE_LAYER_ACTIVATION = ['softmax', 'sigmoid'][1]
LOSS = ['binary_crossentropy', 'categorical_crossentropy'][0]
METRIC = ['acc']
TRANSFER_LEARNING = [ResNet50, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception][1]
NAME = ['ResNet50', 'VGG16', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception'][1]
# Models and respective images size and preprocessing
# ['ResNet50', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception']
# [224, 224, 224, 299, 224, 299, 299]
# [caffe, caffe, caffe,tf, tf, tf, tf]
PROCESSING = ['caffe', 'tf', 'torch'][0]
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
NUMBER_OF_CLASSES = 1
NUM_EPOCHS = 32
BATCH_SIZE = 32
MODEL_NAME = 'model_{}_{}_{}_{}_L_{}_B-{}'.format(
    NAME, IMG_SIZE, DENSE_LAYER_ACTIVATION, NUM_EPOCHS, LR, LOSS, BATCH_SIZE)
# Change FULLY CONNECTED layers setup here
# LAYERS = [120, 'DROPOUT', 84]
LAYERS = [512, 'DROPOUT', 512, 'DROPOUT', 16]


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

    features_file = './features/{}_features_IMG-{}_Pre-{}_SAMPLE-{}.npz'.format(
        NAME, IMG_SIZE, PROCESSING, train_sample_count)
    if os.path.exists(features_file):
        bottleneck_features = np.load(features_file)
        train_features = bottleneck_features['train_features']
        train_labels = bottleneck_features['train_labels']
        print('Loaded {} features from disk'.format(features_file))
    else:
        raise Exception('Extracted Features could not be found: {}'.format(
            features_file))

    tb_call_back = TensorBoard(log_dir='./graphs/{}/'.format(MODEL_NAME),
                               histogram_freq=0, write_graph=True,
                               write_images=True)

    base_model = create_base_model()
    dimensions = get_feat_count(base_model.output_shape)
    print('Base model output shape: {}'.format(base_model.output_shape))
    print('Input dimensions: {}'.format(dimensions))

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

    model.fit(
        train_features, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[tb_call_back], validation_split=0.25)

    # model.evaluate_generator
    #
    # scores = model.evaluate(train_features, validation_labels, verbose=1,
    #                         batch_size=BATCH_SIZE)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))
    # scores_test = model.evaluate(test_features, test_labels, verbose=1,
    #                              batch_size=BATCH_SIZE)
    # print("Accuracy test: %.2f%%" % (scores_test[1] * 100))

    # model.save("./models/{}_model_score-{}.h5".format(MODEL_NAME, int(
    #     scores[1] * 100)))
    # model.save_weights(
    #     './weights/{}_weights_score-{}.h5'.format(MODEL_NAME,
    #                                               int(scores[1] * 100)))

    # gnr_data = create_train_data(validation_dir, 'validation')
    # plot_data(gnr_data, model)


if __name__ == "__main__":
    main()
