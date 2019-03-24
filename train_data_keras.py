import os
from argparse import ArgumentParser
import cv2
from tflearn import SGD
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, \
    BatchNormalization, LeakyReLU, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing import image

IMG_SIZE = 300
LR = 1e-3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# MODEL_NAME = 'trained-model-{}-{}.model'.format(LR, '5conv-basic2')


def label_img(img):
    if img.split('/')[-1].startswith('real'):
        return [1, 0]
    else:
        # For fakes
        return [0, 1]


def create_train_data(train_dir, data_name='train'):
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(
            cv2.imread(path, cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    # shuffle(training_data)
    np.save('{}_data.npy'.format(data_name), training_data)
    return training_data


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

    # def create_model():
    #     model = Sequential()
    #     model.add(Dense(64, input_dim=14, init='uniform'))
    #     model.add(LeakyReLU(alpha=0.3))
    #     model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9,
    #                                  weights=None))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(64, init='uniform'))
    #     model.add(LeakyReLU(alpha=0.3))
    #     model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9,
    #                                  weights=None))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(2, init='uniform'))
    #     model.add(Activation('softmax'))
    #     return model
    #
    # def train(X_train, y_train):
    #     model = create_model()
    #     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #     model.compile(loss='binary_crossentropy', optimizer=sgd)
    #
    #     checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1,
    #                                    save_best_only=True)
    #     model.fit(X_train, y_train, nb_epoch=20, batch_size=16,
    #               show_accuracy=True, validation_split=0.2, verbose=2,
    #               callbacks=[checkpointer])
    #
    # def load_trained_model(weights_path):
    #     model = create_model()
    #     model.load_weights(weights_path)

    train_data = create_train_data(train_dir)
    validation_data = create_train_data(validation_dir, 'validation')
    test_data = create_train_data(test_dir, 'test')

    train_x = np.array([i[0] for i in train_data]).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test_x = np.array([i[1] for i in train_data])

    train_y = np.array([i[0] for i in validation_data]).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test_y = np.array([i[1] for i in validation_data])

    train_z = np.array([i[0] for i in test_data]).reshape(
        -1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    test_z = np.array([i[1] for i in test_data])

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
    #                  activation='relu',
    #                  input_shape=INPUT_SHAPE))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dense(2, activation='softmax'))

    # model = Sequential()
    # model.add(Conv2D(40, kernel_size=5, padding="same",
    #                  input_shape=INPUT_SHAPE, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(70, kernel_size=3, padding="same", activation='relu'))
    # model.add(Conv2D(500, kernel_size=3, padding="same", activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(1024, kernel_size=3, padding="valid", activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Flatten())
    # model.add(Dense(units=100, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(units=100, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(units=100, activation='relu'))
    # model.add(Dropout(0.3))
    #
    # model.add(Dense(2))
    # model.add(Activation("softmax"))


    # model = Sequential([
    # Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same',
    #        input_shape=INPUT_SHAPE),
    # MaxPooling2D(pool_size=(2, 2), strides=2),
    # Conv2D(16, kernel_size=(5, 5), activation='relu'),
    # MaxPooling2D(pool_size=(2, 2), strides=2),
    # Flatten(),
    # Dense(120, activation='relu'),
    # Dense(84, activation='relu'),
    # Dense(10, activation='relu'),
    # Dense(2, activation='softmax')
    # ])

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    tb_call_back = TensorBoard(log_dir='./Graph', histogram_freq=0,
                               write_graph=True, write_images=True)

    # if os.path.exists('real_vs_fake_weights.h5'):
    #     model = create_model()
    #     model.load_weights('real_vs_fake_weights.h5')
    #     print('Model Loaded!')
    # else:
    model.fit(train_x, test_x, batch_size=50, epochs=10,
              validation_data=(train_y, test_y), callbacks=[tb_call_back])

    scores = model.evaluate(train_z, test_z, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    model.save("real_vs_fake_model_{}.h5".format(IMG_SIZE))
    model.save_weights('real_vs_fake_weights_{}.h5'.format(IMG_SIZE))

    model.predict(train_z, verbose=1, batch_size=50)

    fig = plt.figure()
    for num, data in enumerate(test_data[:24]):
        img_num = data[1]
        truth = 'Real' if img_num[0] == 1 else 'Fake'
        img_data = data[0]

        y = fig.add_subplot(6, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
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


if __name__ == "__main__":
    main()
