from argparse import ArgumentParser
import os
from datetime import datetime
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from pathlib import Path

IMG_SIZE = 125
LR = 1e-3

MODEL_NAME = 'trained-model-{}-{}-{}.model'.format(LR, datetime.now(),
                                                   '5conv-basic2')

def label_img(img):
    # word_label = img.split('.')[0]
    if img.split('/')[-1].startswith('real'):
        return [1, 0]
    else:
        # For fakes
        return [0, 1]


def create_train_data(train_dir):
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir,img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data(test_dir):
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir,img)
        # img_num = img.split('.')[0]
        img_num = len(testing_data)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--train_data", required=True,
                        help="train")
    parser.add_argument("--test_data", required=True,
                        help="test")
    parser.add_argument("--validation_data", required=True,
                        help="test")

    args = parser.parse_args()
    train_dir = args.train_data
    test_dir = args.test_data
    validation_dir = args.test_data
    train_data = create_train_data(train_dir)
    validation_data = create_train_data(validation_dir)
    print(train_dir)

    # Define size by IMG_SIZE
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    # Fully connected has 2 outputs
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='binary_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded!')

    X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train_data]

    test_x = np.array([i[0] for i in validation_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in validation_data]

    model.fit({'input': X}, {'targets': Y}, n_epoch=5,
              validation_set=({'input': test_x},
                              {'targets': test_y}), snapshot_step=500,
              show_metric=True, run_id=MODEL_NAME)

    test_data = process_test_data(test_dir)

    fig = plt.figure()

    for num, data in enumerate(test_data[:12]):
        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            str_label = 'Fake'
        else:
            str_label = 'Real'

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    main()
