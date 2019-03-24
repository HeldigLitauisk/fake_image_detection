import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
from os.path import isfile, isdir, getsize
from os import mkdir, makedirs, remove, listdir

from keras.models import Sequential
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#Global constants

NUM_CLASSES = 2
CHANNELS = 3
IMAGE_SIZE = 224
RESNET_POOLING = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']

# Early_stop_patience must be less than num_epochs
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

resnet_weights_path = 'D:\CBS\Master Thesis\CODE\\fake_image_detection\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling=RESNET_POOLING, weights=resnet_weights_path))
model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))
#First layer of ResNet50 not trainable, taking the original weights
model.layers[0].trainable = False
model.summary()

#Compile model
sgd = optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)

#Process image, preprocess_input means batch normalization
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory("D:\CBS\Master Thesis\CODE\Data_set real_and_fake_face\TRANSFER LEARNING\\training",
                                                     target_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=BATCH_SIZE_TRAINING, class_mode='categorical')
validation_generator = data_generator.flow_from_directory("D:\CBS\Master Thesis\CODE\Data_set real_and_fake_face\TRANSFER LEARNING\\validation",
                                                    target_size=(IMAGE_SIZE,IMAGE_SIZE), batch_size=BATCH_SIZE_TRAINING, class_mode='categorical')

#Train model
cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath= "D:\CBS\Master Thesis\CODE\\fake_image_detection\\bestmodel.hdf5", monitor='val_loss', save_best_only=True, mode='auto')

#Fitting model
fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=STEPS_PER_EPOCH_VALIDATION,
    callbacks=[cb_checkpointer, cb_early_stopper]
)
model.load_weights("D:\CBS\Master Thesis\CODE\\fake_image_detection\\bestmodel.hdf5")

#Show training charts
plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(fit_history.history['acc'])
plt.plot(fit_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.show()

