import os
import ssl
from keras import Input, Model
from keras.applications import ResNet50, VGG19, VGG16, InceptionV3, MobileNetV2, InceptionResNetV2, Xception, NASNetLarge, MobileNet, DenseNet201, NASNetMobile
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPool2D

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

ssl._create_default_https_context = ssl._create_unverified_context
BATCH_SIZE = 32
NUM_EPOCHS = 2

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

for model_name, model_values in MODELS.items():
    model = model_name
    input_shape = (model_values['IMG_SIZE'], model_values['IMG_SIZE'], 3)
    input_tensor = Input(shape=input_shape)
    lower_layers = model_values['TRANSFER_LEARNING'](
        weights=None, include_top=False, input_tensor=input_tensor,
        pooling=None)

    top_layers = lower_layers.output
    top_layers = Flatten(input_shape=lower_layers.output_shape)(top_layers)
    top_layers = Dense(10, activation='relu')(top_layers)
    top_layers = Dense(10, activation='relu')(top_layers)
    top_layers = Dense(10, activation='relu')(top_layers)
    top_layers = Dense(1, activation='sigmoid')(top_layers)
    pretrained_model = Model(input=lower_layers.input, output=top_layers)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = datagen.flow_from_directory(
        './data/data_photoshop/training',
        target_size=(model_values['IMG_SIZE'], model_values['IMG_SIZE']),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_generator = datagen.flow_from_directory(
        './data/data_photoshop/validation',
        target_size=(model_values['IMG_SIZE'], model_values['IMG_SIZE']),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    train_count = train_generator.samples
    valid_count = val_generator.samples
    print('Train sample count: {}'.format(train_count))
    print('Val Sample count: {}'.format(valid_count))

    for layer in pretrained_model.layers:
            layer.trainable = True

    pretrained_model.compile(optimizer=Adam(), loss='binary_crossentropy',
                             metrics=['acc'])
    pretrained_model.summary()
    pretrained_model.fit_generator(
        train_generator, steps_per_epoch=train_count / BATCH_SIZE,
        epochs=NUM_EPOCHS, validation_data=val_generator,
        validation_steps=valid_count / BATCH_SIZE)

    pretrained_model.save_weights(
        './custom_weights/weights_photoshop_{}.h5'.format(model_name))
