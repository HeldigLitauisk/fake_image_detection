def create_custom_model():
    # Best performance: XX %

    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',
                     input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(epsilon=LR))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def create_vgg_model():
    # Best performance: XX %

    cnn1 = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
               input_shape=INPUT_SHAPE),
        Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2), Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return cnn1


def create_alexnet_model():
    # Best performance: XX %

    model = Sequential()
    model.add(Conv2D(
        filters=96, input_shape=INPUT_SHAPE, kernel_size=(11, 11),
        strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(
        filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(
        filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(
        filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Conv2D(
        filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization(epsilon=LR))
    model.add(Flatten())
    model.add(Dense(4096, input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model