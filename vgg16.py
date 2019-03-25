import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.applications import VGG16
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten

weights_path = 'D:\CBS\Master Thesis\CODE\\fake_image_detection\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
conv_base = VGG16(include_top=False, input_shape=(150,150,3), weights=weights_path)


#Load bottleneck features

bottleneck_features = np.load('vgg16_bottleneck_features.npz')
train_features = bottleneck_features['train_features']
train_labels = bottleneck_features['train_labels']
validation_features = bottleneck_features['validation_features']
validation_labels = bottleneck_features['validation_labels']
test_features = bottleneck_features['test_features']
test_labels = bottleneck_features['test_labels']

#Reshape arrays

train_features=np.reshape(train_features, (1224, 4 * 4 * 512))
validation_features=np.reshape(validation_features, (409, 4 * 4 * 512))
test_features=np.reshape(test_features, (408, 4 * 4 * 512))

#Create classifier

model = Sequential()
model.add(Dense(256, activation='relu', input_dim= 4 * 4 * 512))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

#adam = optimizers.adam(lr=0.01)
sgd = optimizers.sgd(lr=2e-5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_features, train_labels,
                    epochs=300,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

#Graphs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()