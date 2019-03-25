#Feature extractor made based on: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb
#Serves to extract features according to the VGG_16 architecture, which are then saved into a .npy file
#

import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

weights_path = 'D:\CBS\Master Thesis\CODE\\fake_image_detection\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
conv_base = VGG16(weights=weights_path, include_top=False, input_shape=(150,150,3))

train_dir = "D:\CBS\Master Thesis\CODE\Data_set real_and_fake_face\TRANSFER LEARNING\\training"
validation_dir = "D:\CBS\Master Thesis\CODE\Data_set real_and_fake_face\TRANSFER LEARNING\\validation"
test_dir = "D:\CBS\Master Thesis\CODE\Data_set real_and_fake_face\TRANSFER LEARNING\\test"

data_generator = ImageDataGenerator(rescale=1./255)
batch_size = 20


#Feature Extractor
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = data_generator.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i +=1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 1224)
validation_features, validation_labels = extract_features(validation_dir, 409)
test_features, test_labels = extract_features(test_dir, 408)

np.savez('vgg16_bottleneck_features.npz', train_features=train_features, train_labels=train_labels,
         validation_features=validation_features, validation_labels= validation_labels,
         test_features=test_features, test_labels=test_labels)