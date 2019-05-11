import os
import ssl
from argparse import ArgumentParser
from keras.applications import ResNet50, VGG19, VGG16, InceptionV3, \
    MobileNetV2, InceptionResNetV2, Xception, DenseNet201, MobileNet, \
    NASNetMobile, NASNetLarge
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model

ssl._create_default_https_context = ssl._create_unverified_context

MODELS = {
    'InceptionResNetV2': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionResNetV2},
    'InceptionV3': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionV3},
    'DenseNet201': {'IMG_SIZE': 224, 'PROCESSING': 'torch', 'TRANSFER_LEARNING': DenseNet201},
    # 'MobileNetV2': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': MobileNetV2},
    # 'ResNet50': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': ResNet50},
    # 'VGG19': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': VGG19},
    # 'Xception': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': Xception},
    # 'NASNetLarge': {'IMG_SIZE': 244, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': NASNetLarge},
    # 'NASNetMobile': {'IMG_SIZE': 244, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': NASNetMobile},
}


def reshape_data(data, pretrained_model):
    return np.reshape(data, (
            len(data), pretrained_model.output_shape[1] *
            pretrained_model.output_shape[2] *
            pretrained_model.output_shape[3]))


def get_feat_count(output_shape):
    count = 1
    for i in range(1, len(output_shape)):
        count = count * output_shape[i]
    return count


def get_pretrained_model(model_name):
    model_dir = os.path.join(os.path.curdir, 'models')
    weights_dir = os.path.join(os.path.curdir, 'weights')
    for model in os.listdir(model_dir):
        if model_name.lower() in model.lower():
            print(os.path.join(model_dir, model))
            pretrained_model = load_model(os.path.join(model_dir, model))
            weights_path = model.replace('models', 'weights').replace('_model_', '_weights_')
            pretrained_model.load_weights(os.path.join(weights_dir, weights_path))
            print('Loaded model: {}'.format(model_name))
            return pretrained_model


def eval_model(train_dir, model_name, evaluation=False):
    print(model_name)
    model = MODELS[model_name]
    img_size = model['IMG_SIZE']
    orig_model = model['TRANSFER_LEARNING'](
        weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(model['IMG_SIZE'], model['IMG_SIZE']),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

    if evaluation:
        return generator.labels[0:len(generator.labels)+1]

    pretrained_model = get_pretrained_model(model_name)
    predictions = []
    score = 0
    for inputs_batch, labels_batch in generator:
        features_batch = orig_model.predict(
            preprocess_input(inputs_batch, mode=model['PROCESSING']))
        op_shape = features_batch.shape
        features_batch = np.reshape(features_batch, (
            inputs_batch.shape[0], op_shape[-3] * op_shape[-2] * op_shape[-1]))
        prediction = pretrained_model.predict(features_batch, verbose=0)
        if np.round(prediction[0], 2) >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        scores = pretrained_model.evaluate(features_batch, labels_batch, verbose=0)
        score += scores[1] * 100
        if len(predictions) >= len(generator.labels):
            break
    print('Total score for: {} - {}%'.format(model_name, round(score / len(predictions), 2)))
    return predictions


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--eval", required=False,
                        help="folder containing two directories for evaluation")
    parser.add_argument("--pic", required=False,
                        help="picture for prediction")
    parser.add_argument("--predict", required=False,
                        help="directory for predictions")

    args = parser.parse_args()

    if args.eval:
        eval_dir = os.path.relpath(args.eval)

        any_model = list(MODELS.keys())[0]
        truth = eval_model(eval_dir, any_model, evaluation=True)
        print('Truth: {}'.format(truth))

        votes = []

        combined_votes = []
        for key, _ in MODELS.items():
            single_model_vote = eval_model(eval_dir, key)
            print('{}: predictions {}'.format(key, single_model_vote))
            votes.append(single_model_vote)
            item = 0
            for element in single_model_vote:
                try:
                    combined_votes[item] = (combined_votes[item] + element) / 2
                except IndexError:
                    combined_votes.append(element)
                item += 1
            print('{}: combined {}'.format(key, combined_votes))

        # average_votes = np.average(np.array(votes), axis=0)
        # print(average_votes)
        final_predictions = np.where(np.array(combined_votes) > 0.5, 1, 0)
        print('final_predictions: {}'.format(final_predictions))

        if len(truth) != len(final_predictions):
            raise Exception('Predictions {} != labels {}'.format(len(final_predictions), len(truth)))

        correct_count = 0
        for i in range(0, len(truth)):
            if truth[i] == final_predictions[i]:
                correct_count += 1
        print('Correct predictions after voting from all models: {}%'.format(correct_count / len(truth) * 100))

    elif args.predict:
        predict_dir = os.path.relpath(args.predict)
    elif args.pic:
        predict_pic = os.path.relpath(args.pic)
    else:
        raise Exception('At least one mode needs to be selected. --help for more information')


if __name__ == "__main__":
    main()
