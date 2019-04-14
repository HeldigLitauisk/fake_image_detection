import os
import ssl
from argparse import ArgumentParser
from keras.applications import ResNet50, VGG19, VGG16, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
# Models and respective images size and preprocessing
# ['ResNet50', 'VGG19', 'InceptionV3', 'MobileNetV2', 'InceptionResNetV2', 'Xception']
# [224, 224, 224, 299, 224, 299, 299]
# [caffe, caffe, caffe,tf, tf, tf, tf]

MODELS = {
    'InceptionResNetV2': {'IMG_SIZE': 299, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': InceptionResNetV2},
    'MobileNetV2': {'IMG_SIZE': 224, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': MobileNetV2},
    'VGG19': {'IMG_SIZE': 224, 'PROCESSING': 'caffe', 'TRANSFER_LEARNING': VGG19},
    'Xception': {'IMG_SIZE': 299, 'PROCESSING': 'tf', 'TRANSFER_LEARNING': Xception},
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
            pretrained_model = load_model(os.path.join(model_dir, model))
            weights_path = model.replace('models', 'weights').replace('_model_', '_weights_')
            pretrained_model.load_weights(os.path.join(weights_dir, weights_path))
            print('Loaded model: {}'.format(model_name))
            return pretrained_model


def eval_model(train_dir, model_name, evaluation=False):
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

    pretrained_model = get_pretrained_model(model_name)
    predictions = []
    truth = []
    score = 0
    for inputs_batch, labels_batch in tqdm(generator):
        print(inputs_batch[0].shape)
        preprocessing = reshape_data(preprocess_input(inputs_batch, mode=model['PROCESSING']), orig_model)
        prediction = pretrained_model.predict(preprocessing, verbose=0)
        if np.round(prediction[0], 2) >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        scores = pretrained_model.evaluate(preprocessing, labels_batch, verbose=0)
        score += scores[1] * 100
        truth += labels_batch[0]
    if evaluation:
        return truth
    print('Total score for: {} - {}%'.format(model_name, round(score / generator.samples, 2)))
    return predictions


def count_votes(votes):
    final_votes = []
    votes_count = len(votes[0])
    for vote in votes:
        if len(vote) != votes_count:
            raise Exception('Got unequal votes count: {} != {}'.format(votes_count, len(vote)))
        decision = 0
        for participant in range(0, len(votes)):
            decision += vote[participant]
        if float(decision) / len(votes) >= 0.5:
            final_votes.append(1)
        else:
            final_votes.append(0)


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
        votes = []
        for key, _ in MODELS.items():
            votes.append(eval_model(eval_dir, key))
        final_votes = count_votes(votes)
        print(final_votes)
    elif args.predict:
        predict_dir = os.path.relpath(args.predict)
    elif args.pic:
        predict_pic = os.path.relpath(args.pic)
    else:
        raise Exception('At least one mode needs to be selected. --help for more information')


if __name__ == "__main__":
    main()
