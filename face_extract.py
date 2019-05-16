import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from random import randint
from face_align.align import AlignDlib

IMG_SIZE = 300
SHOW_IMAGES = False
FLIP_IMAGES = False
VALIDATION_PCT = 10
TEST_PCT = 10

from image_loader import load_metadata

# target_dir = 'data/data_photoshop/'
target_dir = 'data/data_gan/'
metadata = load_metadata('data/data_raw/data_raw_gan/')
# metadata = load_metadata('data/data_raw/data_raw_photoshop/')
np.random.shuffle(metadata)


def create_dirs():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(os.path.join(target_dir, 'test')):
        os.makedirs(os.path.join(target_dir, 'test'))
        os.makedirs(os.path.join(target_dir, 'test', 'gnr_fake'))
        os.makedirs(os.path.join(target_dir, 'test', 'gnr_real'))
        os.makedirs(os.path.join(target_dir, 'training'))
        os.makedirs(os.path.join(target_dir, 'training', 'gnr_fake'))
        os.makedirs(os.path.join(target_dir, 'training', 'gnr_real'))
        os.makedirs(os.path.join(target_dir, 'validation'))
        os.makedirs(os.path.join(target_dir, 'validation', 'gnr_fake'))
        os.makedirs(os.path.join(target_dir, 'validation', 'gnr_real'))


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]

alignment = AlignDlib('models/landmarks.dat')


def show_bounding_box(pic_orig, aligned, bb):
    # Show original image
    plt.subplot(131)
    plt.imshow(pic_orig)
    # Show original image with bounding box
    plt.subplot(132)
    plt.imshow(pic_orig)
    plt.gca().add_patch(
        patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(),
                          fill=False, color='red'))
    # Show aligned image
    plt.subplot(133)
    plt.imshow(aligned)
    plt.show()


print("Total amount of data found: {}".format(len(metadata)))
val_counter = int(len(metadata) * VALIDATION_PCT / 100)
test_counter = int(len(metadata) * TEST_PCT / 100)
create_dirs()
for pic in metadata:
    try:
        if not pic.image_path().endswith('.png') and not pic.image_path().endswith('.jpg'):
            continue

        pic_orig = load_image(pic.image_path())
        bb = alignment.getLargestFaceBoundingBox(pic_orig)

        aligned = alignment.align(IMG_SIZE, pic_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        pic_id = randint(0, 1000000)
        pic_name = pic.image_path().split('/')[-1]

        if test_counter != 0:
            new_path = os.path.join(target_dir, 'test', )
            test_counter -= 1
        elif val_counter != 0:
            new_path = os.path.join(target_dir, 'validation')
            val_counter -= 1
        else:
            new_path = os.path.join(target_dir, 'training')

        if 'real' in pic_name:
            new_path = os.path.join(new_path, 'gnr_real', '{}_real_{}'.format(pic_id, pic_name))
        else:
            new_path = os.path.join(new_path, 'gnr_fake', '{}_fake_{}'.format(pic_id, pic_name))

        if SHOW_IMAGES:
            show_bounding_box(pic_orig, aligned, bb)
        else:
            plt.imsave(new_path, aligned)
            if '/training/' in new_path and 'gnr_fake' in new_path and FLIP_IMAGES:
                aligned2 = alignment.align(IMG_SIZE, pic_orig, bb,
                                          landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                flipped_im = np.fliplr(aligned2)
                plt.imsave(new_path.replace('.jpg', '_flipped.jpg').replace('.png', '_flipped.png'), flipped_im)
    except Exception as e:
        print('Failed: {} with error: {}'.format(pic.image_path(), e))
