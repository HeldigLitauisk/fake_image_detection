import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from random import randint
from tqdm import tqdm, tqdm_gui
from face_align.align import AlignDlib

IMG_SIZE = 225
SHOW_IMAGES = True
from image_loader import load_metadata

target_dir = 'data/data_photoshop/'
metadata = load_metadata('data/data_raw/sample/')
np.random.shuffle(metadata)


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


images_count = 0
for pic in tqdm(metadata):
    try:
        if not pic.image_path().endswith('.png') and not pic.image_path().endswith('.jpg'):
            continue
        images_count += 1

        pic_orig = load_image(pic.image_path())
        bb = alignment.getLargestFaceBoundingBox(pic_orig)

        aligned = alignment.align(IMG_SIZE, pic_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        pic_id = randint(0, 1000000)
        pic_name = pic.image_path().split('/')[-1]

        if images_count % 5 == 0:
            new_path = os.path.join(target_dir, 'test', )
        elif images_count % 4 == 0:
            new_path = os.path.join(target_dir, 'validation')
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
            if '/training/' in new_path and 'gnr_fake' in new_path:
                aligned2 = alignment.align(IMG_SIZE, pic_orig, bb,
                                          landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                flipped_im = np.fliplr(aligned2)
                plt.imsave(new_path.replace('.jpg', '_flipped.jpg').replace('.png', '_flipped.png'), flipped_im)
    except Exception:
        print('Failed: {}'.format(pic.image_path()))
