import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from random import randint
from tqdm import tqdm
from face_align.align import AlignDlib

IMG_SIZE = 300
from image_loader import load_metadata

metadata = load_metadata('data/training/')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]

alignment = AlignDlib('models/landmarks.dat')

for pic in tqdm(metadata):
    try:
        if not pic.image_path().endswith('.png'):
            continue

        pic_orig = load_image(pic.image_path())
        bb = alignment.getLargestFaceBoundingBox(pic_orig)

        aligned = alignment.align(IMG_SIZE, pic_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        pic_id = randint(0, 1000000)
        new_path = pic.image_path().replace('/fake/', '/gnr_fake/{}_fake_'.format(pic_id)).replace('/real/', '/gnr_real/{}_real_'.format(pic_id))
        plt.imsave(new_path, aligned)
        # if '/fake/' in pic.image_path():
        #     flipped_im = np.fliplr(plt.imread(new_path))
        #     plt.imsave(new_path.replace('.jpg', '_flipped.jpg').replace('.png', '_flipped.png'), flipped_im)
    except Exception:
        print('Failed: {}'.format(pic.image_path()))