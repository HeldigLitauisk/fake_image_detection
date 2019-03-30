import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from face_align.align import AlignDlib

# %matplotlib inline

IMG_SIZE = 225
from image_loader import load_metadata

metadata = load_metadata('data/training/')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

for pic in metadata:
    # if '/real/' in pic.image_path():
    #     continue
    # Load an image of Jacques Chirac
    pic_orig = load_image(pic.image_path())
    # jc_orig = load_image('data/training/fake/easy_87_0110.jpg')
    # Detect face and return bounding box
    bb = alignment.getLargestFaceBoundingBox(pic_orig)



    # Transform image using specified face landmark indices and crop image to 96x96 INNER_EYES_AND_BOTTOM_LIP # OUTER_EYES_AND_NOSE

    try:
        jc_aligned = alignment.align(IMG_SIZE, pic_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        new_path = pic.image_path().replace('/fake/', '/gnr_fake/').replace('/real/', '/gnr_real/')
        plt.imsave(new_path, jc_aligned)
        if '/fake/' in pic.image_path():
            flipped_im = np.fliplr(plt.imread(new_path))
            plt.imsave(new_path.replace('.jpg', '_flipped.jpg').replace('.png', '_flipped.png'), flipped_im)
    except Exception:
        plt.imsave(pic.image_path().replace('/fake/', '/gnr_failed_fake/').replace('/real/', '/gnr_failed_real/'), pic_orig)
        # try:
        #     jc_aligned = alignment.align(IMG_SIZE, pic_orig, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        #     plt.imsave(pic.image_path().replace('/fake/', '/gnr_fake/BOTTOM_LIP_').replace('/real/', '/gnr_real/BOTTOM_LIP_'), jc_aligned)
        # except AttributeError:
        #     pass
        # plt.subplot(132)
        # plt.imshow(pic_orig)
        # plt.show()
# Show original image
# plt.subplot(131)
# plt.imshow(jc_orig)
#
# # Show original image with bounding box
# plt.subplot(132)
# plt.imshow(jc_orig)
# plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
#
# # Show aligned image
# plt.subplot(133)
# plt.imshow(jc_aligned)
#
# plt.show()
