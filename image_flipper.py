import os
import numpy as np
import matplotlib.pyplot as plt

PATH = './data/data_flipped/'


def load_metadata():
    for folder in os.listdir(PATH):
        if '.DS_Store' in folder:
            continue
        folder = os.path.join(PATH, folder)
        for sub in os.listdir(folder):
            if '.DS_Store' in sub:
                continue
            for f in os.listdir(os.path.join(folder, sub)):
                if '.DS_Store' in f:
                    continue
                # Check file extension. Allow only jpg/jpeg' files.
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    pic_path = os.path.join(folder, sub, f)
                    if 'test' not in pic_path:
                        new_name = pic_path.replace('.jpg', '_flipped.jpg')
                        flipped_im = np.fliplr(plt.imread(pic_path))
                        plt.imsave(new_name, flipped_im)
                        print(new_name)


if __name__ == "__main__":
    load_metadata()
