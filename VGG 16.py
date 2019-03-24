import numpy as np
import matplotlib.pyplot as plt
#from os.path import isfile, isdir, getsize
from os import mkdir, makedirs, remove, listdir

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, MaxPool2D, Conv2D, Flatten, BatchNormalization, Dropout

import shutil

