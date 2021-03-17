import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from unet_drop import *

#for data reading
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

unet = Unet()

#scale inputs to 0-1!!
#train data

TRAIN_PATH_IN = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/denoising/train/in/'
VAL_PATH_IN = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/denoising/val/in/'
TEST_PATH_IN = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/denoising/test/in/'

TRAIN_PATH_REF = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/denoising/train/ref/'
VAL_PATH_REF = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/denoising/val/ref/'
TEST_PATH_REF = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/denoising/test/ref/'


def get_data(path):
    TRAIN_PATH = path
    train_names = next(os.walk(TRAIN_PATH))[2]
    X_train = np.zeros((len(train_names), 512, 512), dtype=np.uint16)
    print('Getting and resizing train images and masks ... ')
    for image_name, count in zip(train_names, range(len(train_names))):
        img = imread(TRAIN_PATH + image_name)
        #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[count] = img
        if count % 500 == 0:
            print('Done with # ', count)
    X_train = X_train[:,:,:,np.newaxis]
    
    return X_train

X_train = get_data(TRAIN_PATH_IN)
Y_train = get_data(TRAIN_PATH_REF)

X_val = get_data(VAL_PATH_IN)
Y_val = get_data(VAL_PATH_REF)

#scale 0 to 1
X_train = X_train / 65535.
Y_train = Y_train / 65535.
X_val = X_val / 65535.
Y_val = Y_val / 65535.

print(np.amax(X_train), np.amin(X_train), np.amax(Y_train), np.amin(Y_train))
print(np.amax(X_val), np.amin(X_val), np.amax(Y_val), np.amin(Y_val))

unet.train(30, 1, 1, X_train[0:100], Y_train[0:100], X_val[0:10], Y_val[0:10], X_val[0:10], Y_val[0:10])
