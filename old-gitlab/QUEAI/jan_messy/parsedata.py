import os
from scipy.io import loadmat
import numpy as np
import tensorflow as tf


def mat2numpy(directory, key, axis=0, dtype=np.float32, verbose=False):
    ''' Reads a collection of .mat files and combines them into a numpy array.

    Reads the data stored under a given dictionary key from .mat files
    in a given directory and stcks the data into one numpy array. For this
    the data in all .mat files needs to have the same shape. The data is
    stacked along a new axis.

    Args:
        directory (str): The directory path.
        key (str): The dictionary key for the data within files.
        axis (int): The new axis along which to stack data (Default 0).
        dtype (dtype): Convert all data to this data type. (Default np.float32)
        verbose (bool): Print progress information or not. (Default False).

    Returns:
        A numpy array of stacked data from all .mat files in directory.

    '''
    arrays = []
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            if verbose:
                print('Processing {}.'.format(filename))
            filepath = os.path.join(directory, filename)
            data = loadmat(filepath)[key].astype(dtype)
            arrays.append(data)
    return np.stack(arrays)


def loadfiles(file1, file2, func=None):
    '''Loads two data arrays from .mat files.

    To be used in combination with tf.data.Dataset.map to load .mat file data
    into Datasets consisting of two sets of tensors (one for inputs and one
    for targets).

    In addition an optional function can be provided to transform the
    first file data after reading it.

    Args:
        file1 (str): Filepath for input files.
        file2 (str): Filepath for output files.
        func (handle): A function transforming data of file1.

    Returns:
        Tensorflow tensors containing the .mat file data.

    '''
    d1 = loadmat(file1)['f'].astype(np.float32)
    d2 = loadmat(file2)['f'].astype(np.float32)
    # transform data of file1 if necessary
    if func is not None:
        d1 = func(d1)
    # add single channel dimension to image arrays
    d1 = np.expand_dims(d1, axis=2)
    d2 = np.expand_dims(d2, axis=2)
    return d1, d2


def immask(images, position=None, size=(80, 80), value=0.0):
    ''' Masks a rectangular region in an stack of images with a fixed value.

    Sets all pixels within a specified rectangular region of all images
    to a fixed value, thus masking that region of the images. The images
    are 3D array of images stacked along the first axis, that is the shape
    is [N, W, H], where N is the number of images, and W and H are width and
    height of each image in the stack respectively.

    Args:
        images (np.array): The image array.
        position (int, int): The position of the lower left corner of the
            masked region. Centers the masked region if position is None.
            (The default is None).
        size (int, int): The size of the rectangular mask region.
            (The default is (60, 60)).
        value (float): The fixed value for pixels in the masked region.
            (The default is 0).

    Returns:
        np.array: The masked image array.

    '''
    singleimage = len(images.shape) == 2
    if singleimage:
        images = np.expand_dims(images, axis=0)
    imshape = np.asarray(images.shape)[1:3]
    size = np.asarray(size)
    if position is None:
        position = np.maximum(
            0,
            np.floor_divide(imshape, 2) - np.floor_divide(size, 2)
        )
    position = np.asarray(position)
    position2 = np.minimum(imshape-1, position + size)
    masked = images.copy()
    masked[:, position[0]:position2[0], position[1]:position2[1]] = value
    if singleimage:
        masked = masked[0, :, :]
    return masked
    

def imnoise(images, position=None, size=(1, 1), value=1.0, seed=None):
    ''' Adds noise to a stack of images with a smoothly varying strength.

    Adds Gaussian white noise multiplied by a smooth amplitude to images 
    so that the noise variance decreases smoothly around a center. The images
    are 3D array of images stacked along the first axis, that is the shape
    is [N, W, H], where N is the number of images, and W and H are width and
    height of each image in the stack respectively.

    Args:
        images (np.array): The image array.
        position (int, int): The position of the highest noise amplitude in
            the multiplicator. Centers the noise if position is None.
            (The default is None).
        size (int, int): The stretch of the ellipsoidal amplitude falloff.
            (The default is (1, 1) which corresponds to circular falloff).
        value (float): The maximum noise value.
            (The default is 1).
        seed (int): Seed for the random noise generator.
            (The default is None, no seeding)

    Returns:
        np.array: The noisy image array.

    '''
    singleimage = len(images.shape) == 2
    if singleimage:
        images = np.expand_dims(images, axis=0)
    imshape = np.asarray(images.shape)[1:]
    size = np.asarray(size)
    if position is None:
        position = np.floor_divide(imshape, 2)
    position = np.asarray(position)
    xr = np.linspace(0, imshape[0], imshape[0]) 
    yr = np.linspace(0, imshape[1], imshape[1])
    X, Y = np.meshgrid(xr, yr)
    amp = np.sqrt((X-position[0])**2/size[0]+(Y-position[1])**2/size[1])
    amp = value*(np.max(amp)-amp)/np.max(amp)
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(*images.shape)
    noisy = (images + noise*amp).astype(np.float32)
    if singleimage:
        noisy = noisy[0, :, :]
    return noisy


def resize_images(image1, image2, size=(256, 256)):
    ''' Resizes two sets of image tensors.

    To be used in combination with tf.data.Dataset.map to resize tensorflow
    Datasets consisting of two sets of image tensors (one for inputs and one
    for targets).

    Args:
        image1 (tensor): A Tensorflow image tensor.
        image2 (tensor): A Tensorflow image tensor.

    Returns:
        Two resized image tensors.

    '''
    image1.set_shape([None, None, 1])
    image2.set_shape([None, None, 1])
    resized1 = tf.image.resize_images(image1, size)
    resized2 = tf.image.resize_images(image2, size)
    return resized1, resized2
    
    
def generate_stripes(normal, shift, domain=(0, 1, 0, 1), 
                     resolution=(256, 256), threshold=0.95):
    xr = np.linspace(domain[0], domain[1], resolution[0])
    yr = np.linspace(domain[2], domain[3], resolution[1])
    X, Y = np.meshgrid(xr, yr)
    ridge = normal[0]*X + normal[1]*Y - shift
    sin_ridge = np.sin(ridge)
    pos_stripes = np.where(sin_ridge>threshold, 1.0, 0.0)
    neg_stripes = np.where(sin_ridge<-threshold, -1.0, 0.0)
    return pos_stripes + neg_stripes

def generate_stripes_batched(normal, shift, domain=(0, 1, 0, 1), 
                             resolution=(256, 256), threshold=0.95):
    xr = np.linspace(domain[0], domain[1], resolution[0])
    yr = np.linspace(domain[2], domain[3], resolution[1])
    X, Y = np.meshgrid(xr, yr)
    Xflat, Yflat = np.reshape(X, [np.prod(resolution)]), np.reshape(Y, [np.prod(resolution)])
    ridge = np.reshape(np.outer(normal[:,0], Xflat) + np.outer(normal[:, 1], Yflat) - shift, [-1]+list(resolution))
    sin_ridge = np.sin(ridge)
    pos_stripes = np.where(sin_ridge>threshold, 1.0, 0.0)
    neg_stripes = np.where(sin_ridge<-threshold, -1.0, 0.0)
    return pos_stripes + neg_stripes


    
