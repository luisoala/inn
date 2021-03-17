from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

from keras import activations

import matplotlib.pyplot as plt

import numpy as np

"""
from the unet notebook
"""
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

class ContextEncoder():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.mask_height = 32
        self.mask_width = 32
        self.channels = 1
        self.channels_out = 2
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        out_mean, out_b = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(out_mean)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_img , [out_mean, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)
    
    def exponential_dist_loss(y_true, y_pred):
        mean = y_pred[0]
        var = y_pred[1]
        term1 = K.sum(K.log(var))
        term2 = K.sqrt(K.sum(K.square(y_true - mean) / var))
        return term1 + term2

    def build_generator(self):


        gen_input = Input(shape=self.img_shape)

        # Encoder
        c1 = Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same")(gen_input)
        c1 = LeakyReLU(alpha=0.2)(c1)
        c1 = BatchNormalization(momentum=0.8)(c1)
        
        c2 = Conv2D(64, kernel_size=3, strides=2, padding="same")(c1)
        c2 = LeakyReLU(alpha=0.2)(c2)
        c2 = BatchNormalization(momentum=0.8)(c2)
        
        c3 = Conv2D(128, kernel_size=3, strides=2, padding="same")(c2)
        c3 = LeakyReLU(alpha=0.2)(c3)
        c3 = BatchNormalization(momentum=0.8)(c3)

        c4 = Conv2D(512, kernel_size=1, strides=2, padding="same")(c3)
        c4 = LeakyReLU(alpha=0.2)(c4)
        c4 = Dropout(0.5)(c4)

        # Decoder
        up1 = UpSampling2D()(c4)
        up1 = Conv2D(128, kernel_size=3, padding="same")(up1)
        up1 = Activation('relu')(up1)
        up1 = BatchNormalization(momentum=0.8)(up1)
        
        up2 = UpSampling2D()(up1)
        up2 = Conv2D(64, kernel_size=3, padding="same")(up2)
        up2 = Activation('relu')(up2)
        up2 = BatchNormalization(momentum=0.8)(up2)
        
        out_mean = Conv2D(self.channels, kernel_size=3, padding="same")(up2)
        out_mean = Activation('tanh')(out_mean)
        
        out_b = Conv2D(self.channels, kernel_size=3, padding="same")(up2)
        out_b = Activation(K.exp)(out_b) #TODO: constrain variance outputs meaningfully as you did with mean
        
        
        model = Model(inputs=[gen_input], outputs=[out_mean, out_b])

        model.summary()
        

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.missing_shape)
        validity = model(img)

        return Model(img, validity)

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)
        
    def mask_center(self, imgs):
        y1 = (self.img_rows// 2) - (self.mask_height // 2)
        y2 = y1 + self.mask_height
        x1 = (self.img_cols // 2) - (self.mask_width // 2)
        x2 = x1 + self.mask_width
        #make lists
        y1 = [y1]*imgs.shape[0]
        y2 = [y2]*imgs.shape[0]
        x1 = [x1]*imgs.shape[0]
        x2 = [x2]*imgs.shape[0]

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)



    def train(self, epochs, batch_size=64, sample_interval=500):

        # Load the dataset
        IMG_WIDTH = self.img_cols
        IMG_HEIGHT = self.img_rows
        DROP_WIDTH = self.mask_width
        DROP_HEIGHT = self.mask_height
        IMG_CHANNELS = self.channels
        TRAIN_PATH = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/f-only-rescaled-png-uint16/'
        TEST_PATH = '/media/oala/4TB/DATA/experiments-hhi/uncertainty-quant/ellipsoid-toy/f-only-rescaled-png-uint16/'
        RESIZE = True

        CENTER_WIDTH = (IMG_WIDTH // 2) - (DROP_WIDTH // 2)
        CENTER_HEIGHT = (IMG_HEIGHT// 2) - (DROP_HEIGHT // 2)
        
        train_names = next(os.walk(TRAIN_PATH))[2]
        X_train = np.zeros((len(train_names), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint16)
        print('Getting and resizing train images and masks ... ')
        for image_name, count in zip(train_names, range(len(train_names))):
            img = imread(TRAIN_PATH + image_name)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_train[count] = img
            if count % 500 == 0:
                print('Done with # ', count)
        X_train = X_train[:,:,:,np.newaxis]

        # Rescale -1 to 1
        X_train = X_train*2. / 65535. - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            masked_imgs, missing_parts, _ = self.mask_center(imgs)

            # Generate a batch of new images
            gen_missing, out_b = self.generator.predict(masked_imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

            # Plot the progress
            if epoch % 200 == 0:
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_center(imgs)
        gen_missing = self.generator.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        for i in range(c):
            axs[0,i].imshow(np.squeeze(imgs[i, :,:]))
            axs[0,i].axis('off')
            axs[1,i].imshow(np.squeeze(masked_imgs[i, :,:]))
            axs[1,i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
            axs[2,i].imshow(np.squeeze(filled_in))
            axs[2,i].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    context_encoder = ContextEncoder()
    context_encoder.train(epochs=30000, batch_size=64, sample_interval=50)
