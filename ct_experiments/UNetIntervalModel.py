""" Defines an Interval U-Net. """
import time

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Concatenate, Input, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_interval_networks.losses import LIMSE, UIMSE

from .keras_interval_networks.layers import Conv2D, Dropout, MaxPooling2D

seed = 1
split = -12
beta = 1e-4
tf.set_random_seed(seed)
np.random.seed(seed)
sess = K.get_session()
K.set_session(sess)


class UNetInterval:
    def __init__(
        self,
        load_path=None,
        load_path_interval=None,
        lr=None,
        split=split,
        beta=beta,
    ):
        self.pretrained_weights = load_path is not None
        self.pretrained_interval_weights = load_path_interval is not None
        self.input_size = (512, 512, 1)

        self.creator = self.get_layer_creator()
        self.single_unet = self.build_single_unet(self.creator)
        self.interval_unet = self.build_interval_unet(self.creator)
        self.both_unet = self.build_both_unet(self.creator)
        if self.pretrained_weights:
            self.single_unet.load_weights(load_path, by_name=True)

        self.set_interval_trainable(split=split)
        if self.pretrained_interval_weights:
            self.interval_unet.load_weights(load_path_interval, by_name=True)
        self.interval_unet.compile(
            optimizer=Adam(lr=lr if lr is not None else 1e-6),
            loss=[LIMSE(beta=beta), UIMSE(beta=beta)],
        )

        self.set_single_trainable()
        self.single_unet.compile(
            optimizer=Adam(lr=lr if lr is not None else 1e-4),
            loss="mean_squared_error",
        )

    def get_layer_creator(self):
        # first level convolutional group
        conv1a = Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv1b = Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        pool1 = MaxPooling2D(pool_size=(2, 2))
        drop1 = Dropout(0.7)
        # second level convolutional group
        conv2a = Conv2D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv2b = Conv2D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        pool2 = MaxPooling2D(pool_size=(2, 2))
        drop2 = Dropout(0.7)
        # third level convolutional group
        conv3a = Conv2D(
            256,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv3b = Conv2D(
            256,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        pool3 = MaxPooling2D(pool_size=(2, 2))
        drop3 = Dropout(0.7)
        # fourth level convolutional group
        conv4a = Conv2D(
            512,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv4b = Conv2D(
            512,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        pool4 = MaxPooling2D(pool_size=(2, 2))
        drop4 = Dropout(0.7)
        # fifth level convolutional group
        conv5a = Conv2D(
            1024,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv5b = Conv2D(
            1024,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # fourth level upsampling
        up4 = UpSampling2D(size=(2, 2))
        upconv4 = Conv2D(
            512,
            2,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # fourth level merge
        merge4 = Concatenate(axis=3)
        drop42 = Dropout(0.7)
        # fourth level second convolutional group
        conv42a = Conv2D(
            512,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv42b = Conv2D(
            512,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # third level upsampling
        up3 = UpSampling2D(size=(2, 2))
        upconv3 = Conv2D(
            256,
            2,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # third level merge
        merge3 = Concatenate(axis=3)
        drop32 = Dropout(0.7)
        # third level second convolutional group
        conv32a = Conv2D(
            256,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv32b = Conv2D(
            256,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # second level upsampling
        up2 = UpSampling2D(size=(2, 2))
        upconv2 = Conv2D(
            128,
            2,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # second level merge
        merge2 = Concatenate(axis=3)
        drop22 = Dropout(0.7)
        # second level second convolutional group
        conv22a = Conv2D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv22b = Conv2D(
            128,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # first level upsampling
        up1 = UpSampling2D(size=(2, 2))
        upconv1 = Conv2D(
            64,
            2,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # first level merge
        merge1 = Concatenate(axis=3)
        drop12 = Dropout(0.7)
        # first level second convolutional group
        conv12a = Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        conv12b = Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        # output layer
        convout = Conv2D(
            1,
            1,
            activation=None,
            padding="same",
            kernel_initializer="he_normal",
        )

        def _build_creator(inputs):
            # first level convolutional group
            c1 = conv1b(conv1a(inputs))
            p1 = drop1(pool1(c1))
            # second level convolutional group
            c2 = conv2b(conv2a(p1))
            p2 = drop2(pool2(c2))
            # third level convolutional group
            c3 = conv3b(conv3a(p2))
            p3 = drop3(pool3(c3))
            # fourth level convolutional group
            c4 = conv4b(conv4a(p3))
            p4 = drop4(pool4(c4))
            # fifth level convolutional group
            c5 = conv5b(conv5a(p4))
            # fourth level upsampling
            if isinstance(c4, list):
                u4 = upconv4([up4(c) for c in c5])
                m4 = drop42([merge4([c, u]) for c, u in zip(c4, u4)])
            else:
                u4 = upconv4(up4(c5))
                m4 = drop42(merge4([c4, u4]))
            # fourth level second convolutional group
            c42 = conv42b(conv42a(m4))
            # third level upsampling
            if isinstance(c3, list):
                u3 = upconv3([up3(c) for c in c42])
                m3 = drop32([merge3([c, u]) for c, u in zip(c3, u3)])
            else:
                u3 = upconv3(up3(c42))
                m3 = drop32(merge3([c3, u3]))
            # third level second convolutional group
            c32 = conv32b(conv32a(m3))
            # second level upsampling
            if isinstance(c2, list):
                u2 = upconv2([up2(c) for c in c32])
                m2 = drop22([merge2([c, u]) for c, u in zip(c2, u2)])
            else:
                u2 = upconv2(up2(c32))
                m2 = drop22(merge2([c2, u2]))
            # second level second convolutional group
            c22 = conv22b(conv22a(m2))
            # first level upsampling
            if isinstance(c1, list):
                u1 = upconv1([up1(c) for c in c22])
                m1 = drop12([merge1([c, u]) for c, u in zip(c1, u1)])
            else:
                u1 = upconv1(up1(c22))
                m1 = drop12(merge1([c1, u1]))
            # first level second convolutional group
            c12 = conv12b(conv12a(m1))
            # output layer
            outputs = convout(c12)
            return Model(inputs, outputs)

        return _build_creator

    def build_single_unet(self, creator):
        inputs = Input(self.input_size)
        return creator(inputs)

    def build_interval_unet(self, creator):
        inputs = [Input(self.input_size), Input(self.input_size)]
        return creator(inputs)

    def build_both_unet(self, creator):
        inputs = [
            Input(self.input_size),
            Input(self.input_size),
            Input(self.input_size),
        ]
        return creator(inputs)

    def set_single_trainable(self):
        for l in self.single_unet.layers:
            if isinstance(l, Conv2D):
                l.set_single_trainable()
            l.trainable = True

    def set_interval_trainable(self, split=split):
        for l in self.interval_unet.layers[:split]:
            if isinstance(l, Conv2D):
                l.set_minmax_trainable()
                l.reset_minmax()
            l.trainable = False
        for l in self.interval_unet.layers[split:]:
            if isinstance(l, Conv2D):
                l.set_minmax_trainable()
                l.reset_minmax()
            l.trainable = True

    def train(
        self, epochs, batch_size, train_generator, val_generator, verbose=0
    ):
        self.single_unet.summary()
        for epoch in range(1, epochs + 1):
            start = time.time()
            History = self.single_unet.fit_generator(
                train_generator,
                epochs=1,
                steps_per_epoch=2036 // batch_size,
                verbose=verbose,
                validation_data=val_generator,
                validation_steps=214 // batch_size,
                use_multiprocessing=True,
                workers=32,
            )
            end = time.time()
            print("Single: Epoch %d of %d" % (epoch, epochs))
            print("\t Train loss: " + str(History.history["loss"][0]))
            print("\t Val loss: " + str(History.history["val_loss"][0]))
            print("\t Took " + str(end - start) + " seconds")
            print("\n")

    def train_interval(
        self, epochs, batch_size, train_generator, val_generator, verbose=0
    ):
        self.set_interval_trainable(split=split)
        self.interval_unet.summary()
        for epoch in range(1, epochs + 1):
            start = time.time()
            History = self.interval_unet.fit_generator(
                train_generator,
                epochs=1,
                steps_per_epoch=2036 // batch_size,
                verbose=verbose,
                validation_data=val_generator,
                validation_steps=214 // batch_size,
                use_multiprocessing=True,
                workers=32,
            )
            end = time.time()
            print("Intervals: Epoch %d of %d" % (epoch, epochs))
            print("\t Train loss: " + str(History.history["loss"][0]))
            print("\t Val loss: " + str(History.history["val_loss"][0]))
            print("\t Took " + str(end - start) + " seconds")
            print("\n")
