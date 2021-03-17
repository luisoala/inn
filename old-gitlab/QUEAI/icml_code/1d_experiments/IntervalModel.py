from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from keras_interval_networks.layers import Conv1D, Dropout
from keras_interval_networks.losses import LIMSE, UIMSE

import time
import pickle

import numpy as np
import tensorflow as tf
seed = 1
tf.set_random_seed(seed)
np.random.seed(seed)


class ConvNet_Interval():

    def __init__(self):
        self.pretrained_weights = None
        self.input_size = (512, 1)

        self.creator = self.get_layer_creator()
        self.single_convnet = self.build_single_convnet(self.creator)
        self.interval_convnet = self.build_interval_convnet(self.creator)
        self.both_convnet = self.build_both_convnet(self.creator)

        self.single_convnet.summary()
        self.interval_convnet.summary()
        self.both_convnet.summary()

        if(self.pretrained_weights):
            self.single_convnet.load_weights(self.pretrained_weights)

        self.set_interval_trainable()
        beta = 2e-3
        self.interval_convnet.compile(
            optimizer=Adam(lr=1e-7),
            loss=[LIMSE(beta=beta), UIMSE(beta=beta)],
        )

        self.set_single_trainable()
        self.single_convnet.compile(
            optimizer=Adam(lr=1e-3),
            loss='mean_squared_error',
        )


    def get_layer_creator(self):
        conv1 = Conv1D(16, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        conv2 = Conv1D(32, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        conv2_1 = Conv1D(64, 5, activation='relu', padding='same',
                         kernel_initializer='he_normal', strides=1)
        conv3 = Conv1D(128, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        conv4 = Conv1D(256, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        conv5 = Conv1D(256, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        drop1 = Dropout(0.2)
        conv6 = Conv1D(64, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        conv7 = Conv1D(32, 5, activation='relu', padding='same',
                       kernel_initializer='he_normal', strides=1)
        conv8 = Conv1D(1, 1, activation=None, padding='valid',
                       kernel_initializer='he_normal', strides=1)

        def _build_creator(inputs):
            tmp = conv1(inputs)
            tmp = conv2(tmp)
            tmp = conv2_1(tmp)
            tmp = conv3(tmp)
            tmp = conv4(tmp)
            tmp = conv5(tmp)
            tmp = drop1(tmp)
            tmp = conv6(tmp)
            tmp = conv7(tmp)
            outputs = conv8(tmp)
            return Model(inputs, outputs)

        return _build_creator

    def build_single_convnet(self, creator):
        inputs = Input(self.input_size)
        return creator(inputs)

    def build_interval_convnet(self, creator):
        inputs = [Input(self.input_size), Input(self.input_size)]
        return creator(inputs)

    def build_both_convnet(self, creator):
        inputs = [Input(self.input_size), Input(self.input_size),
                  Input(self.input_size)]
        return creator(inputs)

    def set_single_trainable(self):
        for l in self.single_convnet.layers:
            if isinstance(l, Conv1D):
                l.set_single_trainable()

    def set_interval_trainable(self):
        for l in self.interval_convnet.layers:
            if isinstance(l, Conv1D):
                l.set_minmax_trainable()
                l.reset_minmax()

    def train(self, epochs, batch_size, sample_interval,
              X_train, Y_train, X_val, Y_val, X_test, Y_test,
              save_params, save_samples):

        for epoch in range(1, epochs+1):
            start = time.time()
            History = self.single_convnet.fit(
                X_train,
                Y_train,
                batch_size=batch_size,
                epochs=1,
                verbose=0,
                validation_data=(X_val, Y_val)
            )
            end = time.time()
            print ("Single: Epoch %d of %d" % (epoch, epochs))
            print("\t Train loss: " + str(History.history["loss"][0]))
            print("\t Val loss: " + str(History.history["val_loss"][0]))
            print("\t Took " + str(end-start) + " seconds")
            print('\n')

            # if at save interval => save generated image samples and model
            if epoch % sample_interval == 0:
                if save_params:
                    self.single_convnet.save(
                        'convnet_interval_single_epoch' + str(epoch)+'.h5'
                    )

                if save_samples:
                    # if saving image samples => train intervals first
                    self.train_interval(50, batch_size, epoch, X_train,
                                        Y_train, X_val, Y_val, X_test, Y_test,
                                        save_params)

                    imgs, targets = X_test, Y_test
                    self.sample_images(epoch, imgs, targets, 'test')

                    idx = np.arange(6)
                    imgs, targets = X_train[idx], Y_train[idx]
                    self.sample_images(epoch, imgs, targets, 'train')

                self.set_single_trainable()

    def train_interval(self, epochs, batch_size, sample_epoch,
                       X_train, Y_train, X_val, Y_val, X_test, Y_test,
                       save_params):

        self.set_interval_trainable()
        for epoch in range(1, epochs+1):
            start = time.time()
            History = self.interval_convnet.fit(
               [X_train, X_train],
               [Y_train, Y_train],
               batch_size=batch_size,
               epochs=1,
               verbose=0,
               validation_data=([X_val, X_val], [Y_val, Y_val])
            )
            end = time.time()
            print ("Intervals: Epoch %d of %d" % (epoch, epochs))
            print("\t Train loss: " + str(History.history["loss"][0]))
            print("\t Val loss: " + str(History.history["val_loss"][0]))
            print("\t Took " + str(end-start) + " seconds")
            print('\n')

        if save_params:
            self.interval_convnet.save(
                'convnet_interval_interval_epoch' + str(sample_epoch)+'.h5'
            )

    def sample_images(self, epoch, imgs, targets, test_or_train):
        img_out, min_out, max_out = self.both_convnet.predict(
            [imgs, imgs, imgs]
        )

        matrices = {
            'inputs': imgs,
            'targets': targets,
            'pred': img_out,
            'uncertainty': max_out-min_out,
            'min': min_out,
            'max': max_out,
        }
        with open(
            'matrices-{}/epoch'.format(test_or_train)+str(epoch)+'.pickle',
            'wb',
        ) as handle:
            pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
