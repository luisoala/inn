import numpy as np
import pickle
import os
import sys

import keras.backend as K

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

sys.path.append(os.path.join('..'))
from keras_interval_networks.layers import Conv2D, MaxPooling2D, Dense, Flatten
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
		self.input_size = (28,28, 1)

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
		beta = 5e-4
		self.interval_convnet.compile(
			optimizer=Adam(lr=1e-7),
			loss=[LIMSE(beta=beta), UIMSE(beta=beta)],
		)
		self.initialized_beta = beta

		self.set_single_trainable()
		self.single_convnet.compile(
			optimizer=Adam(lr=1e-3),
			loss='mean_squared_error',
		)
		
	def get_layer_creator(self):
		conv1 = Conv2D(16, (5,5), activation='relu', padding='valid',
					   kernel_initializer='he_normal', strides=(1,1))
		pool1 = MaxPooling2D(pool_size=(2,2))
		conv2 = Conv2D(32, (5,5), activation='relu', padding='valid',
					   kernel_initializer='he_normal', strides=(1,1))
		pool2 = MaxPooling2D(pool_size=(2,2))
		conv3 = Conv2D(64, (5,5), activation='relu', padding='valid',
						 kernel_initializer='he_normal', strides=(1,1))
		pool3 = MaxPooling2D(pool_size=(2,2))
		conv4 = Conv2D(128, (5,5), activation='relu', padding='valid',
					   kernel_initializer='he_normal', strides=(1,1))
		pool4 = MaxPooling2D(pool_size=(2,2))
		conv5 = Conv2D(256, (5,5), activation='relu', padding='valid',
					   kernel_initializer='he_normal', strides=(1,1))
		pool5 = MaxPooling2D(pool_size=(2,2))
					   
		dense1 = Dense(1024,activation='relu')
		dense2 = Dense(512,activation='relu')
		dense3 = Dense(10)

		def _build_creator(inputs):
			tmp = conv1(inputs)
			tmp = pool1(tmp)
			tmp = conv2(inputs)
			tmp = pool2(tmp)
			tmp = conv3(inputs)
			tmp = pool3(tmp)
			tmp = conv4(inputs)
			tmp = pool4(tmp)
			tmp = conv5(inputs)
			tmp = pool5(tmp)
			tmp = dense1(Flatten()(tmp))
			tmp = dense2(tmp)
			outputs = dense3(tmp)
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
			if isinstance(l, (Conv2D, Dense)):
				l.set_single_trainable()

	def set_interval_trainable(self):
		for l in self.interval_convnet.layers:
			if isinstance(l, (Conv2D, Dense)):
				l.set_minmax_trainable()
				l.reset_minmax()
				
				
	def initialize_new_beta(self, beta):
		if beta==self.initialized_beta: return
		self.interval_convnet.compile(
			optimizer=Adam(lr=1e-7),
			loss=[LIMSE(beta=beta), UIMSE(beta=beta)],
		)
		self.initialized_beta = beta
		
		
	def single_training_run(self, epochs, batch_size, X_train, Y_train, X_val, Y_val):
		self.set_single_trainable()
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
			
	def get_heuristic_beta(self, X_val, Y_val):
		output = self.single_convnet.predict(X_val)
		beta_choice = 0.1*K.get_value(K.mean(K.abs(output-Y_val)))
		return beta_choice
		
		
	def interval_training_run(self, epochs, batch_size, X_train, Y_train, X_val, Y_val, beta=-1):
		self.set_interval_trainable()
		if beta == -1:
			beta = self.get_heuristic_beta(X_val, Y_val)
		self.initialize_new_beta(beta)
		
		for epoch in range(1,epochs+1):
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
			print (("Intervals: Epoch %d of %d with beta="+str(beta)) % (epoch, epochs))
			print("\t Train loss: " + str(History.history["loss"][0]))
			print("\t Val loss: " + str(History.history["val_loss"][0]))
			print("\t Took " + str(end-start) + " seconds")
			print('\n')
			
	def sample_tests(self, X_test, Y_test):
		output, min_output, max_output = self.both_convnet.predict([X_test, X_test, X_test])
		return output, min_output, max_output