import numpy as np 
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.models import load_model

import tensorflow as tf

import pickle

import time

seed=1
tf.set_random_seed(seed)
np.random.seed(seed)


class ConvNet_Det():
	def __init__(self):
		self.pretrained_weights = None
		self.input_size = (512,1)
		
		self.convnet = self.build_convnet()
		self.convnet.compile(optimizer = Adam(lr = 1e-3), loss = 'mean_squared_error') #TODO: what loss they use?
	
		self.convnet.summary()

		if(self.pretrained_weights):
			self.convnet.load_weights(self.pretrained_weights)
		
	def build_convnet(self):
		inputs = Input(self.input_size)
		conv1 = Conv1D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(inputs)
		conv2 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv1)
		conv2_1 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2)
		conv3 = Conv1D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2_1)
		conv4 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv3)
		conv5 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv4)
		drop1 = Dropout(0.2)(conv5)
		conv6 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(drop1)
		conv7 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv6)
		conv8 = Conv1D(1, 1, padding = 'valid', kernel_initializer = 'he_normal', strides=1)(conv7)
		
		
		model = Model(inputs = [inputs], outputs = [conv8])

		return model
	
	def train(self, epochs, batch_size, sample_interval, X_train, Y_train, X_val, Y_val, X_test, Y_test, save_params, save_samples):
		
		for epoch in range(epochs):
			start = time.time()
			History = self.convnet.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  verbose = 0,
				  validation_data=(X_val, Y_val))
			end = time.time()
			print ("Epoch %d of %d" % (epoch, epochs))
			print("Train loss: "+ str(History.history["loss"][0]))
			print("Val loss: "+ str(History.history["val_loss"][0]))
			print("Took "+str(end-start)+" seconds")
			print('\n')
			
			# If at save interval => save generated image samples and model
			if epoch % sample_interval == 0:
				if save_params:
					self.convnet.save('convnet_det_epoch'+str(epoch)+'.h5')
				
				if save_samples:
					imgs = X_test
					self.sample_images(epoch, imgs, 'test')
				
					idx = np.arange(6)
					imgs = X_train[idx]
					self.sample_images(epoch, imgs, 'train')
	
	def sample_images(self, epoch, imgs, test_or_train):
		if test_or_train == 'train':
			imgs_out = self.convnet.predict(imgs)
			
			matrices = {'pred':imgs_out, 'uncertainty':imgs_out}
			with open('matrices-train/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if test_or_train == 'test':
			imgs_out = self.convnet.predict(imgs)
			
			matrices = {'pred':imgs_out, 'uncertainty':imgs_out}
			with open('matrices-test/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save(self, save_string):
		self.convnet.save(save_string)
	
	def load(self, load_string):
		self.convnet = load_model(load_string)
		

class ConvNet_Drop():
	def __init__(self):
		self.pretrained_weights = None
		self.input_size = (512,1)
		self.num_draws = 10 # number of samples to draw when doing mc sampling
		
		self.convnet = self.build_convnet()
		self.convnet.compile(optimizer = Adam(lr = 1e-3), loss = 'mean_squared_error') #TODO: what loss they use?
	
		self.convnet.summary()

		if(self.pretrained_weights):
			self.convnet.load_weights(self.pretrained_weights)
		
	def build_convnet(self):
		inputs = Input(self.input_size)
		conv1 = Conv1D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(inputs)
		conv2 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv1)
		conv2_1 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2)
		#drop1 = Dropout(0.2)(conv2)
		conv3 = Conv1D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2_1)#(drop1)
		conv4 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv3)
		conv5 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv4)
		drop1 = Dropout(0.2)(conv5, training = True)
		conv6 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(drop1)#(conv5)
		#drop2 = Dropout(0.2)(conv6)
		conv7 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv6)#(drop2)
		conv8 = Conv1D(1, 1, padding = 'valid', kernel_initializer = 'he_normal', strides=1)(conv7)
		
		
		model = Model(inputs = [inputs], outputs = [conv8])

		return model
	
	def train(self, epochs, batch_size, sample_interval, X_train, Y_train, X_val, Y_val, X_test, Y_test, save_params, save_samples):
		
		for epoch in range(epochs):
			start = time.time()
			History = self.convnet.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  verbose = 0,
				  validation_data=(X_val, Y_val))
			end = time.time()
			print ("Epoch %d of %d" % (epoch, epochs))
			print("Train loss: "+ str(History.history["loss"][0]))
			print("Val loss: "+ str(History.history["val_loss"][0]))
			print("Took "+str(end-start)+" seconds")
			print('\n')
			
			# If at save interval => save generated image samples and model
			if epoch % sample_interval == 0:
				if save_params:
					self.convnet.save('convnet_drop_epoch'+str(epoch)+'.h5')
				
				if save_samples:
					imgs = X_test
					self.sample_images(epoch, imgs, 'test')
				
					idx = np.arange(6)
					imgs = X_train[idx]
					self.sample_images(epoch, imgs, 'train')
	
	def sample_images(self, epoch, imgs, test_or_train):
		if test_or_train == 'train':
		
		
			num_draws = self.num_draws
			results = []
			for i in range(num_draws):
				result = self.convnet.predict(imgs, batch_size = 1)
				results.append(result)
			results = np.array(results)
			#print(results.shape) #results has following dims [num_drams, num_data_points, W, H, channel]


			means = []
			variances = []

			for i in range(imgs.shape[0]):
				mean = np.mean(results[:,i], axis=0)
				means.append(mean)
				variance = np.var(results[:,i], axis=0)
				variances.append(variance)

			means = np.array(means)
			variances = np.array(variances)
			
			matrices = {'pred':means, 'uncertainty':variances}
			with open('matrices-train/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if test_or_train == 'test':
			num_draws = self.num_draws
			results = []
			for i in range(num_draws):
				result = self.convnet.predict(imgs, batch_size = 1)
				results.append(result)
			results = np.array(results)
			#print(results.shape) #results has following dims [num_drams, num_data_points, W, H, channel]


			means = []
			variances = []

			for i in range(imgs.shape[0]):
				mean = np.mean(results[:,i], axis=0)
				means.append(mean)
				variance = np.var(results[:,i], axis=0)
				variances.append(variance)

			means = np.array(means)
			variances = np.array(variances)
			
			matrices = {'pred':means, 'uncertainty':variances}
			with open('matrices-test/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save(self, save_string):
		self.convnet.save(save_string)
	
	def load(self, load_string):
		self.convnet = load_model(load_string)
		
		

class ConvNet_Prob():
	def __init__(self):
		self.pretrained_weights = None
		self.input_size = (512,1)
		self.eps = 1e-3
		
		self.convnet = self.build_convnet()
		self.convnet.compile(optimizer = Adam(lr = 1e-3), loss = self.gaussian_dist_loss) #TODO: what loss they use?
		self.convnet.summary()

		if(self.pretrained_weights):
			self.convnet.load_weights(self.pretrained_weights)
	
	def exponential_dist_loss(self, y_true, y_pred):
		N = y_true.shape[0]
		mean = K.expand_dims(y_pred[:,:,0], axis=-1)
		var = K.expand_dims(y_pred[:,:,1], axis=-1)
		#prevar = K.clip(prevar, -709, 709)
		term1 = K.sum(K.log(var+self.eps))
		#term1 = K.sum(var)
		term2 = K.sqrt(K.sum(K.square(y_true - mean) / (var+self.eps)))
		return (term1 + term2)
	
	def gaussian_dist_loss(self, y_true, y_pred):
		mean = K.expand_dims(y_pred[:,:,0], axis=-1)
		var = K.expand_dims(y_pred[:,:,1], axis=-1)
		
		term1 = K.sum(K.log(var+self.eps))
		
		term2 = K.sum(K.square(y_true - mean) / (var+self.eps))
		return term1 + term2
		
	def build_convnet(self):
		inputs = Input(self.input_size)
		conv1 = Conv1D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(inputs)
		conv2 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv1)
		conv2_1 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2)
		#drop1 = Dropout(0.2)(conv2)
		conv3 = Conv1D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2_1)#(drop1)
		conv4 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv3)
		conv5 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv4)
		drop1 = Dropout(0.2)(conv5)
		conv6 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(drop1)#(conv5)
		#drop2 = Dropout(0.2)(conv6)
		conv7 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv6)#(drop2)
		conv8_mean = Conv1D(1, 1, padding = 'valid', kernel_initializer = 'he_normal', strides=1)(conv7)
		conv8_var = Conv1D(1, 1, activation = 'exponential', padding = 'valid', kernel_initializer = 'he_normal', strides=1)(conv7)
		
		out = Concatenate(axis=-1)([conv8_mean, conv8_var])
		
		
		model = Model(inputs = [inputs], outputs = [out])

		return model
	
	def train(self, epochs, batch_size, sample_interval, X_train, Y_train, X_val, Y_val, X_test, Y_test, save_params, save_samples):
		
		for epoch in range(epochs):
			start = time.time()
			History = self.convnet.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  verbose = 0,
				  validation_data=(X_val, Y_val))
			end = time.time()
			print ("Epoch %d of %d" % (epoch, epochs))
			print("Train loss: "+ str(History.history["loss"][0]))
			print("Val loss: "+ str(History.history["val_loss"][0]))
			print("Took "+str(end-start)+" seconds")
			print('\n')
			
			# If at save interval => save generated image samples and model
			if epoch % sample_interval == 0:
				if save_params:
					self.convnet.save('convnet_prob_epoch'+str(epoch)+'.h5')
				
				if save_samples:
					imgs = X_test
					self.sample_images(epoch, imgs, 'test')
				
					idx = np.arange(6)
					imgs = X_train[idx]
					self.sample_images(epoch, imgs, 'train')
	
	def sample_images(self, epoch, imgs, test_or_train):
		if test_or_train == 'train':
			imgs_out = self.convnet.predict(imgs)
			
			mean_out, var_out = np.expand_dims(imgs_out[:,:,0], axis=-1), np.expand_dims(imgs_out[:,:,1], axis=-1)
			
			matrices = {'pred':mean_out, 'uncertainty':var_out}
			with open('matrices-train/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if test_or_train == 'test':
			imgs_out = self.convnet.predict(imgs)
			
			mean_out, var_out = np.expand_dims(imgs_out[:,:,0], axis=-1), np.expand_dims(imgs_out[:,:,1], axis=-1)
			
			matrices = {'pred':mean_out, 'uncertainty':var_out}
			with open('matrices-test/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save(self, save_string):
		self.convnet.save(save_string)
	
	def load(self, load_string):
		self.convnet = load_model(load_string)
		
class ConvNet_ErrorPred(object):
	def __init__(self):
		self.pretrained_weights = None
		self.input_size = (512,2)
		self.eps = 1e-3
		
		self.convnet = self.build_convnet()
		self.convnet.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['accuracy']) #TODO: what loss they use?
	
		self.convnet.summary()

		if(self.pretrained_weights):
			self.convnet.load_weights(self.pretrained_weights)
		

	def build_convnet(self):
		inputs = Input(self.input_size)
		conv1 = Conv1D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(inputs)
		conv2 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv1)
		conv2_1 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2)
		conv3 = Conv1D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv2_1)
		conv4 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv3)
		conv5 = Conv1D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv4)
		drop1 = Dropout(0.2)(conv5)
		conv6 = Conv1D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(drop1)
		conv7 = Conv1D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', strides=1)(conv6)
		conv8 = Conv1D(1, 1, padding = 'valid', kernel_initializer = 'he_normal', strides=1)(conv7)
		
		
		model = Model(inputs = [inputs], outputs = [conv8])

		return model
	
	def train(self, epochs, batch_size, sample_interval, X_train, Y_train, X_val, Y_val, X_test, Y_test, save_params, save_samples, outer_epoch):
		
		for epoch in range(epochs):
			start = time.time()
			History = self.convnet.fit(X_train, Y_train,
				  batch_size=batch_size,
				  epochs=1,
				  verbose = 0,
				  validation_data=(X_val, Y_val))
			end = time.time()
			print ("Epoch %d of %d" % (epoch, epochs))
			print("Train loss: "+ str(History.history["loss"][0]))
			print("Val loss: "+ str(History.history["val_loss"][0]))
			print("Took "+str(end-start)+" seconds")
			print("\n")
			
			
			# If at save interval => save generated image samples
			if epoch - sample_interval + 1 == 0:
				if save_params:
					self.convnet.save('convnet_errorpred_outerepoch'+str(outer_epoch)+'_epoch'+str(epoch)+'.h5')
				
				if save_samples:
					imgs = X_test
					self.sample_images(epoch, imgs, 'test', outer_epoch)

					idx = np.arange(6)
					imgs = X_train[idx]
					self.sample_images(epoch, imgs, 'train', outer_epoch)
	
	def sample_images(self, epoch, imgs, test_or_train, outer_epoch):
		if test_or_train == 'train':
			error_pred = self.convnet.predict(imgs)
			
			_, imgs_pred = np.expand_dims(imgs[:,:,0], axis=-1), np.expand_dims(imgs[:,:,1], axis=-1)
			
			matrices = {'pred':imgs_pred, 'uncertainty':error_pred}
			with open('matrices-train/outerepoch'+str(outer_epoch)+'_epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if test_or_train == 'test':
			error_pred = self.convnet.predict(imgs)
			
			_, imgs_pred = np.expand_dims(imgs[:,:,0], axis=-1), np.expand_dims(imgs[:,:,1], axis=-1)
			
			matrices = {'pred':imgs_pred, 'uncertainty':error_pred}
			with open('matrices-test/outerepoch'+str(outer_epoch)+'_epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save(self, save_string):
		self.convnet.save(save_string)
	
	def load(self, load_string):
		self.convnet = load_model(load_string)

