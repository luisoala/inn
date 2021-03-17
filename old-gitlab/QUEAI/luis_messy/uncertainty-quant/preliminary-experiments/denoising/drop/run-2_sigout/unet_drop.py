import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.models import load_model

#for data processing
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf

import pickle

import time

seed=1
tf.set_random_seed(seed)
np.random.seed(seed)

class Unet():
	def __init__(self):
		self.pretrained_weights = None
		self.input_size = (512,512,1)
		
		self.unet = self.build_unet()
		self.unet.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['accuracy']) #TODO: what loss they use?
	
		self.unet.summary()

		if(self.pretrained_weights):
			self.unet.load_weights(self.pretrained_weights)
		
	def build_unet(self):
		inputs = Input(self.input_size)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = BatchNormalization()(conv1)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization()(conv1)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		conv1 = BatchNormalization()(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = BatchNormalization()(conv2)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		conv2 = BatchNormalization()(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = BatchNormalization()(conv3)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = BatchNormalization()(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = BatchNormalization()(conv4)
		drop4 = Dropout(0.5)(conv4, training=True)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = BatchNormalization()(conv5)
		drop5 = Dropout(0.5)(conv5, training=True)

		up6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		up6 = BatchNormalization()(up6)
		merge6 = concatenate([drop4,up6])
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = BatchNormalization()(conv6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		conv6 = BatchNormalization()(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		up7 = BatchNormalization()(up7)
		merge7 = concatenate([conv3,up7])
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = BatchNormalization()(conv7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = BatchNormalization()(conv7)

		up8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		up8 = BatchNormalization()(up8)
		merge8 = concatenate([conv2,up8])
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = BatchNormalization()(conv8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8)

		up9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		up9 = BatchNormalization()(up9)
		merge9 = concatenate([conv1,up9])
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = BatchNormalization()(conv9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = BatchNormalization()(conv9)
	
		conv9 = Conv2D(1, 1, padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(1, 1, padding = 'same', kernel_initializer = 'he_normal')(conv9)
		resid = Add()([conv9, inputs])
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(resid) #TODO: what output activation they use?

		model = Model(inputs = [inputs], outputs = [conv10])

		return model
	
	def train(self, epochs, batch_size, sample_interval, X_train, Y_train, X_val, Y_val, X_test, Y_test):
		
		for epoch in range(epochs):
			start = time.time()
			History = self.unet.fit(X_train, Y_train,
				  batch_size=1,
				  epochs=1,
				  verbose = 0,
				  validation_data=(X_val, Y_val))
			end = time.time()
			print ("Epoch %d of %d" % (epoch, epochs))
			print("Train loss: "+ str(History.history["loss"][0]))
			print("Val loss: "+ str(History.history["val_loss"][0]))
			print("Took "+str(end-start)+" seconds")
			print('\n')
			
			# If at save interval => save generated image samples
			if epoch % sample_interval == 0:
				idx = np.random.randint(0, X_test.shape[0], 6)
				imgs = X_test[idx]
				refs = Y_test[idx]
				self.sample_images(epoch, imgs, refs, 'test')

			if epoch % sample_interval == 0:
				idx = np.random.randint(0, X_train.shape[0], 6)
				imgs = X_train[idx]
				refs = Y_train[idx]
				self.sample_images(epoch, imgs, refs, 'train')
	
	def sample_images(self, epoch, imgs, refs, test_or_train):
		if test_or_train == 'train':
			imgs_out = self.unet.predict(imgs)
			
			matrices = {'imgs':imgs, 'imgs_out':imgs_out, 'refs':refs}
			with open('matrices-train/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
		if test_or_train == 'test':
			imgs_out = self.unet.predict(imgs)
			
			matrices = {'imgs':imgs, 'imgs_out':imgs_out, 'refs':refs}
			with open('matrices-test/epoch'+str(epoch)+'.pickle', 'wb') as handle:
				pickle.dump(matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save(self, save_string):
		self.unet.save(save_string)
	
	def load(self, load_string):
		self.unet = load_model(load_string)
