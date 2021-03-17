import numpy as np
import matplotlib.pyplot as plt

from Model import ConvNet_Interval

import keras.backend as K
import keras
from keras.datasets import mnist
from keras.layers import Input

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train[:,:,:,np.newaxis]/255
X_test = X_test[:,:,:,np.newaxis]/255

X_val = X_test[:128,:,:,:]


Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)
Y_val = Y_test[:128,:]


convnet_interval = ConvNet_Interval()


num_epochs = 10
interval_epochs = 5
batch_size = 128

width = 0.7

def make_sub_graph(input):
	assert input.shape[0] == 1
	out, min_out, max_out = convnet_interval.sample_tests(input, None)
	plt.subplot(2,1,1)
	plt.imshow(input[0,:,:,0],cmap='gray')
	plt.subplot(2,1,2)
	ind = np.arange(10)
	plt.bar(ind, out[0,:], width, yerr=np.vstack((out[0,:]-min_out[0,:],max_out[0,:]-out[0,:])))


convnet_interval.single_training_run(num_epochs, batch_size, X_train, Y_train, X_val, Y_val)
convnet_interval.interval_training_run(interval_epochs, batch_size, X_train, Y_train, X_val, Y_val, beta=0.02)

counter = 0
while counter < 5:
	indx = np.random.randint(10000)
	out, min_out, max_out = convnet_interval.sample_tests(X_test[indx:indx+1,:,:,:], None)
	if np.argmax(out[0,:]) == np.argmax(Y_test[indx:indx+1,:][0,:]):
		continue
	else:
		counter += 1
	plt.figure()
	make_sub_graph(X_test[indx:indx+1,:,:,:])
for i in range(5):
	indx = np.random.randint(10000)
	out, min_out, max_out = convnet_interval.sample_tests(X_test[indx:indx+1,:,:,:], None)
	plt.figure()
	make_sub_graph(X_test[indx:indx+1,:,:,:])
plt.show()




##########################
###INTERVAL ADVERSARIAL EXAMPLE
##########################

img_in = convnet_interval.interval_convnet.inputs[0]
gt_in = Input([10])
out, out_min, out_max = convnet_interval.both_convnet([img_in, img_in, img_in])
loss = K.mean(K.square(K.max(out*(1-gt_in),0)))

error_func = K.function([img_in, gt_in],[loss])
gradient = K.gradients(loss, img_in)[0]
gradient_func = K.function([img_in, gt_in], [gradient])


for i in range(10):
	sample_idx = np.random.randint(10000)
	img = X_test[sample_idx:sample_idx+1, :, :, :]
	gt = Y_test[sample_idx:sample_idx+1, :]


	pre_pred, pre_min, pre_max = convnet_interval.both_convnet.predict([img,img,img])
	pre_err = np.abs(pre_pred - gt)
	pre_err_pred = pre_max-pre_min
	pre_adv_error = error_func([img, gt])[0]

	tmp_img = img
	for i in range(200):
		adv_gradient = gradient_func([tmp_img, gt])[0]
		regularizer = img-tmp_img
		tmp_img = tmp_img + 1e-2 * adv_gradient/np.linalg.norm(adv_gradient) + 1e-2 * 1e-1 * (regularizer/(np.linalg.norm(regularizer)+1e-1))
	img_adv = tmp_img
	
	plt.figure()
	make_sub_graph(img_adv)
	plt.show()