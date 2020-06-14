from __future__ import print_function

"""
REFERENCES
"""
#https://keras.io/examples/neural_style_transfer/
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
#https://docs.scipy.org/doc/scipy/reference/optimize.html

"""
IMPORTS
"""

#keras
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras import backend as K
from keras.models import load_model
from keras.models import *
from tensorflow import Graph, Session

#python
from skimage.io import imread, imshow, imread_collection, concatenate_images
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import matplotlib.pyplot as plt
import sys
import mlflow
import os
import shutil
import datetime

#ood & adv
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import radon, iradon
import skimage
from skimage import draw

#my utils and extras
#from utils import *

#model import
from UNetDropoutModel import *
from UNetIntervalModel import *
from UNetProboutModel import *
from UNetProboutModelFindAdv import *

#TODO: bound option
#TODO: easy parameter carry over for visualization notebook (corrspondence between experiments and visualization session)
"""
UTILS
"""
# util function to convert a tensor into a valid image
def deprocess_image(x):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((n_channels, img_nrows, img_ncols))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((img_nrows, img_ncols, n_channels))
	return x

def eval_loss_and_grads(x):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((1, n_channels, img_nrows, img_ncols))
	else:
		x = x.reshape((1, img_nrows, img_ncols, n_channels))
	outs = f_outputs([x])
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')
	return loss_value, grad_values
	
class Evaluator(object):

	def __init__(self):
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values
		
		
def min_max(x):
	return (np.min(x), np.max(x))




"""
CONFIGURATIONS BEFORE WE START
"""
np.random.seed(42)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


artifact_path = 'adversarial-inputs'#None #path to store artifacts for the run
artifact_path2 = 'adversarial-preds'
artifact_path3 = 'original-inputs'
artifact_path4 = 'targets'
artifact_path5 = 'adv-targets'
"""
INPUTS
#assumes there exists a dir relative to the script file in which files input_img.png, target_img.png, adv_artifact_target.png, and model_weights.h5 can be found
"""
#args
iterations = 1 # number of iterations to run the optimization, e.g. 20000
data_path = sys.argv[1]#path to folder with data .npz files, include /
data_prefix = 'mayo_test_' # prefix before 000.npz
model_path = sys.argv[2] # path to model weights'/model_weights.h5'#
model_type = sys.argv[3] #drop or prob
run = sys.argv[4]#run number
adv_shape = sys.argv[5]
img_nrows, img_ncols, n_channels = 512, 512, 1

#make tmp dir to store intermediate results
tmp_path = 'adversarials_'+model_type+'_'+run+'_tmp'
os.mkdir(tmp_path)
npz_path = 'adversarials_'+model_type+'_'+run
os.mkdir(npz_path)
#bounds = True
epsilon = 0.01

#mlflow
mlflow.set_experiment(model_type+'_'+run)
mlflow.start_run(run_name=datetime.datetime.fromtimestamp(time.time()).strftime('%c')) #start the mlflow run for logging
mlflow.log_param("eps", epsilon)

try:

	"""
	PREPARE MODEL
	"""
	if K.image_data_format() == 'channels_first':
		combination_image = K.placeholder((1, 1, img_nrows, img_ncols))
	else:
		combination_image = K.placeholder((1, img_nrows, img_ncols, 1)) #the generated image, i.e. the adversarial example
	input_tensor = combination_image

	#load the model
	if model_type == 'drop':
		model_raw = UNetDropout(model_path).model
	elif model_type == 'prob':
		model_raw = UNetProboutFindAdv(model_path).model
	
	 #model_raw is the raw prediction model for which we try to find the adversarial examole, we need to do some modifications in order to run the optimization to find the adversarial perturbation in the input
	model_raw.layers.pop(0)

	img_input = Input(tensor=input_tensor, shape=(img_nrows, img_ncols, n_channels))
	output_tensor = model_raw(img_input)
	model = Model([img_input], [output_tensor]) #model is the representation of the prediction model which we can use for the optimization to find the adversarial noise
	print('Model loaded.')

	# get the symbolic output of the model
	output_img = model.output
	
	"""
	READ DATA
	"""
	
	for i in range(330):
		test_sample = np.load(data_path+data_prefix+'{:03d}.npz'.format(i))
		
		input_img = test_sample['input'][0,:,:,0:1]
		target_img = test_sample['target'][0,:,:,0:1]
		recon_img = test_sample['reconstruction'][0,:,:,0:1]
		

		"""
		PREPARE DATA
		"""
		#make adv target image
		
		mean = np.mean(recon_img)
		fill_number = mean*1.5
		
		if adv_shape == 'square':
			#SQUARE
			sigma = 2
			rectangle_y = np.random.randint(160,260)
			rectangle_x = np.random.randint(112,400)
			extent = 50
			
			#now squares shadow
			recon_img_copy = np.copy(recon_img)
			arr = np.zeros((512, 512))
			rr, cc = draw.rectangle((rectangle_y, rectangle_x), extent = extent, shape=arr.shape)
			#rr, cc = disk(256, 256, radius=10, shape=arr.shape)
			arr[rr, cc] = fill_number
			arr = skimage.filters.gaussian(arr, sigma=sigma, preserve_range=True)
		
		elif adv_shape == 'circle':
			#CIRCLE
			sigma = 2
			radius = 20
			circle_y = np.random.randint(160,260)
			circle_x = np.random.randint(112,400)
			#circle shadow
			recon_img_copy = np.copy(recon_img)
			arr = np.zeros((512, 512))
			rr, cc = draw.circle(circle_y, circle_x , radius=radius, shape=arr.shape)
			arr[rr, cc] = fill_number
			arr = skimage.filters.gaussian(arr, sigma=sigma, preserve_range=True)

		adv_target_img = recon_img_copy - arr[:,:,np.newaxis]
		adv_target_img[adv_target_img<0.] = np.min(recon_img_copy)

		#ensure range
		input_img_min, input_img_max = min_max(input_img)
		recon_img_min, recon_img_max = min_max(recon_img)
		adv_target_img_min, adv_target_img_max = min_max(adv_target_img)
		

		if adv_target_img_min < recon_img_min or adv_target_img_max > recon_img_max:
			print('Range warning at input {}'.format(i))
			print('adv_target_img_min: {} | target_img_min: {} | adv_target_img_max: {} | target_img_max: {}'.format(adv_target_img_min, recon_img_min, adv_target_img_max, recon_img_max))
		
		"""
		FIND THE ADVERSARIAL INPUT
		"""
		# create symbolic tensors for input and adv_target tensors
		adv_target_tensor = K.variable(adv_target_img)
		

		# define symbolic loss
		loss = K.variable(0.0)
		loss += K.mean(K.square(adv_target_tensor - output_img))
		#loss += K.mean(K.binary_crossentropy(K.flatten(adv_target_tensor), K.flatten(output_img))) #binary ce as alternative

		# get the gradients of the generated image wrt the loss
		grads = K.gradients(loss, combination_image)

		outputs = [loss]
		if isinstance(grads, (list, tuple)):
			outputs += grads
		else:
			outputs.append(grads)

		f_outputs = K.function([combination_image], outputs)

		# this Evaluator class makes it possible
		# to compute loss and gradients in one pass
		# while retrieving them via two separate functions,
		# "loss" and "grads". This is done because scipy.optimize
		# requires separate functions for loss and gradients,
		# but computing them separately would be inefficient.
		evaluator = Evaluator()

		# run scipy-based optimization (L-BFGS) over the pixels of the generated image
		# so as to minimize the neural style loss
		
		x = input_img
		best_min_val = 1000.
		best_it = 0

		for j in range(iterations):
			x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, epsilon=epsilon)#, maxfun=100, epsilon=epsilon)

			if min_val < best_min_val:
				img = x.copy()
				best_min_val = min_val
				best_it = j
			
		
		"""
		do all continual loggin here
		"""
		mlflow.log_metric("adv_loss", min_val) #log loss
		
		"""
		do all inspection logging here
		"""
		if i % 30 == 0:
			#log input_img
			fname = tmp_path+"/input_%d.png" % i
			save_img(fname, input_img)
			mlflow.log_artifact(fname, artifact_path=artifact_path3)
			#log target_img
			fname = tmp_path + "/target_%d.png" % i
			save_img(fname, target_img)
			mlflow.log_artifact(fname, artifact_path=artifact_path4)
			#log adv_target_img
			fname = tmp_path + "/adv_target_%d.png" % i
			save_img(fname, adv_target_img)
			mlflow.log_artifact(fname, artifact_path=artifact_path5)
			
			img = deprocess_image(x.copy())
			fname = tmp_path + '/' + 'adv_input_%d.png' % i
			save_img(fname, img)
			mlflow.log_artifact(fname, artifact_path=artifact_path)#log adv input
			
			adv_pred = model_raw.predict(img[np.newaxis,:,:,:])
			fname = tmp_path + '/' + 'predon_adv_input_%d.png' % i
			save_img(fname, adv_pred[0,:,:,:])
			mlflow.log_artifact(fname, artifact_path=artifact_path2)
		
		"""
		do all continual saving here
		"""
		#saving
		np.savez_compressed(
                os.path.join(
                    npz_path,
                    "mayo_test_{:03d}.npz".format(i),
                ),
                input=input_img[np.newaxis,:,:,:],
                target=target_img[np.newaxis,:,:,:],
                adv_target=adv_target_img,
                adv_input=deprocess_image(img)[np.newaxis,:,:,:],
            )

finally:
	"""
	CLEAN UP
	"""
	shutil.rmtree(tmp_path)
	shutil.make_archive(npz_path, 'zip', npz_path)
	mlflow.log_artifact(npz_path+'.zip')
	
	os.remove(npz_path+'.zip')
	shutil.rmtree(npz_path)
