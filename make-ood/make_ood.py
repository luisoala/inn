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
"""
CONFIGURATIONS BEFORE WE START
"""
artifact_path = 'ood_target'
artifact_path2 = 'ood_lat_rec'

"""
INPUTS
#assumes there exists a dir relative to the script file in which files input_img.png, target_img.png, adv_artifact_target.png, and model_weights.h5 can be found
"""
#args
data_path = sys.argv[1]#path to folder with data .npz files, include /
ood_path = sys.argv[2] #path to ood artifact
data_prefix = 'mayo_test_' # prefix before 000.npz
img_nrows, img_ncols, n_channels = 512, 512, 1
exp_name = 'dove_ood'

#make tmp dir to store intermediate results
tmp_path = 'ood_tmp'
os.mkdir(tmp_path)
npz_path = 'ood'
os.mkdir(npz_path)

#mlflow
mlflow.set_experiment(exp_name)
mlflow.start_run(run_name=datetime.datetime.fromtimestamp(time.time()).strftime('%c')) #start the mlflow run for logging

try:
	"""
	READ DATA
	"""
	
	for i in range(330):
		#get original data
		test_sample = np.load(data_path+data_prefix+'{:03d}.npz'.format(i))
		input_img = test_sample['input'][0,:,:,0:1]
		target_img = test_sample['target'][0,:,:,0:1]
		recon_img = test_sample['reconstruction'][0,:,:,0:1]
		
		#create ood img in image domain
		ood_img_artifact = np.zeros_like(target_img[:,:,0])
		x = np.random.randint(0,256)
		ood_img_artifact[128:128+256,x:x+256] = plt.imread(ood_path)[:,:,0]#TODO:wiggle a little along a similar axis
		#print(ood_img_artifact.shape)
		target_copy = np.copy(target_img[:,:,0])
		mean = np.mean(target_copy)
		ood_img_artifact[ood_img_artifact!=0] = mean
		target_copy[ood_img_artifact!=0]=0
		ood_img = target_copy + ood_img_artifact

		#map ood to measurement domain
		biggest_dim = 512
		theta = np.linspace(0.0, 180.0, biggest_dim, endpoint=False)
		#theta_limited = np.concatenate([theta[:(theta.size-wedge)//2], theta[(theta.size+wedge)//2:]]
		sinogram = radon(ood_img, theta=theta, circle=False)
		sinogram[:,:(15*512)//180] = .0
		sinogram[:,-(15*512)//180:] = .0
		
		#lat reconstruction of ood
		ood_img_lat_rec = iradon(sinogram, theta=theta, circle=False)
		ood_img_lat_rec = skimage.exposure.rescale_intensity(ood_img_lat_rec, out_range=(.0,1.))
		
		#do all inspection logging here
		if i % 30 == 0:
			fname = tmp_path + '/' + 'ood_target.png'
			save_img(fname, ood_img[:,:,np.newaxis])
			mlflow.log_artifact(fname, artifact_path=artifact_path)
			
			fname = tmp_path + '/' + 'ood_img_lat_rec.png'
			save_img(fname, ood_img_lat_rec[:,:,np.newaxis])
			mlflow.log_artifact(fname, artifact_path=artifact_path2)
			
		#do all continual saving here
		#saving
		np.savez_compressed(
                os.path.join(
                    npz_path,
                    "mayo_test_{:03d}.npz".format(i),
                ),
                input=input_img[np.newaxis,:,:,:],
                target=target_img[np.newaxis,:,:,:],
                ood_target=ood_img[np.newaxis,:,:,np.newaxis],
                ood_lat_rec=ood_img_lat_rec[np.newaxis,:,:,np.newaxis],
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
