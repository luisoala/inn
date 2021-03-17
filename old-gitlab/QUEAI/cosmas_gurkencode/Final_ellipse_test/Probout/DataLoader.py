import numpy as np
from scipy.io import loadmat
from skimage.transform import resize
import matplotlib.pyplot as plt

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
path = 'D:/Cosmas/Desktop/NN_uncertainty/Ellipse_data'

def load_data_point(data_index):
	mat_dict = loadmat(path+'/data_'+str(data_index+1)+'.mat')
	return resize(mat_dict['f'],(512,512),anti_aliasing=False,mode='constant')

def load_reconstruced_data_point(data_index):
	mat_dict = loadmat(path+'/ellipse_reconstructed_'+str(data_index+1)+'.mat')
	return resize(mat_dict['f'],(512,512),anti_aliasing=False,mode='constant')
	
def load_radon_ellipses_batch(data_indices):
	input_data = np.zeros((len(data_indices),1,512,512),dtype=np.float64)
	output_data = np.zeros((len(data_indices),1,512,512),dtype=np.float64)
	for i in range(len(data_indices)):
		output_data[i,0,:,:] = load_data_point(data_indices[i])
		input_data[i,0,:,:] = load_reconstruced_data_point(data_indices[i])
	return input_data, output_data