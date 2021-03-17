import numpy as np
from scipy.io import loadmat
from skimage.transform import resize
import matplotlib.pyplot as plt

dir_path = 'D:/Cosmas/Desktop/NN_uncertainty/GenuineTests/1DInverseProblem'
#dir_path = '/media/oala/4TB/DATA/experiments-hhi/smoothed_signal/Uncertainty_IP'

def load_batch_pow_2(batch_indices):
	input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	for i in range(len(batch_indices)):
		mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_2/data_'+str(batch_indices[i]+1)+'.mat')
		output_data[i,0,:] = mat_dict['x'][:,0]
		input_data[i,0,:] = mat_dict['y'][:,0]
	return input_data, output_data
	
def load_batch_pow_4(batch_indices):
	input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	for i in range(len(batch_indices)):
		mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_4/data_'+str(batch_indices[i]+1)+'.mat')
		output_data[i,0,:] = mat_dict['x'][:,0]
		input_data[i,0,:] = mat_dict['y'][:,0]
	return input_data, output_data
	
def load_batch_pow_8(batch_indices):
	input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	for i in range(len(batch_indices)):
		mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_8/data_'+str(batch_indices[i]+1)+'.mat')
		output_data[i,0,:] = mat_dict['x'][:,0]
		input_data[i,0,:] = mat_dict['y'][:,0]
	return input_data, output_data