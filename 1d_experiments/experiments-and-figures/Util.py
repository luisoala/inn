import numpy as np
from scipy.io import loadmat
from skimage.transform import resize
import matplotlib.pyplot as plt

dir_path = 'D:/...'

np.random.seed(1)
noise_vector_in = np.random.normal(scale=0.05, size=(2000,512))
noise_vector_out = np.random.normal(scale=0.05, size=(2000,512))

def load_batch_pow_8_noisy(batch_indices):
	input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	for i in range(len(batch_indices)):
		mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_8/data_'+str(batch_indices[i]+1)+'.mat')
		output_data[i,0,:] = mat_dict['x'][:,0]+noise_vector_out[i,:]
		input_data[i,0,:] = mat_dict['y'][:,0]+noise_vector_in[i,:]
	return input_data, output_data
	
def load_batch_pow_8(batch_indices):
	input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
	for i in range(len(batch_indices)):
		mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_8/data_'+str(batch_indices[i]+1)+'.mat')
		output_data[i,0,:] = mat_dict['x'][:,0]
		input_data[i,0,:] = mat_dict['y'][:,0]
	return input_data, output_data


