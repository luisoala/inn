import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch
import time

from Model import TestConvNet
from TrainingScripts import do_base_training_run, do_interval_training_run, do_probout_training_run


dir_path = 'D:/...'

np.random.seed(1)
noise_vector_in = [np.random.normal(scale=0.02, size=(2000,512)), np.random.normal(scale=0.05, size=(2000,512)), np.random.normal(scale=0.08, size=(2000,512)), np.random.normal(scale=0.11, size=(2000,512)), np.random.normal(scale=0.14, size=(2000,512))]
noise_vector_out = [np.random.normal(scale=0.02, size=(2000,512)), np.random.normal(scale=0.05, size=(2000,512)), np.random.normal(scale=0.08, size=(2000,512)), np.random.normal(scale=0.11, size=(2000,512)), np.random.normal(scale=0.14, size=(2000,512))]


def get_noisy_batch_pow_8(noise_level):
	def load_batch_pow_8(batch_indices):
		input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
		output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
		for i in range(len(batch_indices)):
			mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_8/data_'+str(batch_indices[i]+1)+'.mat')
			output_data[i,0,:] = mat_dict['x'][:,0]+noise_vector_out[noise_level][i,:]
			input_data[i,0,:] = mat_dict['y'][:,0]+noise_vector_in[noise_level][i,:]
		return input_data, output_data
	return load_batch_pow_8


import os 
new_dir_path = os.path.dirname(os.path.realpath(__file__))

sigmas = [0.02,0.05,0.08,0.11,0.14,0.17]
betas = [0.002,0.002,0.003,0.005,0.008,0.01]
for noise_level in range(5,6):
	beta = betas[noise_level]
	num_epochs = 100
	learning_rate = 1e-3
	interval_learning_rate = 1e-6
	probout_learning_rate = 1e-4
	batch_size = 256

	num_interval_epochs = 100
	num_probout_epochs = 100

	data_loading_function = get_noisy_batch_pow_8(noise_level)

	num_training_runs = 1

	for i in range(num_training_runs):
		print('############################')
		print('Beginning noisy run [{}/{}], beta={}'.format(i+1, num_training_runs, beta))
		print('############################')
		model = TestConvNet()

		model = do_base_training_run(model, num_epochs, learning_rate, batch_size, data_loading_function)
		torch.save(model.state_dict(), new_dir_path+'/ModelSaves/BaseModel_run{}_{}.pt'.format(i, noise_level))

		model = do_interval_training_run(model, num_interval_epochs, interval_learning_rate, batch_size, beta, data_loading_function)
		torch.save(model.state_dict(), new_dir_path+'/ModelSaves/IntervalModel_run{}_{}.pt'.format(i, noise_level))

		model = TestConvNet()
		model.load_state_dict(torch.load(new_dir_path+'/ModelSaves/BaseModel_run{}_{}.pt'.format(i, noise_level)))
		model = do_probout_training_run(model, num_probout_epochs, probout_learning_rate, batch_size, data_loading_function)
		torch.save(model.state_dict(), new_dir_path+'/ModelSaves/ProboutModel_run{}_{}.pt'.format(i, noise_level))
