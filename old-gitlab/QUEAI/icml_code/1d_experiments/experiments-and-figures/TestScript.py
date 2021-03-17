import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pickle

from Util import load_batch_pow_8, load_batch_pow_8_noisy
from Model import TestConvNet


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


start = 1600


def do_test_run(run_number, noisy=False, noise_index=None, new_data_loading_function=None):
	if noisy:
		data_loading_function = load_batch_pow_8_noisy
		noisy_str = 'Noise'
	else:
		data_loading_function = load_batch_pow_8
		noisy_str = ''
	if new_data_loading_function is not None:
		data_loading_function = new_data_loading_function


	model = TestConvNet()
	if noise_index is not None:
		model.load_state_dict(torch.load(dir_path+'/ModelSaves/BaseModel'+noisy_str+'_run{}_{}.pt'.format(run_number, noise_index)))
	else:
		model.load_state_dict(torch.load(dir_path+'/ModelSaves/BaseModel'+noisy_str+'_run{}.pt'.format(run_number)))
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('Running on',device)
	
	###DROPOUT TESTING
	dropout_sample_dict = {}

	output_list = []
	output_std_list = []
	target_list = []
	input_list = []
	for cut in range(20):
		k = [start+ind for ind in range(cut*10,(cut+1)*10)]
		data_input, data_output = data_loading_function(k)
		input_list.append(data_input.copy()[:,0,:])
		data_input = torch.as_tensor(data_input.astype(np.float32))

		data_input = data_input.to(device)

		output = model.test_forward(data_input.clone())
		output_std = model.compute_dropout_std_uncertainty(data_input.clone())
		output_list.append(output.detach().cpu().numpy()[:,0,:])
		output_std_list.append(output_std.detach().cpu().numpy()[:,0,:])
		target_list.append(data_output[:,0,:])

	dropout_sample_dict['pred'] = np.concatenate(output_list, axis=0)
	dropout_sample_dict['uncertainty'] = np.concatenate(output_std_list, axis=0)
	dropout_sample_dict['target'] = np.concatenate(target_list, axis=0)
	dropout_sample_dict['input'] = np.concatenate(input_list, axis=0)
	

	model = TestConvNet()
	if noise_index is not None:
		model.load_state_dict(torch.load(dir_path+'/ModelSaves/IntervalModel'+noisy_str+'_run{}_{}.pt'.format(run_number, noise_index)))
	else:
		model.load_state_dict(torch.load(dir_path+'/ModelSaves/IntervalModel'+noisy_str+'_run{}.pt'.format(run_number)))
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('Running on',device)
	###INTERVAL TESTING
	interval_sample_dict = {}

	k = [start+ind for ind in range(200)]
	data_input, data_output = data_loading_function(k)
	or_input =  data_input.copy()

	data_input = torch.as_tensor(data_input.astype(np.float32))

	data_input = data_input.to(device)

	output, output_min, output_max = model.forward_normal_and_min_max(data_input.clone())
	output = output.detach().cpu().numpy()[:,0,:]
	output_min = output_min.detach().cpu().numpy()[:,0,:]
	output_max = output_max.detach().cpu().numpy()[:,0,:]

	interval_sample_dict['input'] = or_input[:,0,:]
	interval_sample_dict['pred'] = output
	interval_sample_dict['uncertainty'] = np.maximum(output_max-output_min,0)
	interval_sample_dict['min'] = output_min
	interval_sample_dict['max'] = output_max
	interval_sample_dict['target'] = data_output[:,0,:]



	model = TestConvNet()
	if noise_index is not None:
		model.load_state_dict(torch.load(dir_path+'/ModelSaves/ProboutModel'+noisy_str+'_run{}_{}.pt'.format(run_number, noise_index)))
	else:
		model.load_state_dict(torch.load(dir_path+'/ModelSaves/ProboutModel'+noisy_str+'_run{}.pt'.format(run_number)))
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('Running on',device)
	###PROBOUT TESTING
	probout_sample_dict = {}

	k = [start+ind for ind in range(200)]
	data_input, data_output = data_loading_function(k)
	probout_sample_dict['input'] = data_input.copy()[:,0,:]

	data_input = torch.as_tensor(data_input.astype(np.float32))

	data_input = data_input.to(device)

	output = model.test_forward(data_input.clone())
	output_uncert = model.get_probout_expstd_uncertainty(data_input.clone())
	output = output.detach().cpu().numpy()[:,0,:]
	output_uncert = output_uncert.detach().cpu().numpy()[:,0,:]

	probout_sample_dict['pred'] = output
	probout_sample_dict['uncertainty'] = output_uncert
	probout_sample_dict['target'] = data_output[:,0,:]

	return dropout_sample_dict, interval_sample_dict, probout_sample_dict

