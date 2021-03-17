import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from Util import load_batch_pow_8
from Model import TestConvNet
from TrainingScripts import do_base_training_run, do_interval_training_run, do_probout_training_run

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


beta = 0.002
num_epochs = 100
learning_rate = 1e-3
interval_learning_rate = 1e-5
probout_learning_rate = 1e-4
batch_size = 256

num_interval_epochs = 100
num_probout_epochs = 100

data_loading_function = load_batch_pow_8

num_training_runs = 1

for i in range(num_training_runs):
	print('############################')
	print('Beginning run [{}/{}]'.format(i+1, num_training_runs))
	print('############################')
	model = TestConvNet()

	model = do_base_training_run(model, num_epochs, learning_rate, batch_size, data_loading_function)
	torch.save(model.state_dict(), dir_path+'/ModelSaves/BaseModel_run{}.pt'.format(i))

	model = do_interval_training_run(model, num_interval_epochs, interval_learning_rate, batch_size, beta, data_loading_function)
	torch.save(model.state_dict(), dir_path+'/ModelSaves/IntervalModel_run{}.pt'.format(i))

	model = TestConvNet()
	model.load_state_dict(torch.load(dir_path+'/ModelSaves/BaseModel_run{}.pt'.format(i)))
	model = do_probout_training_run(model, num_probout_epochs, probout_learning_rate, batch_size, data_loading_function)
	torch.save(model.state_dict(), dir_path+'/ModelSaves/ProboutModel_run{}.pt'.format(i))