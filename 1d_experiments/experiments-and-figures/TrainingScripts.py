import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from Model import TestConvNet
from Util import load_batch_pow_8

def do_base_training_run(model, num_epochs, learning_rate, batch_size, data_loading_function):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	print('Running on',device)

	print('Initializing optimizer..')
	min_max_parameters = ['min_weight','max_weight','min_bias','max_bias']

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if not name.split('.')[-1] in min_max_parameters], lr=learning_rate)
	print('Optimizer initialized.')


	for epoch in range(1,num_epochs+1):
		start_time = time.time()
		
		for i in range(1600//batch_size):
			k = [np.random.randint(1600) for i in range(batch_size)]
			data_input, data_output = data_loading_function(k)
			
			data_input = torch.as_tensor(data_input.astype(np.float32))
			data_output = torch.as_tensor(data_output.astype(np.float32))
			
			data_input = data_input.to(device)
			data_output = data_output.to(device)
			
			input_img = torch.autograd.Variable(data_input)
			output_target = torch.autograd.Variable(data_output)
			
			output = model(input_img)
			
			loss = criterion(output, output_target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		stop_time = time.time()
		time_el = int(stop_time-start_time)
		print('power {}, epoch [{}/{}], loss:{:.4f} in {}h {}m {}s'.format(8, epoch, num_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))

	model.to('cpu')
	return model


def do_interval_training_run(model, num_epochs, learning_rate, batch_size, beta, data_loading_function):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to('cpu')
	model.initialize_min_max_parameters()
	model.to(device)
	print('Running on',device)

	print('Initializing optimizer..')
	min_max_parameters = ['min_weight','max_weight','min_bias','max_bias']

	optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if name.split('.')[-1] in min_max_parameters], lr=learning_rate)
	print('Optimizer initialized.')

	for epoch in range(1,num_epochs+1):
		start_time = time.time()
		
		for min_max_iteration in range(1600//batch_size):
			k = [np.random.randint(1600) for i in range(batch_size)]
			data_input, data_output = data_loading_function(k)
			
			data_input = torch.as_tensor(data_input.astype(np.float32))
			data_output = torch.as_tensor(data_output.astype(np.float32))
			
			data_input = data_input.to(device)
			data_output = data_output.to(device)

			loss = model.get_min_max_loss(data_input, data_output, beta)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			model.apply_min_max_constraints()
		
		
		stop_time = time.time()
		time_el = int(stop_time-start_time)
		print('power {}, interval epoch [{}/{}], estimation loss:{:.4f} in {}h {}m {}s'.format(8, epoch, num_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))

	model.to('cpu')
	return model

def do_probout_training_run(model, num_epochs, learning_rate, batch_size, data_loading_function):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.initialize_probout_layer()
	model.to(device)
	print('Running on',device)

	print('Initializing optimizer..')
	min_max_parameters = ['min_weight','max_weight','min_bias','max_bias']

	optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if not name.split('.')[-1] in min_max_parameters], lr=learning_rate)
	print('Optimizer initialized.')


	for epoch in range(1,num_epochs+1):
		start_time = time.time()
		
		for i in range(1600//batch_size):
			k = [np.random.randint(1600) for i in range(batch_size)]
			data_input, data_output = data_loading_function(k)
			
			data_input = torch.as_tensor(data_input.astype(np.float32))
			data_output = torch.as_tensor(data_output.astype(np.float32))
			
			data_input = data_input.to(device)
			data_output = data_output.to(device)
			
			input_img = torch.autograd.Variable(data_input)
			output_target = torch.autograd.Variable(data_output)
			
			loss = model.get_probout_loss(input_img, output_target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		stop_time = time.time()
		time_el = int(stop_time-start_time)
		print('power {}, probout epoch [{}/{}], loss:{:.4f} in {}h {}m {}s'.format(8, epoch, num_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))

	model.to('cpu')
	return model