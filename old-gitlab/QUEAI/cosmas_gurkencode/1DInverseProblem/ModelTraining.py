import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from Model import IntervalConvNet
from Util import load_batch_pow_2, load_batch_pow_4, load_batch_pow_8

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


snapshot_epochs = [10,20,35,50,80,100]
betas = [0.0002,0.001,0.005]
num_epochs = 100
learning_rate = 1e-3
batch_size = 256

num_min_max_epochs = 50


for power in [2,4,8]:
	if power==2:data_loading_function = load_batch_pow_2
	if power==4:data_loading_function = load_batch_pow_4
	if power==8:data_loading_function = load_batch_pow_8
	
	seed = 1
	np.random.seed(seed)
	torch.manual_seed(seed)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('Running on',device)
	print('Building model..')	
	model = IntervalConvNet()
	model.to(device)
	print('Model Built.')
	
	print('Initializing optimizer..')
	min_max_parameters = ['min_weight','max_weight','min_bias','max_bias']

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if not name.split('.')[-1] in min_max_parameters], lr=learning_rate)
	min_max_optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if name.split('.')[-1] in min_max_parameters], lr=1e-5)
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
		print('power {}, epoch [{}/{}], loss:{:.4f} in {}h {}m {}s'.format(power, epoch, num_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))
		
		if epoch in snapshot_epochs:
			for beta in betas:
				print('Commencing interval training for beta =',beta)
				model.to('cpu')
				model.initialize_min_max_parameters()
				model.to(device)
		
				for min_max_epoch in range(1,num_min_max_epochs+1):
					start_time = time.time()
					
					for min_max_iteration in range(1600//batch_size):
						k = [np.random.randint(1600) for i in range(batch_size)]
						data_input, data_output = data_loading_function(k)
						
						data_input = torch.as_tensor(data_input.astype(np.float32))
						data_output = torch.as_tensor(data_output.astype(np.float32))
						
						data_input = data_input.to(device)
						data_output = data_output.to(device)

						loss = model.get_min_max_loss(data_input, data_output, beta)

						min_max_optimizer.zero_grad()
						loss.backward()
						min_max_optimizer.step()
						
						model.apply_min_max_constraints()
					
					
					stop_time = time.time()
					time_el = int(stop_time-start_time)
					print('power {}, epoch [{}/{}],min max epoch [{}/{}], estimation loss:{:.4f} in {}h {}m {}s'.format(power, epoch, num_epochs, min_max_epoch, num_min_max_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))
			
				model.to('cpu')
				torch.save(model.state_dict(), dir_path+'/ModelSnapshots/pow{}_epoch{}_b{}.pt'.format(power,epoch,str(beta).split('.')[-1]))
				model.to(device)