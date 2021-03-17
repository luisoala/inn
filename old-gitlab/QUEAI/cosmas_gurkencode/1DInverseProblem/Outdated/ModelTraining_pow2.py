import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from Model import IntervalConvNet
from Util import load_batch_pow_2, load_batch_pow_4

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model = IntervalConvNet()
model.to(device)
print('Model Built.')

print('Initializing optimizer..')
min_max_parameters = ['min_weight','max_weight','min_bias','max_bias']

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if not name.split('.')[-1] in min_max_parameters], lr=1e-3)
min_max_optimizer = torch.optim.Adam([param for name,param in model.named_parameters() if name.split('.')[-1] in min_max_parameters], lr=2e-5)
print('Optimizer initialized.')


print('Commencing training..')
num_epochs = 100
snapshot_epochs = [19,39,59,79,99]
betas = [0.0002,0.001,0.005]

def train_epoch():
	batch_size = 200
	for i in range(2000//batch_size):
		k = [np.random.randint(1600) for i in range(batch_size)]
		data_input, data_output = load_batch_pow_2(k)
		
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
	return loss
		
def get_validation_loss():
	batch_size = 200
	k = [i+1600 for i in range(batch_size)]
	data_input, data_output = load_batch_pow_2(k)
	
	data_input = torch.as_tensor(data_input.astype(np.float32))
	data_output = torch.as_tensor(data_output.astype(np.float32))
	
	data_input = data_input.to(device)
	data_output = data_output.to(device)
	
	output = model(data_input)
	
	loss = criterion(output, data_output)
	
	return loss

def train_min_max_epoch(beta):
	batch_size = 100
	for i in range(2000//batch_size):
		k = [np.random.randint(1600) for i in range(batch_size)]
		data_input, data_output = load_batch_pow_2(k)
		
		data_input = torch.as_tensor(data_input.astype(np.float32))
		data_output = torch.as_tensor(data_output.astype(np.float32))
		
		data_input = data_input.to(device)
		data_output = data_output.to(device)

		loss = model.get_min_max_loss(data_input, data_output, beta)

		min_max_optimizer.zero_grad()
		loss.backward()
		min_max_optimizer.step()
		
		model.apply_min_max_constraints()
	
	return loss
	
for epoch in range(num_epochs):
	start_time = time.time()
	loss = train_epoch()
	validation_loss = get_validation_loss()
	stop_time = time.time()
	time_el = int(stop_time-start_time)
	print('epoch [{}/{}], loss:{:.4f}, validation loss:{:.4f} in {}h {}m {}s'.format(epoch+1, num_epochs, loss.data.item(),validation_loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))
	
	if epoch in snapshot_epochs:
		print('Making Snapshot..')
		for beta in betas:
			print('Commencing interval training for beta =',beta)
			model.to('cpu')
			model.initialize_min_max_parameters()
			model.to(device)
			num_min_max_epochs=30
			for min_max_epoch in range(num_min_max_epochs):
				start_time = time.time()
				loss = train_min_max_epoch(beta)
				stop_time = time.time()
				time_el = int(stop_time-start_time)
				print('epoch [{}/{}], estimation loss:{:.4f} in {}h {}m {}s'.format(min_max_epoch+1, num_min_max_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))
		
			model.to('cpu')
			torch.save(model.state_dict(), dir_path+'/ModelSnapshots/pow2_epoch{}_b{}.pt'.format(epoch+1,str(beta).split('.')[-1]))
			model.to(device)
		
print('Finished training.')
