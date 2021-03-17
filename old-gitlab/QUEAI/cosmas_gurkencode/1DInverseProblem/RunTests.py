import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pickle

from Model import IntervalConvNet
from Util import load_batch_pow_2, load_batch_pow_4, load_batch_pow_8

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


snapshot_epochs = [10,20,35,50,80,100]
betas = [0.0002,0.001,0.005]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)

print('Building model..')	
model = IntervalConvNet()

for power in [2,4,8]:

	if power==2:data_loading_function = load_batch_pow_2
	if power==4:data_loading_function = load_batch_pow_4
	if power==8:data_loading_function = load_batch_pow_8
	
	for epoch in snapshot_epochs:
		for beta in betas:
			print('Commencing test for power {} for epoch {} for beta {}'.format(power, epoch, beta))
			sample_dict = {}
		
			model.to('cpu')
			model.load_state_dict(torch.load(dir_path+'/ModelSnapshots/pow{}_epoch{}_b{}.pt'.format(power,epoch,str(beta).split('.')[-1])))
			model.to(device)
			
			k = [1800+ind for ind in range(200)]
			data_input, data_output = data_loading_function(k)

			data_input = torch.as_tensor(data_input.astype(np.float32))

			data_input = data_input.to(device)

			output, output_min, output_max = model.forward_normal_and_min_max(data_input)
			output = output.detach().cpu().numpy()[:,0,:]
			output_min = output_min.detach().cpu().numpy()[:,0,:]
			output_max = output_max.detach().cpu().numpy()[:,0,:]
			
			sample_dict['pred'] = output
			sample_dict['uncertainty'] = np.maximum(output_max-output_min,0)
			sample_dict['min'] = output_min
			sample_dict['max'] = output_max
			sample_dict['target'] = data_output[:,0,:]
			print('Saving..')
			pickle.dump(sample_dict, open('TestResults/sample_pow{}_epoch{}_b{}.pickle'.format(power,epoch,str(beta).split('.')[-1]),'wb'))
			print('Saved')