import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from Model import IntervalConvNet
from Util import load_batch_pow_2, load_batch_pow_4

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


snapshot_epochs = [19,39,59,79,99]
betas = [0.0002,0.001,0.005]
num_test_samples = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)

print('Building model..')	
model = IntervalConvNet()

def get_interval_accuracy(model):
	k = [1800+ind for ind in range(200)]
	data_input, data_output = load_batch_pow_4(k)
	
	data_input = torch.as_tensor(data_input.astype(np.float32))
	
	data_input = data_input.to(device)

	output_min, output_max = model.forward_min_max(data_input)
	output_min = output_min.detach().cpu().numpy()[:,0,:]
	output_max = output_max.detach().cpu().numpy()[:,0,:]
	
	percentage = np.sum((data_output[:,0,:]<=output_max)*(data_output[:,0,:]>=output_min),axis=0)/200

	return percentage
	
def run_test(model, data_index):
	data_input, data_output = load_batch_pow_4([data_index])
	
	data_input_pt = torch.as_tensor(data_input.astype(np.float32)).to(device)
	
	data_input = data_input[0,0,:]
	data_output = data_output[0,0,:]
	output = model(data_input_pt).detach().cpu().numpy()[0,0,:]
	
	output_min, output_max = model.forward_min_max(data_input_pt)
	output_min = output_min.detach().cpu().numpy()[0,0,:]
	output_max = output_max.detach().cpu().numpy()[0,0,:]
	
	test_dict = {'data_input':data_input,'target':data_output,'model_output':output,'model_min_output':output_min,'model_max_output':output_max}
	
	return test_dict
	

for i in range(num_test_samples):
	data_index = np.random.randint(1800,2000)
	x = np.arange(512)
	
	plt.figure(figsize=(12,9))
	
	data_input, data_output = load_batch_pow_4([data_index])
	plt.subplot(len(snapshot_epochs)+1,1,1).set_title('Data point')
	plt.ylim((-1,1))
	plt.plot(x,data_output[0,0,:],color='black')
	plt.plot(x,data_input[0,0,:],color='orange')
	
	for ind in range(len(snapshot_epochs)):
		snapshot_epoch = snapshot_epochs[ind]
		print('Sample [{}/{}]: performing tests for epoch {}'.format(i+1,num_test_samples,snapshot_epoch+1))
		
		plt.subplot(len(snapshot_epochs)+1,1,ind+2).set_title('epoch: {}'.format(snapshot_epoch+1))
		plt.ylim((-1,1))
		for beta in betas:
			model.to('cpu')
			model.load_state_dict(torch.load(dir_path+'/ModelSnapshots/pow4_epoch{}_b{}.pt'.format(snapshot_epoch+1,str(beta).split('.')[-1])))
			model.to(device)
			
			interval_accuracy = np.sum(get_interval_accuracy(model))/512
			test_results = run_test(model,data_index)
			
			plt.fill_between(x,test_results['model_min_output'],test_results['model_max_output'],facecolor=[0,0,1,1-interval_accuracy*0.85],interpolate=True,label='{:.2f}%'.format(interval_accuracy*100))
			
		plt.plot(x,test_results['target'],color='black')
		plt.plot(x,test_results['model_output'],color='red')
		plt.legend(bbox_to_anchor=(1,1))
	plt.show()