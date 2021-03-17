import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.nn import Conv1d, ReLU, Dropout
from Layers import *

class TestConvNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		#Data size = 512
		network_list = []
		network_list.append(IntervalConv1d(1,16,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(16,32,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(32,64,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(64,128,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(128,256,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalDropout1d(p=0.2))
		network_list.append(IntervalConv1d(256,256,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalDropout1d(p=0.5))
		network_list.append(IntervalConv1d(256,256,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalDropout1d(p=0.5))
		network_list.append(IntervalConv1d(256,64,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(64,32,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(32,2,kernel_size=1, stride=1, padding=0))
		
		self.network = torch.nn.ModuleList(network_list)
		
	def forward(self, x):
		data = x
		for module in self.network:
			data = module(data)
		return data[:,0:1,:]

	def test_forward(self, x):
		self.train(False)
		out = self.forward(x)
		self.train(True)
		return out[:,0:1,:]
			
	def forward_min_max(self, x):
		data_min, data_max = x, x
		for module in self.network:
			data_min, data_max = module.forward_min_max(data_min, data_max)
		return data_min[:,0:1,:], data_max[:,0:1,:]
		
	def initialize_min_max_parameters(self):
		for module in self.network:
			module.initialize_min_max_parameters()

	def initialize_probout_layer(self):
		last_layer = self.network[-1]
		last_layer.bias.data[1] *= 0
		last_layer.weight.data[1,...] *= 0.1
		
	def apply_min_max_constraints(self):
		for module in self.network:
			module.apply_min_max_constraints()
	
	def get_min_max_loss(self, v, label, beta):
		input_img = torch.autograd.Variable(v, requires_grad=False)
		self.forward(input_img)
		out_min, out_max = self.forward_min_max(input_img)
		loss_min = ((out_min-label).masked_fill_(label>out_min,0))**2
		loss_max = ((out_max-label).masked_fill_(label<out_max,0))**2
		loss = torch.sum(loss_min+loss_max+beta*torch.clamp(out_max-out_min,0,10))/512
		return loss
		
	def get_probout_loss(self, x, target):
		self.train(False)
		for module in self.network:
			x = module(x)
		self.train(True)
		x_out, x_uncert = x[:,0:1,:], x[:,1:2,:]
		return torch.mean(x_uncert)+torch.mean((x_out-target)**2/torch.exp(x_uncert))

	def forward_normal_and_min_max(self, x):
		normal_output = self.test_forward(x.clone())
		self.train(False)
		data_min, data_max = x.clone(), x.clone()
		for module in self.network:
			data_min, data_max = module.forward_min_max(data_min, data_max)
		self.train(True)
		return normal_output, data_min[:,0:1,:], data_max[:,0:1,:]

	def compute_dropout_std_uncertainty(self, x, num_samples = 50):
		outputs = []
		for i in range(num_samples):
			outputs.append(self.forward(x).detach().cpu()[:,0:1,:])
		mean_out = sum(outputs)/len(outputs)
		std = torch.sqrt(sum([(out-mean_out)**2 for out in outputs])/(len(outputs)-1))
		return std

	def get_probout_expstd_uncertainty(self, x):
		self.train(False)
		for module in self.network:
			x = module(x)
		self.train(True)
		return torch.exp(0.5*x[:,1:2,:])