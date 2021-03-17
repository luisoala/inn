import numpy as np
import torch
import matplotlib.pyplot as plt

from Layers import *

class IntervalConvNet(torch.nn.Module):
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
		network_list.append(IntervalConv1d(256,64,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(64,32,kernel_size=5, stride=1, padding=2))
		network_list.append(IntervalReLU())
		network_list.append(IntervalConv1d(32,1,kernel_size=1, stride=1, padding=0))
		
		self.network = torch.nn.ModuleList(network_list)
		
	def forward(self, x):
		data = x
		for module in self.network:
			data = module(data)
		return data
			
	def forward_min_max(self, x):
		data_min, data_max = x, x
		for module in self.network:
			data_min, data_max = module.forward_min_max(data_min, data_max)
		return data_min, data_max
		
	def initialize_min_max_parameters(self):
		for module in self.network:
			module.initialize_min_max_parameters()
		
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
		
	def forward_normal_and_min_max(self, x):
		input_img = torch.autograd.Variable(x, requires_grad=False)
		normal_output = self.forward(input_img)
		data_min, data_max = x, x
		for module in self.network:
			data_min, data_max = module.forward_min_max(data_min, data_max)
		return normal_output, data_min, data_max
	