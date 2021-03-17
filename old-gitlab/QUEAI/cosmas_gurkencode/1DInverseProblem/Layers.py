import numpy as np
import torch


class IntervalConv1d(torch.nn.Conv1d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.key_word_args = kwargs
		self.key_word_args.pop('kernel_size',None)

		self.min_weight = torch.nn.Parameter(self.weight.data.clone(), requires_grad=True)
		self.max_weight = torch.nn.Parameter(self.weight.data.clone(), requires_grad=True)
		self.min_bias = torch.nn.Parameter(self.bias.data.clone(), requires_grad=True)
		self.max_bias = torch.nn.Parameter(self.bias.data.clone(), requires_grad=True)
		
		self.min_max_parameters = torch.nn.ParameterDict({'min_weight': self.min_weight,
														  'max_weight': self.max_weight,
														  'min_bias': self.min_bias,
														  'max_bias': self.max_bias})
	
	def initialize_min_max_parameters(self):
		self.min_weight.data = self.weight.data
		self.max_weight.data = self.weight.data
		self.min_bias.data = self.bias.data
		self.max_bias.data = self.bias.data
	
	def forward_min_max(self, v_min, v_max):
		###it is assumed that v_min>0
		previous_value_min = torch.nn.functional.conv1d(v_min,torch.clamp(self.min_weight,0,None),					**self.key_word_args)	\
							+torch.nn.functional.conv1d(v_max,torch.clamp(self.min_weight,None,0),					**self.key_word_args)	\
							+self.min_bias[None,:,None]
		
		previous_value_max = torch.nn.functional.conv1d(v_min,torch.clamp(self.max_weight,None,0),					**self.key_word_args)	\
							+torch.nn.functional.conv1d(v_max,torch.clamp(self.max_weight,0,None),					**self.key_word_args)	\
							+self.max_bias[None,:,None]
		
		previous_value_max = torch.max(previous_value_max,previous_value_min)
		
		return previous_value_min,previous_value_max
	
	def apply_min_max_constraints(self):
		self.min_weight.data = torch.min(self.min_weight.data,self.weight.data)
		self.max_weight.data = torch.max(self.max_weight.data,self.weight.data)
		self.min_bias.data = torch.min(self.min_bias.data,self.bias.data)
		self.max_bias.data = torch.max(self.max_bias.data,self.bias.data)
		
class IntervalReLU(torch.nn.ReLU):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward_min_max(self, v_min, v_max):
		return self.forward(v_min), self.forward(v_max)
			
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
class IntervalMaxPool1d(torch.nn.MaxPool1d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward_min_max(self, v_min, v_max):
		return self.forward(v_min), self.forward(v_max)
		
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
class IntervalDropout1d(torch.nn.Module):
	def __init__(self, p=0.5):
		super().__init__()
		self.p = p
		self.mask = None
		
	def forward(self, v):
		self.mask = torch.autograd.Variable(torch.bernoulli(torch.full_like(v,1-self.p)))
		return v*self.mask
		
	def forward_min_max(self, v_min, v_max):
		#print(torch.max(v_max-v_min),type(self))
		#mask = torch.autograd.Variable(torch.bernoulli(torch.full_like(v_min,self.p)))/(1-self.p)
		return v_min*self.mask, v_max*self.mask
	
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass