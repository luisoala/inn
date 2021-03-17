import numpy as np
import torch


class IntervalConv2d(torch.nn.Conv2d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.weight.data = torch.abs(self.weight.data*0.001)
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
		previous_value_min = torch.nn.functional.conv2d(v_min,torch.clamp(self.min_weight,0,None),					**self.key_word_args)	\
							+torch.nn.functional.conv2d(v_max,torch.clamp(self.min_weight,None,0),					**self.key_word_args)	\
							+self.min_bias[None,:,None,None]
		
		previous_value_max = torch.nn.functional.conv2d(v_min,torch.clamp(self.max_weight,None,0),					**self.key_word_args)	\
							+torch.nn.functional.conv2d(v_max,torch.clamp(self.max_weight,0,None),					**self.key_word_args)	\
							+self.max_bias[None,:,None,None]
		
		previous_value_max = torch.max(previous_value_max,previous_value_min)
		
		return previous_value_min,previous_value_max
		
	def forward_min_max_test(self, v_min, v_max, v):
		real_value = self.forward(v)
		
		print(torch.min(v_max-v))
		print(torch.min(v-v_min))
		print('====')
		previous_value_min = torch.nn.functional.conv2d(v_min,torch.clamp(self.min_weight,0,None),					**self.key_word_args)	\
							+torch.nn.functional.conv2d(v_max,torch.clamp(self.min_weight,None,0),					**self.key_word_args)	\
							+self.min_bias[None,:,None,None]
		
		previous_value_max = torch.nn.functional.conv2d(v_min,torch.clamp(self.max_weight,None,0),					**self.key_word_args)	\
							+torch.nn.functional.conv2d(v_max,torch.clamp(self.max_weight,0,None),					**self.key_word_args)	\
							+self.max_bias[None,:,None,None]
							
		print(torch.min(previous_value_max-real_value))
		print(torch.min(real_value-previous_value_min))
		print('#########')
		return previous_value_min,previous_value_max
	
	def apply_min_max_constraints(self):
		self.min_weight.data = torch.min(self.min_weight.data,self.weight.data)
		self.max_weight.data = torch.max(self.max_weight.data,self.weight.data)
		self.min_bias.data = torch.min(self.min_bias.data,self.bias.data)
		self.max_bias.data = torch.max(self.max_bias.data,self.bias.data)
		
	def initialize_parameters(self, v, eps=1e-5):
		old_v = v.clone()
		out_data = self.forward(v)
		out_mean = out_data.mean(0).mean(1).mean(1)#torch.mean(out_data,(0,2,3))
		self.bias.data -= out_mean
		
		out_data -= out_mean[None,:,None,None]
		out_std = torch.sqrt((out_data**2).mean(0).mean(1).mean(1))
		
		self.weight.data /= out_std[:,None,None,None]+eps

		return self.forward(old_v)
		
		
class IntervalReLU(torch.nn.ReLU):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward_min_max(self, v_min, v_max):
		#print(torch.max(v_max-v_min),type(self))
		return self.forward(v_min), self.forward(v_max)
			
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
	def initialize_parameters(self, v, eps=1e-5):
		return self.forward(v)
		
class IntervalSigmoid(torch.nn.Sigmoid):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward_min_max(self, v_min, v_max):
		return self.forward(v_min), self.forward(v_max)
			
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
	def initialize_parameters(self, v, eps=1e-5):
		return self.forward(v)
		
class IntervalMaxPool2d(torch.nn.MaxPool2d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward_min_max(self, v_min, v_max):
		#print(torch.max(v_max-v_min),type(self))
		return self.forward(v_min), self.forward(v_max)
		
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
	def initialize_parameters(self, v, eps=1e-5):
		return self.forward(v)
		
class IntervalUpsample(torch.nn.Module):
	def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
		if mode != 'nearest':
			raise Exception('Interpolation is not supported for intervals')
		super().__init__()
		self.size=size
		self.scale_factor = scale_factor
		self.mode = mode
		self.align_corners = align_corners
	
	def forward(self, v):
		return torch.nn.functional.interpolate(v,size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
	
	def forward_min_max(self, v_min, v_max):
		#print(torch.max(v_max-v_min),type(self))
		return self.forward(v_min), self.forward(v_max)
		
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
	def initialize_parameters(self, v, eps=1e-5):
		return self.forward(v)
		
class IntervalDropout2d(torch.nn.Module):
	def __init__(self, p=0.5):
		super().__init__()
		self.p = p
		self.mask = None
		
	def forward(self, v):
		self.mask = torch.autograd.Variable(torch.bernoulli(torch.full_like(v,self.p)))/(1-self.p)
		return v*self.mask
		
	def forward_min_max(self, v_min, v_max):
		#print(torch.max(v_max-v_min),type(self))
		#mask = torch.autograd.Variable(torch.bernoulli(torch.full_like(v_min,self.p)))/(1-self.p)
		return v_min*self.mask, v_max*self.mask
	
	def apply_min_max_constraints(self):
		pass
		
	def initialize_min_max_parameters(self):
		pass
		
	def initialize_parameters(self, v, eps=1e-5):
		return self.forward(v)
		
class IntervalBatchNorm2d(torch.nn.Module):
	def __init__(self, batch_size, eps=1e-5):
		super().__init__()
		self.eps = eps
		
		self.weight = torch.nn.Parameter(torch.rand(batch_size),requires_grad=True)
		self.bias = torch.nn.Parameter(torch.ones(batch_size),requires_grad=True)
	
	def forward(self, v):
		###It is assumed that v>0
		v = v-v.mean(0).mean(1).mean(1)[None,:,None,None]
		v = v*(1/torch.sqrt((v**2).mean(0).mean(1).mean(1)+self.eps))[None,:,None,None]
		return v*self.weight[None,:,None,None]+self.bias[None,:,None,None]
	
	def forward_min_max_test(self, v_min, v_max, v):
		real_centerd_v = v-v.mean(0).mean(1).mean(1)[None,:,None,None]
		real_standart_deviation = torch.sqrt((real_centerd_v**2).mean(0).mean(1).mean(1)+self.eps)
		real_value = real_centerd_v*(1/real_standart_deviation)[None,:,None,None]
		real_value = real_value*self.weight[None,:,None,None]+self.bias[None,:,None,None]
		###It is assumed that v_min>0
		#approximation (could be abgeschaetzt more strongly)
		max_centers = v_max.mean(0).mean(1).mean(1)
		min_centers = v_min.mean(0).mean(1).mean(1)
		
		v_min_centered = v_min-max_centers[None,:,None,None]
		v_max_centered = v_max-min_centers[None,:,None,None]
		
		print(torch.min(v_max-v_min))
		print(torch.min(v_max-v))
		print(torch.min(v-v_min))
		print('--------')
		
		print(torch.min(v_max_centered-v_min_centered))
		print(torch.min(v_max_centered-real_centerd_v))
		print(torch.min(real_centerd_v-v_min_centered))
		print('--------')
		standart_deviation_min = torch.sqrt(((torch.clamp(v_min_centered,0,None))**2+(torch.clamp(v_max_centered,None,0))**2).mean(0).mean(1).mean(1)+self.eps)
		standart_deviation_max = torch.sqrt((torch.max((v_min_centered)**2,(v_max_centered)**2)).mean(0).mean(1).mean(1)+self.eps)
		print(torch.min(standart_deviation_max-standart_deviation_min))
		print(torch.min(standart_deviation_max-real_standart_deviation))
		print(torch.min(real_standart_deviation-standart_deviation_min))
		print('#######')
		print('')
		
		effective_weight_min = self.min_weight/standart_deviation_max
		effective_weight_max = self.max_weight/standart_deviation_min
		
		out_min = v_min*torch.clamp(effective_weight_min,0,None)[None,:,None,None]+v_max*torch.clamp(effective_weight_min,None,0)[None,:,None,None]
		out_max = v_max*torch.clamp(effective_weight_max,0,None)[None,:,None,None]+v_min*torch.clamp(effective_weight_max,None,0)[None,:,None,None]
		
		effective_bias_min = -max_centers*torch.clamp(effective_weight_max,0,None)-min_centers*torch.clamp(effective_weight_max,None,0)+self.min_bias
		effective_bias_max = -min_centers*torch.clamp(effective_weight_min,0,None)-max_centers*torch.clamp(effective_weight_min,None,0)+self.max_bias
		return out_min+effective_bias_min[None,:,None,None], out_max+effective_bias_max[None,:,None,None]
	
	def forward_min_max(self, v_min, v_max):
		###It is assumed that v_min>0
		#approximation (could be abgeschaetzt more strongly)
		max_centers = v_max.mean(0).mean(1).mean(1)
		min_centers = v_min.mean(0).mean(1).mean(1)
		
		v_min_centered = v_min-max_centers[None,:,None,None]
		v_max_centered = v_max-min_centers[None,:,None,None]
		
		standart_deviation_min = torch.sqrt(((torch.clamp(v_min_centered,0,None))**2+(torch.clamp(v_max_centered,None,0))**2).mean(0).mean(1).mean(1)+self.eps)
		standart_deviation_max = torch.sqrt((torch.max((v_min_centered)**2,(v_max_centered)**2)).mean(0).mean(1).mean(1)+self.eps)
		
		effective_weight_min = self.min_weight/standart_deviation_max
		effective_weight_max = self.max_weight/standart_deviation_min
		
		out_min = v_min*torch.clamp(effective_weight_min,0,None)[None,:,None,None]+v_max*torch.clamp(effective_weight_min,None,0)[None,:,None,None]
		out_max = v_max*torch.clamp(effective_weight_max,0,None)[None,:,None,None]+v_min*torch.clamp(effective_weight_max,None,0)[None,:,None,None]
		
		effective_bias_min = -max_centers*torch.clamp(effective_weight_max,0,None)-min_centers*torch.clamp(effective_weight_max,None,0)+self.min_bias
		effective_bias_max = -min_centers*torch.clamp(effective_weight_min,0,None)-max_centers*torch.clamp(effective_weight_min,None,0)+self.max_bias
		return out_min+effective_bias_min[None,:,None,None], out_max+effective_bias_max[None,:,None,None]
		
		'''
		#print(torch.max(v_max-v_min),type(self))
		effective_weight_min = self.min_weight/self.standart_deviation
		effective_weight_max = self.max_weight/self.standart_deviation
		scaled_mean = (self.mean/self.standart_deviation)
		effective_bias_min = -scaled_mean*self.max_weight+self.min_bias
		effective_bias_max = -scaled_mean*self.min_weight+self.max_bias
		out_min = v_min*torch.clamp(effective_weight_min,0,None)[None,:,None,None]+v_max*torch.clamp(effective_weight_min,None,0)[None,:,None,None]
		out_max = v_max*torch.clamp(effective_weight_max,0,None)[None,:,None,None]+v_min*torch.clamp(effective_weight_max,None,0)[None,:,None,None]
		return out_min+effective_bias_min[None,:,None,None], out_max+effective_bias_max[None,:,None,None]
		'''
		
	def apply_min_max_constraints(self):
		self.min_weight.data = torch.min(self.min_weight.data,self.weight.data)
		self.max_weight.data = torch.max(self.max_weight.data,self.weight.data)
		self.min_bias.data = torch.min(self.min_bias.data,self.bias.data)
		self.max_bias.data = torch.max(self.max_bias.data,self.bias.data)
		
	def initialize_min_max_parameters(self):
		self.min_weight = torch.nn.Parameter(self.weight.data.clone(), requires_grad=True)
		self.max_weight = torch.nn.Parameter(self.weight.data.clone(), requires_grad=True)
		self.min_bias = torch.nn.Parameter(self.bias.data.clone(), requires_grad=True)
		self.max_bias = torch.nn.Parameter(self.bias.data.clone(), requires_grad=True)
		
		self.min_max_parameters = torch.nn.ParameterDict({'min_weight': self.min_weight,
														  'max_weight': self.max_weight,
														  'min_bias': self.min_bias,
														  'max_bias': self.max_bias})
		
	def initialize_parameters(self, v, eps=1e-5):
		#intelligent initialization is meant
		raise Exception('Weight and bias initialization not implemented for Interval Batch Normalization')