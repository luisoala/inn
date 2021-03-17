import numpy as np
import torch
import matplotlib.pyplot as plt

from Layers import *
from torch.nn import ReLU, Conv2d, BatchNorm2d, Dropout, Upsample, MaxPool2d

class ProboutUNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		#Data size = 512x512
		network_encoder = []
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(1,64,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(64))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(64,64,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(64))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(64,64,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(64))
		
		self.block_one = torch.nn.ModuleList(network_encoder)###UNET CONNECTION SERVER
		network_encoder = []
		
		network_encoder.append(MaxPool2d(kernel_size=2, stride=2, padding=0))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(64,128,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(128))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(128,128,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(128))
		
		self.block_two = torch.nn.ModuleList(network_encoder)###UNET CONNECTION SERVER
		network_encoder = []
		
		network_encoder.append(MaxPool2d(kernel_size=2, stride=2, padding=0))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(128,256,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(256))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(256,256,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(256))
		
		self.block_three = torch.nn.ModuleList(network_encoder)###UNET CONNECTION SERVER
		network_encoder = []
		
		network_encoder.append(MaxPool2d(kernel_size=2, stride=2, padding=0))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(256,512,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(512))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(512,512,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(512))
		
		network_encoder.append(Dropout(p=0.5))
		
		self.block_four = torch.nn.ModuleList(network_encoder)###UNET CONNECTION SERVER
		network_encoder = []
		
		network_encoder.append(MaxPool2d(kernel_size=2, stride=2, padding=0))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(512,1024,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(1024))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(1024,1024,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(1024))
		
		network_encoder.append(Dropout(p=0.5))
		network_encoder.append(Upsample(scale_factor=2))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(1024,512,kernel_size=3, stride=1, padding=1))
		#network_encoder.append(Dropout(p=0.5))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(512))
		
		self.block_five = torch.nn.ModuleList(network_encoder)###UNET CONNECTION RECEIVER
		network_encoder = []
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(1024,512,kernel_size=3, stride=1, padding=1))
		#network_encoder.append(Dropout(p=0.5))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(512))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(512,512,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(512))
		
		network_encoder.append(Upsample(scale_factor=2))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(512,256,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(256))
		
		self.block_six = torch.nn.ModuleList(network_encoder)###UNET CONNECTION RECEIVER
		network_encoder = []
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(512,256,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(256))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(256,256,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(256))
		
		network_encoder.append(Upsample(scale_factor=2))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(256,128,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(BatchNorm2d(128))
		
		self.block_seven = torch.nn.ModuleList(network_encoder)###UNET CONNECTION RECEIVER
		network_encoder = []
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(256,128,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(128,128,kernel_size=3, stride=1, padding=1))
		
		network_encoder.append(Upsample(scale_factor=2))
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(128,64,kernel_size=3, stride=1, padding=1))
		
		
		self.block_eight = torch.nn.ModuleList(network_encoder)###UNET CONNECTION RECEIVER
		network_encoder = []
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(128,64,kernel_size=3, stride=1, padding=1))
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(64,64,kernel_size=3, stride=1, padding=1))
		
		
		network_encoder.append(ReLU())
		network_encoder.append(Conv2d(64,2,kernel_size=1, stride=1, padding=0))
		network_encoder.append(ReLU())
		
		self.block_nine = torch.nn.ModuleList(network_encoder)###RESNET CONNECTION RECEIVER
		network_encoder = []
		
		
		network_encoder.append(Conv2d(2,2,kernel_size=1, stride=1, padding=0))
		
		self.block_ten = torch.nn.ModuleList(network_encoder)
		
		self.block_list = [self.block_one, self.block_two, self.block_three, self.block_four, self.block_five, self.block_six, self.block_seven, self.block_eight, self.block_nine, self.block_ten]
		
	def forward(self, x):
		block_one_out = x.clone()
		for module in self.block_one:
			block_one_out = module(block_one_out)
			
		block_two_out = block_one_out.clone()
		for module in self.block_two:
			block_two_out = module(block_two_out)
			
		block_three_out = block_two_out.clone()
		for module in self.block_three:
			block_three_out = module(block_three_out)
			
		block_four_out = block_three_out.clone()
		for module in self.block_four:
			block_four_out = module(block_four_out)
			
		block_five_out = block_four_out.clone()
		for module in self.block_five:
			block_five_out = module(block_five_out)
			
		block_six_out = torch.cat((block_four_out,block_five_out),dim=1)
		for module in self.block_six:
			block_six_out = module(block_six_out)
			
		block_seven_out = torch.cat((block_three_out,block_six_out),dim=1)
		for module in self.block_seven:
			block_seven_out = module(block_seven_out)
			
		block_eight_out = torch.cat((block_two_out,block_seven_out),dim=1)
		for module in self.block_eight:
			block_eight_out = module(block_eight_out)
			
		block_nine_out = torch.cat((block_one_out,block_eight_out),dim=1)
		for module in self.block_nine:
			block_nine_out = module(block_nine_out)
			
		block_ten_out = block_nine_out+x
		for module in self.block_ten:
			block_ten_out = module(block_ten_out)

		block_ten_out_new = block_ten_out.clone()
		block_ten_out_new[:,0,:,:] = torch.nn.functional.relu(block_ten_out[:,0,:,:])

		return block_ten_out_new