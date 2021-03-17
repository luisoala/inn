import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw, ImageFont
from matplotlib.cm import viridis, copper

from ModelRadonProbout import ProboutUNet
from DataLoader import load_radon_ellipses_batch, load_data_point
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model = ProboutUNet()
model.load_state_dict(torch.load('Model_Radon_probout_quicksave.pt'))
model.to(device)
print('Model Built.')

print('Showing examples..')
for i in range(10):
	k = np.random.randint(1900,2000)
	data_input, data_output = load_radon_ellipses_batch([k])
	
	data_input_pt = torch.as_tensor(data_input.astype(np.float32)).to(device)
	
	data_input = data_input[0,0,:,:]
	data_output = data_output[0,0,:,:]
	output = model(data_input_pt.clone()).detach().cpu().numpy()[0,:,:,:]
	out, var = output[0,:,:], np.exp(output[1,:,:])
	
	print(out.shape, var.shape)

	plt.figure()
	plt.subplot(3,2,1).set_title('data point')
	plt.imshow(data_output)
	
	plt.subplot(3,2,2).set_title('input')
	plt.imshow(data_input)
	
	plt.subplot(3,2,3).set_title('expected output')
	plt.imshow(data_output)
	
	plt.subplot(3,2,4).set_title('output')
	plt.imshow(out)

	plt.subplot(3,2,5).set_title('var')
	plt.imshow(var)

	
	plt.show()
	
plt.show()