import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from ModelRadonProbout import ProboutUNet
from DataLoader import load_radon_ellipses_batch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model = ProboutUNet()

answer = input('Do you want to load the previous model? [y/n]:')
if answer == 'y':
	print('Loading model..')
	model.load_state_dict(torch.load('Model_Radon_probout.pt'))
	print('Model loaded.')

model.to(device)
print('Model Built.')

print('Initializing optimizer..')
def proboutLoss(x, target):
	x_out, x_uncert = x[:,0:1,:,:], x[:,1:2,:,:]*0.1
	return torch.mean(x_uncert)+torch.mean((x_out-target)**2/torch.exp(x_uncert))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
print('Optimizer initialized.')


print('Commencing training..')
num_epochs = 40
for epoch in range(num_epochs):
	start_time = time.time()
	
	batch_size = 1
	for i in range(2000//batch_size):
		k = [np.random.randint(1900) for i in range(batch_size)]
		data_input, data_output = load_radon_ellipses_batch(k)
		
		data_input = torch.as_tensor(data_input.astype(np.float32))
		data_output = torch.as_tensor(data_output.astype(np.float32))
		
		data_input = data_input.to(device)
		data_output = data_output.to(device)
		
		input_img = torch.autograd.Variable(data_input)
		output_target = torch.autograd.Variable(data_output)
		
		output = model(input_img)
		
		loss = proboutLoss(output, output_target)#torch.mean((output[:,0:1,:,:]-output_target)**2)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	stop_time = time.time()
	print(torch.all(output[:,0,:,:]==0))
	time_el = int(stop_time-start_time)
	torch.save(model.state_dict(), 'Model_Radon_probout_quicksave.pt')
	print('epoch [{}/{}], loss:{:.4f} in {}h {}m {}s'.format(epoch+1, num_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))
		  
print('Finished training.')

'''
print('Commencing training..')
num_epochs = 10
for epoch in range(num_epochs):
	start_time = time.time()
	
	batch_size = 1
	for i in range(2000//batch_size):
		k = [np.random.randint(1900) for i in range(batch_size)]
		data_input, data_output = load_radon_ellipses_batch(k)
		
		data_input = torch.as_tensor(data_input.astype(np.float32))
		data_output = torch.as_tensor(data_output.astype(np.float32))
		
		data_input = data_input.to(device)
		data_output = data_output.to(device)
		
		input_img = torch.autograd.Variable(data_input)
		output_target = torch.autograd.Variable(data_output)
		
		output = model(input_img)
		
		loss = proboutLoss(output, output_target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	stop_time = time.time()
	print(torch.all(output[:,0,:,:]==0))
	time_el = int(stop_time-start_time)
	torch.save(model.state_dict(), 'Model_Radon_probout_quicksave.pt')
	print('epoch [{}/{}], loss:{:.4f} in {}h {}m {}s'.format(epoch+1, num_epochs, loss.data.item(), time_el//3600, (time_el%3600)//60, time_el%60))
		  
print('Finished training.')
'''
print('Showing examples..')
for i in range(10):
	k = np.random.randint(1900,2000)
	data_input, data_output = load_radon_ellipses_batch([k])
	
	data_input_pt = torch.as_tensor(data_input.astype(np.float32)).to(device)
	
	data_input = data_input[0,0,:,:]
	data_output = data_output[0,0,:,:]
	output = model(data_input_pt).detach().cpu().numpy()[0,0,:,:]
	
	plt.figure()
	plt.subplot(2,2,1).set_title('data point')
	plt.imshow(data_output)
	
	plt.subplot(2,2,2).set_title('input')
	plt.imshow(data_input)
	
	plt.subplot(2,2,3).set_title('expected output')
	plt.imshow(data_output)
	
	plt.subplot(2,2,4).set_title('output')
	plt.imshow(output)
	
	plt.show()
	
answer = input('Do you want to save the model? [y/n]:')
if answer == 'y':
	print('Saving model..')
	torch.save(model.state_dict(), 'Model_Radon_probout.pt')
	print('Model saved.')
else:
	print('Exiting..')
