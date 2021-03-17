import matplotlib.pyplot as plt
import numpy as np
import pickle

samples = [27,28,29]
betas = [0.005]
epochs = [10,20,35,50,80,100]
num_epochs = 100
power = 8

num_plots = epochs*2
x_axis = np.arange(512)

for sample in samples:
	plt.figure(figsize=(12,9))
	for epoch_index in range(len(epochs)):
		epoch = epochs[epoch_index]
		
		plt.subplot(2*len(epochs),1,2*epoch_index+1).set_title('epoch [{}/{}]'.format(epoch, num_epochs))
		plt.ylim((-1,1))
		
		for beta in betas:
			test_dict = pickle.load(open('TestResults/sample_pow{}_epoch{}_b{}.pickle'.format(power,epoch,str(beta).split('.')[-1]),'rb'))
			plt.fill_between(x_axis,test_dict['min'][sample,:],test_dict['max'][sample,:],facecolor=[0,0,0.9,1/len(betas)],interpolate=True)
			
		plt.plot(x_axis,test_dict['target'][sample,:],color='black')
			
		for beta in betas:
			test_dict = pickle.load(open('TestResults/sample_pow{}_epoch{}_b{}.pickle'.format(power,epoch,str(beta).split('.')[-1]),'rb'))
			plt.plot(x_axis,test_dict['pred'][sample,:],color=[1,0,0,1/len(betas)])
		
		plt.subplot(2*len(epochs),1,2*epoch_index+2).set_title('epoch [{}/{}]'.format(epoch, num_epochs))
		plt.ylim((0,0.2))
		
		for beta in betas:
			test_dict = pickle.load(open('TestResults/sample_pow{}_epoch{}_b{}.pickle'.format(power,epoch,str(beta).split('.')[-1]),'rb'))
			plt.fill_between(x_axis,0*x_axis,test_dict['uncertainty'][sample,:],facecolor=[0,0,0.9,1/len(betas)],interpolate=True)
			
		for beta in betas:
			test_dict = pickle.load(open('TestResults/sample_pow{}_epoch{}_b{}.pickle'.format(power,epoch,str(beta).split('.')[-1]),'rb'))
			plt.plot(x_axis,np.abs(test_dict['pred'][sample,:]-test_dict['target'][sample,:]),color=[1,0,0,1/len(betas)])
		
	plt.show()