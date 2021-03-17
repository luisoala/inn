import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pickle

from TestScript import do_test_run

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 43

plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels

x_axis = np.arange(512)

def gauss_kurve(x):
	return np.exp(-0.5*x**2)

def search_gauss_value(y, max_iter=10):
	assert 0<y and y<1
	a = -4
	while gauss_kurve(a)>y:
		a *= 2
	b = 0
	for i in range(max_iter):
		c = 0.5*(a+b)
		if gauss_kurve(c)< y:
			a = c
		else:
			b = c
	return np.abs(c)


def fill_gaussian(x_axis, mu_values, sigma_values, num_samples=20):
	distances = [search_gauss_value(i/num_samples) for i in range(1,num_samples)]
	whole_value = 0
	for dist in distances:
		plt.fill_between(x_axis, mu_values-sigma_values*dist, mu_values+sigma_values*dist,facecolor=[0,0,0.9,(gauss_kurve(dist)-whole_value)*0.9],interpolate=True)
		whole_value += gauss_kurve(dist)-whole_value

def fill_abs_gaussian(x_axis, mu_values, sigma_values, num_samples=20):
	distances = [search_gauss_value(i/num_samples) for i in range(1,num_samples)]
	whole_value = 0
	for dist in distances:
		plt.fill_between(x_axis, 0*x_axis, mu_values+sigma_values*dist,facecolor=[0,0,0.9,(gauss_kurve(dist)-whole_value)*0.9],interpolate=True)
		whole_value += gauss_kurve(dist)-whole_value


def make_figures(noisy=False):
	noisy_str = 'Noisy' if noisy else ''
	y_max = 0.6 if noisy else 0.6
	x_min, x_max = (256,512) if noisy else (0,256)

	fig_size = (16,6)
	#matplotlib.rc('xtick', labelsize=40) 
	#matplotlib.rc('ytick', labelsize=40)

	sample = 111
	dropout_sample_dict, interval_sample_dict, probout_sample_dict = do_test_run(0, noisy=noisy)

	test_dict = interval_sample_dict
	plt.figure(figsize=fig_size)
	plt.ylim((-1.2,1.2))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	plt.plot(x_axis,test_dict['target'][sample,:],color='black', linewidth=3)
	plt.plot(x_axis,test_dict['input'][sample,:],color='green', linewidth=3)
	plt.savefig('Figures/interval'+noisy_str+'_0.png')

	plt.figure(figsize=fig_size)
	plt.ylim((-1.2,1.2))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	plt.fill_between(x_axis,test_dict['min'][sample,:],test_dict['max'][sample,:],facecolor=[0,0,1,0.5],interpolate=True)
		
	plt.plot(x_axis,test_dict['target'][sample,:],color='black', linewidth=3)
		
	plt.plot(x_axis,test_dict['pred'][sample,:],color=[0.9290, 0.6940, 0.1250], linewidth=3)
	plt.savefig('Figures/interval'+noisy_str+'_1.png')

	plt.figure(figsize=fig_size)
	plt.ylim((0,y_max))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	plt.fill_between(x_axis,0*x_axis,test_dict['uncertainty'][sample,:],facecolor=[0,0,1,0.5],interpolate=True)

	plt.plot(x_axis,np.abs(test_dict['pred'][sample,:]-test_dict['target'][sample,:]),color='maroon', linewidth=3)
	plt.savefig('Figures/interval'+noisy_str+'_2.png')
	plt.close()



	test_dict = dropout_sample_dict
	plt.figure(figsize=fig_size)
	plt.ylim((-1.2,1.2))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	plt.plot(x_axis,test_dict['target'][sample,:],color='black', linewidth=3)
	plt.plot(x_axis,test_dict['input'][sample,:],color='green', linewidth=3)
	plt.savefig('Figures/dropout'+noisy_str+'_0.png')

	plt.figure(figsize=fig_size)
	plt.ylim((-1.2,1.2))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	fill_gaussian(x_axis, test_dict['pred'][sample,:], test_dict['uncertainty'][sample,:], num_samples = 20)
		
	plt.plot(x_axis,test_dict['target'][sample,:],color='black', linewidth=3)
		
	plt.plot(x_axis,test_dict['pred'][sample,:],color=[0.9290, 0.6940, 0.1250], linewidth=3)
	plt.savefig('Figures/dropout'+noisy_str+'_1.png')

	plt.figure(figsize=fig_size)
	plt.ylim((0,y_max))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	fill_abs_gaussian(x_axis, 0*x_axis, test_dict['uncertainty'][sample,:])

	plt.plot(x_axis,np.abs(test_dict['pred'][sample,:]-test_dict['target'][sample,:]),color='maroon', linewidth=3)
	plt.savefig('Figures/dropout'+noisy_str+'_2.png')
	plt.close()


	test_dict = probout_sample_dict
	plt.figure(figsize=fig_size)
	plt.ylim((-1.2,1.2))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	plt.plot(x_axis,test_dict['target'][sample,:],color='black', linewidth=3)
	plt.plot(x_axis,test_dict['input'][sample,:],color='green', linewidth=3)
	plt.savefig('Figures/probout'+noisy_str+'_0.png')

	plt.figure(figsize=fig_size)
	plt.ylim((-1.2,1.2))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	fill_gaussian(x_axis, test_dict['pred'][sample,:], test_dict['uncertainty'][sample,:], num_samples = 20)
		
	plt.plot(x_axis,test_dict['target'][sample,:],color='black')
		
	plt.plot(x_axis,test_dict['pred'][sample,:],color=[0.9290, 0.6940, 0.1250], linewidth=3)
	plt.savefig('Figures/probout'+noisy_str+'_1.png')

	plt.figure(figsize=fig_size)
	plt.ylim((0,y_max))
	plt.xlim((x_min, x_max))
	if noisy: plt.yticks([])
	fill_abs_gaussian(x_axis, 0*x_axis, test_dict['uncertainty'][sample,:])

	plt.plot(x_axis,np.abs(test_dict['pred'][sample,:]-test_dict['target'][sample,:]),color='maroon', linewidth=3)
	plt.savefig('Figures/probout'+noisy_str+'_2.png')
	plt.close()

	figlegend = plt.figure(figsize=(32,6))
	handles = []
	handles.append(mpatches.Patch(color='green', label=r'$\mathrm{Input}$'))
	handles.append(mpatches.Patch(color=[0,0,1,0.5], label=r'$\mathrm{Uncertainty}$'))
	handles.append(mpatches.Patch(color='black', label=r'$\mathrm{Target}$'))
	handles.append(mpatches.Patch(color=[0.9290, 0.6940, 0.1250], label=r'$\mathrm{Model\ Output}$'))
	handles.append(mpatches.Patch(color='maroon', label=r'$\mathrm{Absolute\ Error}$'))

	figlegend.legend(handles = handles, fontsize=50, mode='expand', ncol=5)
	plt.savefig('Figures/legend'+noisy_str+'.png')
	plt.close()


make_figures(noisy=False)
make_figures(noisy=True)