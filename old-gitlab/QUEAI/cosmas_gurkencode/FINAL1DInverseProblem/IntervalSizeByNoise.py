import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pickle
from scipy.io import loadmat

from TestScript import do_test_run

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels

dir_path = 'D:/Cosmas/Desktop/NN_uncertainty/GenuineTests/1DInverseProblem'

np.random.seed(1)
noise_vector_in = [np.random.normal(scale=0.02, size=(2000,512)), np.random.normal(scale=0.05, size=(2000,512)), np.random.normal(scale=0.08, size=(2000,512)), np.random.normal(scale=0.11, size=(2000,512)), np.random.normal(scale=0.14, size=(2000,512))]
noise_vector_out = [np.random.normal(scale=0.02, size=(2000,512)), np.random.normal(scale=0.05, size=(2000,512)), np.random.normal(scale=0.08, size=(2000,512)), np.random.normal(scale=0.11, size=(2000,512)), np.random.normal(scale=0.14, size=(2000,512))]


def get_noisy_batch_pow_8(noise_level):
	def load_batch_pow_8(batch_indices):
		input_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
		output_data = np.zeros((len(batch_indices),1,512),dtype=np.float64)
		for i in range(len(batch_indices)):
			mat_dict = loadmat(dir_path+'/n_512_dist_5_jumps_60_pow_8/data_'+str(batch_indices[i]+1)+'.mat')
			output_data[i,0,:] = mat_dict['x'][:,0]+noise_vector_out[noise_level][i,:]
			input_data[i,0,:] = mat_dict['y'][:,0]+noise_vector_in[noise_level][i,:]
		return input_data, output_data
	return load_batch_pow_8



sigmas = [0.02,0.05,0.08,0.11,0.14]
interval_sizes = []
dropout_sizes = []
probout_sizes = []

for k, sigma in enumerate(sigmas):
	dropout_sample_dict, interval_sample_dict, probout_sample_dict = do_test_run(0, noisy=False, noise_index=k, new_data_loading_function=get_noisy_batch_pow_8(k))

	min_values = interval_sample_dict['min']
	max_values = interval_sample_dict['max']
	target_values = interval_sample_dict['target']

	interval_sizes.append(np.mean(max_values-min_values))
	dropout_sizes.append(np.mean(dropout_sample_dict['uncertainty']))
	probout_sizes.append(np.mean(probout_sample_dict['uncertainty']))



fig, ax1 = plt.subplots()
fig.set_size_inches(5, 3)
#fig = plt.figure(figsize=(5, 3))

ax2 = ax1.twinx()
ax1.plot(sigmas, interval_sizes, color = 'white', alpha=0.5)
ax2.plot(sigmas, interval_sizes, color = 'white', alpha=0.5)
plt.plot(sigmas, interval_sizes, color ='black', ls='--', marker='x', markersize=7, label=r'$\textsc{INN}$')
plt.plot(sigmas, dropout_sizes, color ='blue', ls='--', marker='x', markersize=7, label=r'$\textsc{MCDrop}$')
plt.plot(sigmas, probout_sizes, color ='green', ls='--', marker='x', markersize=7, label=r'$\textsc{ProbOut}$')
#ax1.plot(interval_ratios, [0.5]*len(interval_ratios), color = 'red', alpha=0.8)
#ax1.ylim((0.,1.))
#ax2.plot(interval_ratios, np.array(pixel_portions), color ='blue', ls='--', marker='o', markersize=5, alpha=0.7)
#ax2.plot(interval_ratios, pixel_portions, 'b-')
plt.xticks(sigmas)
ax1.set_xlabel(r'$\mathrm{\sigma}$', fontsize=BIGGER_SIZE) #r'$\mathrm{Iterations}$',fontsize=15
ax1.set_ylabel(r'$\mathrm{Mean\ Uncertainty\ Magnitude}$', fontsize=BIGGER_SIZE) #color='g'
ax2.set_ylabel(r'$\mathrm{Mean\ Interval\ Size}$', fontsize=BIGGER_SIZE, color='w')
#ax1.yaxis.label.set_color('red')
#ax2.yaxis.label.set_color('blue')
#ax1.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='white')
plt.legend()
plt.savefig('Interval_by_noise.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.show()