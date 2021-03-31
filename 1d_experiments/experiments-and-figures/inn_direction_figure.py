import numpy as np
import matplotlib.pyplot as plt
import sys


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels

'''
specify data point, method and path
'''

data_path = 'data/run00/interval/'


interval_ratios = [1., 2., 3., 4., 5., 6., 7.]
direction_accuracy = []
pixel_portions = []



def save_fig(image, process_step, cmap, vmin, vmax, dataset_name, uq_method, data_point_id):
	plt.figure()	
	plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.savefig(uq_method+'_'+process_step+'_'+dataset_name+'_'+str(data_point_id), bbox_inches='tight', pad_inches=0.0, dpi=300)

def direction_test(pred, target, lower, upper, t):
	pt_directions = np.less(pred, target)
	lower_size = pred - lower
	upper_size = upper - pred
	
	size_ratio = lower_size/upper_size
	weights = np.zeros_like(size_ratio)
	weights[size_ratio>=t] = 1.
	weights[size_ratio<=(1./t)] = 1.
	if np.sum(weights) == 0.:
		print('hohoho')
		weights[0] = 1.

	#sanity check on positive interval sizes
	if True in np.less(lower_size,0.) or True in np.less(upper_size,0.):
		print('below 0 alarm')
	pi_directions = np.less(lower_size, upper_size)

	#uncertainty bigger in direction of target?
	tid_same_rate = np.average(np.equal(pt_directions, pi_directions), weights=weights)

	#uncertainty bigger in direction against target?
	tid_opposite_rate = np.average(np.not_equal(pt_directions, pi_directions), weights=weights)
	
	return tid_same_rate, tid_opposite_rate, np.average(weights)

all_data = np.load(data_path+'interval_test_dict.npz.npy', allow_pickle=True)
all_data = all_data[np.newaxis]
all_data = all_data[0]
num_data_points = 200.
for t in interval_ratios:
	tid_same_rate = 0
	tid_opposite_rate = 0
	weight_pixel_coverage = 0.

	test_sample = all_data
	d_results = direction_test(test_sample['pred'], test_sample['target'], test_sample['min'], test_sample['max'], t)
	tid_same_rate += d_results[0]
	tid_opposite_rate += d_results[1]
	weight_pixel_coverage += d_results[2]
	
	direction_accuracy.append(tid_same_rate)
	pixel_portions.append(weight_pixel_coverage)

	print('Interval:')
	print(tid_same_rate, tid_opposite_rate, t, weight_pixel_coverage)

yleft_range = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

fig, ax1 = plt.subplots()

fig.set_size_inches(5, 2.5)

ax2 = ax1.twinx()
ax1.plot([1.0]*len(yleft_range), yleft_range, color = 'white', alpha=0.5)
ax2.plot([1.0]*len(yleft_range), yleft_range, color = 'white', alpha=0.5)
ax1.plot(interval_ratios, direction_accuracy, color ='red', ls='--', marker='x', markersize=7, alpha=0.8, label='$\mathrm{INN}$')
ax1.plot(interval_ratios, [0.5]*len(interval_ratios), color = 'red', alpha=0.8, label='$\mathrm{Chance\ Level\ (MCDrop\ and\ ProbOut)}$')
ax2.plot(interval_ratios, np.array(pixel_portions), color ='blue', ls='--', marker='o', markersize=5, alpha=0.7)
ax1.set_xlabel(r'$\mathrm{Interval\ Direction\ Threshold}$', fontsize=BIGGER_SIZE)
ax1.set_ylabel(r'$\mathrm{Binary\ Direction\ Accuracy}$', fontsize=BIGGER_SIZE)
ax2.set_ylabel(r'$\mathrm{Proportion\ of\ Pixels}$', fontsize=BIGGER_SIZE)

ax1.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')
ax1.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='blue')
plt.xticks(interval_ratios)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)

plt.savefig('inn_direction_figure_1D', bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.show()
