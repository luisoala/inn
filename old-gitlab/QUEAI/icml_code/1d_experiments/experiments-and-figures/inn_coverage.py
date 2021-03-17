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
#num_data_points = int(sys.argv[1]) #integer between 1 and 330

data_path = 'data/run00/interval/'

interval_ratios = [1., 2., 3., 4., 5., 6., 7.]
direction_accuracy = []
pixel_portions = []



def save_fig(image, process_step, cmap, vmin, vmax, dataset_name, uq_method, data_point_id):
	plt.figure()	
	plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
	plt.axis('off')
	plt.savefig(uq_method+'_'+process_step+'_'+dataset_name+'_'+str(data_point_id), bbox_inches='tight', pad_inches=0.0, dpi=300)

def coverage_test(pred, target, lower, upper, t):
	pt_directions = np.less(pred, target)
	lower_size = pred - lower
	upper_size = upper - pred
	
	size_ratio = lower_size/upper_size
	weights = np.ones_like(size_ratio)
	weights[target>=upper] = 0.
	weights[target<=lower] = 1.
	if np.sum(weights) == 0.:
		weights[0,0,0,0] = 1.

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
t = 1.

tid_same_rate = 0
tid_opposite_rate = 0
weight_pixel_coverage = 0.

test_sample = all_data
d_results = coverage_test(test_sample['pred'], test_sample['target'], test_sample['min'], test_sample['max'], t)
tid_same_rate += d_results[0]
tid_opposite_rate += d_results[1]
weight_pixel_coverage += d_results[2]

direction_accuracy.append(tid_same_rate)
pixel_portions.append(weight_pixel_coverage)

print(pixel_portions)
