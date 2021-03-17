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
num_data_points = int(sys.argv[1]) #integer between 1 and 330

runs = sys.argv[2].split(',') #comma separated enumeration of runs, e.g. run00,run01,run02,run03 ...

method = 'interval'


interval_ratios = [1., 1.25, 1.5, 1.75, 2.]



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
	#weights[t+0.1>size_ratio] = 0.
	weights[size_ratio<=(1./t)] = 1.
	#weights[(1./t)-0.1<size_ratio] = 0.
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
	
	return tid_same_rate, tid_opposite_rate, np.average(weights), np.average(pt_directions) #np.sum(weights) #np.average(weights)

direction_accuracy_means = []
direction_accuracy_stds = []
pixel_portions_means = []
pixel_portions_stds = []
for t in interval_ratios:
	direction_accuracy_run_collector = []
	pixel_portions_run_collector = []
	for run in runs:
		tid_same_rate = 0
		tid_opposite_rate = 0
		weight_pixel_coverage = 0.
		pt_direction_dist = 0.
		for i in range(num_data_points):
			test_sample = np.load('data/'+run+'/'+method+'/'+'mayo_test_{:03d}.npz'.format(i))
			d_results = direction_test(test_sample['reconstruction'], test_sample['target'], test_sample['lo'], test_sample['hi'], t)
			tid_same_rate += d_results[0]
			tid_opposite_rate += d_results[1]
			weight_pixel_coverage += d_results[2]
			pt_direction_dist += d_results[3]
			
		tid_same_rate /= num_data_points
		tid_opposite_rate /= num_data_points
		weight_pixel_coverage /= num_data_points
		pt_direction_dist /= num_data_points

		print(run)
		print(tid_same_rate, tid_opposite_rate, t, weight_pixel_coverage, pt_direction_dist)
		direction_accuracy_run_collector.append(tid_same_rate)
		pixel_portions_run_collector.append(weight_pixel_coverage)
		
	direction_accuracy_run_collector = np.array(direction_accuracy_run_collector)
	pixel_portions_run_collector = np.array(pixel_portions_run_collector)
		
	direction_accuracy_means.append(np.mean(direction_accuracy_run_collector))
	direction_accuracy_stds.append(np.std(direction_accuracy_run_collector))
	pixel_portions_means.append(np.mean(pixel_portions_run_collector))
	pixel_portions_stds.append(np.std(pixel_portions_run_collector))

print(direction_accuracy_means)
print(direction_accuracy_stds)

direction_accuracy_means = np.array(direction_accuracy_means)
direction_accuracy_stds = np.array(direction_accuracy_stds)
pixel_portions_means = np.array(pixel_portions_means)
pixel_portions_stds = np.array(pixel_portions_stds)

yleft_range = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

fig, ax1 = plt.subplots()

fig.set_size_inches(5, 2.5)

ax2 = ax1.twinx()
ax1.plot([1.0]*len(yleft_range), yleft_range, color = 'white', alpha=0.0)
ax2.plot([1.0]*len(yleft_range), yleft_range, color = 'white', alpha=0.0)
ax1.plot(interval_ratios, direction_accuracy_means, color ='red', ls='--', marker='x', markersize=7, alpha=0.8, label='$\mathrm{INN}$')
ax1.fill_between(interval_ratios, direction_accuracy_means+direction_accuracy_stds, direction_accuracy_means-direction_accuracy_stds, facecolor='red', alpha=0.1)
ax1.plot(interval_ratios, [0.5]*len(interval_ratios), color = 'red', alpha=0.8, label='$\mathrm{Chance\ Level\ (MCDrop\ and\ ProbOut)}$')
#ax1.ylim((0.,1.))
ax2.plot(interval_ratios, pixel_portions_means, color ='blue', ls='--', marker='o', markersize=5, alpha=0.7)
ax2.fill_between(interval_ratios, pixel_portions_means+pixel_portions_stds, pixel_portions_means-pixel_portions_stds, facecolor='blue', alpha=0.1)
#ax2.plot(interval_ratios, pixel_portions, 'b-')

ax1.set_xlabel(r'$\mathrm{Interval\ Direction\ Threshold}$', fontsize=BIGGER_SIZE) #r'$\mathrm{Iterations}$',fontsize=15
ax1.set_ylabel(r'$\mathrm{Binary\ Direction\ Accuracy}$', fontsize=BIGGER_SIZE) #color='g'
ax2.set_ylabel(r'$\mathrm{Proportion\ of\ Pixels}$', fontsize=BIGGER_SIZE)

ax1.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')
ax1.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='blue')
plt.xticks(interval_ratios)
#ax1.legend(loc='upper center')

plt.savefig('inn_direction_figure_CT', bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.show()
