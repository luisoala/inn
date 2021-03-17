import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pickle

from TestScript import do_test_run

dropout_sample_dict, interval_sample_dict, probout_sample_dict = do_test_run(0, noisy=True)

min_values = interval_sample_dict['min']
max_values = interval_sample_dict['max']
target_values = interval_sample_dict['target']

distances = (np.maximum(max_values-target_values,0)+np.maximum(target_values-min_values, 0)).flatten()

normalized_targets = (target_values-0.5*(min_values+max_values))/(0.5*(max_values-min_values))
normalized_targets = normalized_targets.flatten()[distances.nonzero()[0]]


hist_interval, bin_edges = np.histogram(normalized_targets, bins=101, density=True, range=(-1, 1))
middles_interval = 0.5*(bin_edges[:-1]+bin_edges[1:])
mpl.rc('xtick', labelsize=40) 
mpl.rc('ytick', labelsize=40)
plt.figure(figsize=(16,9))
plt.bar(middles_interval, hist_interval, width = 2/101)
plt.savefig('TargetDist1.png')

plt.figure(figsize=(16,9))
plt.plot(middles_interval, hist_interval)
plt.savefig('TargetDist2.png')