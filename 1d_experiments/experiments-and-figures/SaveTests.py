import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pickle

from TestScript import do_test_run

dropout_sample_dict, interval_sample_dict, probout_sample_dict = do_test_run(0, noisy=False)

np.save('dropout_test_dict.npz', dropout_sample_dict)
np.save('interval_test_dict.npz', interval_sample_dict)
np.save('probout_test_dict.npz', probout_sample_dict)

dropout_sample_dict, interval_sample_dict, probout_sample_dict = do_test_run(0, noisy=True)

np.save('dropout_test_dict_noise.npz', dropout_sample_dict)
np.save('interval_test_dict_noise.npz', interval_sample_dict)
np.save('probout_test_dict_noise.npz', probout_sample_dict)