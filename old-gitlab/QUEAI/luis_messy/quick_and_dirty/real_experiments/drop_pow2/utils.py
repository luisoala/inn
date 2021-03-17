import scipy.io
import re
import os
import numpy as np

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
     
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def get_data(path):
    file_names = next(os.walk(path))[2]
    file_names.sort(key=alphanum_key)
    X = []
    Y = []
    print('Reading data ... ')
    for file_name, count in zip(file_names, range(len(file_names))):
        mat = scipy.io.loadmat(path + file_name)
        X.append(mat['y'])
        Y.append(mat['x'])
        if count % 500 == 0:
            print('Done with # ', count)
    X = np.array(X)
    Y = np.array(Y)
    
    print(X.shape)
    print(Y.shape)
    
    return X[:1600], Y[:1600], X[1600:1800], Y[1600:1800], X[1800:], Y[1800:]


def safe_mkdir(path):
	try:
		os.mkdir(path)
	except OSError:
		pass
		
