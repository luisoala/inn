import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('./matrices-test/epoch5.pickle', 'rb') as handle:
    data = pickle.load(handle)
    target = data['targets']
    pred = data['pred']
    lo, hi = data['min'], data['max']

    for i in range(6):
        plt.figure()
        plt.subplot(2, 1, 1)
        x_axis = np.arange(512)

        plt.fill_between(x_axis, lo[i,:,0], hi[i,:,0],facecolor=[0,0,0.85],interpolate=True)
        plt.plot(x_axis,target[i,:,0],color='black')
        plt.plot(x_axis,pred[i,:,0],color='orange')

        plt.subplot(2,1,2)
        plt.fill_between(x_axis, 0*x_axis, hi[i,:,0]-lo[i,:,0],facecolor=[0,0,0.85],interpolate=True)
        plt.plot(x_axis, np.abs(pred[i,:,0]-target[i,:,0]),color='red')

    plt.show()
