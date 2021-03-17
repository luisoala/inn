import numpy as np
import matplotlib.pyplot as plt
import pickle


sample = 0
snapshot_epochs = [5,10,20,40,80,100]

x_axis = np.arange(512)

def show_snapshots(test_path, title):
    for epoch in snapshot_epochs[::-1]:
        plt.figure().suptitle('epoch {}, '.format(epoch)+title)

        path = test_path+'/IntervalConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))
        beta = test_dict['beta']
        plt.subplot(6,1,1).set_title('Output of Interval Model beta='+str(beta))
        plt.ylim(-1,1)

        plt.fill_between(x_axis,test_dict['min'][sample,:],test_dict['max'][sample,:],facecolor=[0,0,0.8],interpolate=True)
        plt.plot(x_axis,test_dict['targets'][sample,:],color='black')
        plt.plot(x_axis,test_dict['pred'][sample,:],color='orange')

        plt.subplot(6,1,2).set_title('Errors of Interval Model beta='+str(beta))
        y_lim = 1.5*np.max(np.abs(test_dict['targets'][sample,:]-test_dict['pred'][sample,:]))
        plt.ylim(0,y_lim)

        plt.fill_between(x_axis,0*x_axis,test_dict['max'][sample,:]-test_dict['min'][sample,:],facecolor=[0.6,0,0],interpolate=True)
        plt.plot(x_axis,np.abs(test_dict['targets'][sample,:]-test_dict['pred'][sample,:]),color='black')


        path = test_path+'/DropoutConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))
        plt.subplot(6,1,3).set_title('Output of Dropout Model')
        plt.ylim(-1,1)

        plt.plot(x_axis,test_dict['targets'][sample,:],color='black')
        plt.plot(x_axis,test_dict['mean'][sample,:],color='orange')

        plt.subplot(6,1,4).set_title('Errors of Dropout Model')
        plt.ylim(0,y_lim)


        abs_error = np.abs(test_dict['targets'][sample,:]-test_dict['mean'][sample,:])
        standard_deviation = np.dot(abs_error,test_dict['standard_deviation'][sample,:])*test_dict['standard_deviation'][sample,:]/np.linalg.norm(test_dict['standard_deviation'][sample,:])**2
        variance = np.dot(abs_error,test_dict['variance'][sample,:])*test_dict['variance'][sample,:]/np.linalg.norm(test_dict['variance'][sample,:])**2
        plt.plot(x_axis,np.abs(test_dict['targets'][sample,:]-test_dict['mean'][sample,:]),color='black')
        plt.plot(x_axis,variance,color='red')
        plt.plot(x_axis,standard_deviation,color='blue')


        path = test_path+'/ErrorPredConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))
        plt.subplot(6,1,5).set_title('Output of Error Prediction Model')
        plt.ylim(-1,1)

        plt.plot(x_axis,test_dict['targets'][sample,:],color='black')
        plt.plot(x_axis,test_dict['pred'][sample,:],color='orange')

        plt.subplot(6,1,6).set_title('Errors of Error Prediction Model')
        plt.ylim(0,y_lim)

        plt.plot(x_axis,np.abs(test_dict['targets'][sample,:]-test_dict['pred'][sample,:]),color='black')
        plt.plot(x_axis,test_dict['error_prediction'][sample,:],color='red')

    plt.show()


for power in [2,4,8]:
    show_snapshots('Power{}Training/TestingOnPow{}'.format(power, power), 'Power {} tested on power {}'.format(power, power))


show_snapshots('Power2Training/TestingOnPow8', 'Trained on power 2, tested on power 8')
show_snapshots('Power8Training/TestingOnPow8Noisy', 'Trained on power 8, tested on noisy power 8 inputs')
show_snapshots('Power8TrainingNoisyOutput/TestingOnPow8', 'Trained on power 8 with noisy labels, tested on power 8')


def optimized_mse_similarity(signal, pred):
    #if len(signal.shape) != 1:
    #    raise ValueError('Optimized mse similarity not implemented for batch data')

    mse = 0.0
    for i in range(signal.shape[0]):
        pred_norm_sq = np.dot(pred[i,:],pred[i,:])
        #if pred_norm < 1e-8:
        #    mse += np.dot(signal[i,:], signal[i,:])/512
        #    continue

        difference = signal[i,:]-np.dot(signal[i,:],pred[i,:]/pred_norm_sq)*pred[i,:]

        mse += np.dot(difference, difference)/512.0

    return mse/signal.shape[0]

x_axis = np.array(snapshot_epochs)

def run_comparison(test_path):
    similarity = np.zeros(x_axis.shape)
    for i,epoch in enumerate(snapshot_epochs):
        path = test_path+'/ErrorPredConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))

        similarity[i] = optimized_mse_similarity(
                                np.abs(test_dict['targets']-test_dict['pred']),
                                test_dict['error_prediction'],
                                )
    plt.plot(x_axis, similarity, color='red', label='Error Prediction')

    for i,epoch in enumerate(snapshot_epochs):
        path = test_path+'/DropoutConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))

        similarity[i] = optimized_mse_similarity(
                                np.abs(test_dict['targets']-test_dict['mean']),
                                test_dict['variance'],
                                )
    plt.plot(x_axis, similarity, color='darkblue', label='Dropout Variance')

    for i,epoch in enumerate(snapshot_epochs):
        path = test_path+'/DropoutConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))

        similarity[i] = optimized_mse_similarity(
                                np.abs(test_dict['targets']-test_dict['mean']),
                                test_dict['standard_deviation'],
                                )
    plt.plot(x_axis, similarity, color='lightblue', label='Dropout Standard Deviation')

    for i,epoch in enumerate(snapshot_epochs):
        path = test_path+'/IntervalConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))

        similarity[i] = optimized_mse_similarity(
                                np.abs(test_dict['targets']-test_dict['pred']),
                                test_dict['uncertainty'],
                                )
    plt.plot(x_axis, similarity, color='green', label='Interval size')


def run_comparison_extra(test_path):
    similarity = np.zeros(x_axis.shape)
    for i,epoch in enumerate(snapshot_epochs):
        path = test_path+'/ErrorPredConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))

        similarity[i] = np.mean(np.maximum(test_dict['targets']-(test_dict['pred']+test_dict['error_prediction']),0)**2+np.maximum(test_dict['pred']-test_dict['error_prediction']-test_dict['targets'],0)**2)

    plt.plot(x_axis, similarity, color='red', label='Error Prediction')

    for i,epoch in enumerate(snapshot_epochs):
        path = test_path+'/IntervalConvNet/Epoch{}.pickle'.format(epoch)
        test_dict = pickle.load(open(path,'rb'))

        similarity[i] = np.mean(np.maximum(test_dict['targets']-test_dict['max'],0)**2+np.maximum(test_dict['min']-test_dict['targets'],0)**2)

    plt.plot(x_axis, similarity, color='green', label='Interval size')



plt.figure().suptitle('different mathods similarity measure')

plt.subplot(3,3,1).set_title('Power 2 tested on power 2')
plt.yscale('log')
run_comparison('Power2Training/TestingOnPow2')

plt.subplot(3,3,2).set_title('Power 4 tested on power 4')
plt.yscale('log')
run_comparison('Power4Training/TestingOnPow4')

plt.subplot(3,3,3).set_title('Power 8 tested on power 8')
plt.yscale('log')
run_comparison('Power8Training/TestingOnPow8')


plt.subplot(3,3,4).set_title('Power 2 tested on noisy power 2')
plt.yscale('log')
run_comparison('Power2Training/TestingOnPow2Noisy')

plt.subplot(3,3,5).set_title('Power 4 tested on noisy power 4')
plt.yscale('log')
run_comparison('Power4Training/TestingOnPow4Noisy')

plt.subplot(3,3,6).set_title('Power 8 tested on noisy power 8')
plt.yscale('log')
run_comparison('Power8Training/TestingOnPow8Noisy')


plt.subplot(3,3,7).set_title('Power 2 tested on power 8')
plt.yscale('log')
run_comparison('Power2Training/TestingOnPow8')

plt.subplot(3,3,8).set_title('Power 8 tested on power 2')
plt.yscale('log')
run_comparison('Power8Training/TestingOnPow2')

plt.subplot(3,3,9).set_title('Trained on noisy power 8 labels tested with noisy power 8 labels')
plt.yscale('log')
run_comparison('Power8TrainingNoisyOutput/TestingOnPow8')

plt.legend()




plt.figure().suptitle('interval error pred comparion how much is drin')

plt.subplot(3,3,1).set_title('Power 2 tested on power 2')
plt.yscale('log')
run_comparison_extra('Power2Training/TestingOnPow2')

plt.subplot(3,3,2).set_title('Power 4 tested on power 4')
plt.yscale('log')
run_comparison_extra('Power4Training/TestingOnPow4')

plt.subplot(3,3,3).set_title('Power 8 tested on power 8')
plt.yscale('log')
run_comparison_extra('Power8Training/TestingOnPow8')


plt.subplot(3,3,4).set_title('Power 2 tested on noisy power 2')
plt.yscale('log')
run_comparison_extra('Power2Training/TestingOnPow2Noisy')

plt.subplot(3,3,5).set_title('Power 4 tested on noisy power 4')
plt.yscale('log')
run_comparison_extra('Power4Training/TestingOnPow4Noisy')

plt.subplot(3,3,6).set_title('Power 8 tested on noisy power 8')
plt.yscale('log')
run_comparison_extra('Power8Training/TestingOnPow8Noisy')


plt.subplot(3,3,7).set_title('Power 2 tested on power 8')
plt.yscale('log')
run_comparison_extra('Power2Training/TestingOnPow8')

plt.subplot(3,3,8).set_title('Power 8 tested on power 2')
plt.yscale('log')
run_comparison_extra('Power8Training/TestingOnPow2')

plt.subplot(3,3,9).set_title('Trained on noisy power 8 labels tested with noisy power 8 labels')
plt.yscale('log')
run_comparison_extra('Power8TrainingNoisyOutput/TestingOnPow8')

plt.legend()
plt.show()
