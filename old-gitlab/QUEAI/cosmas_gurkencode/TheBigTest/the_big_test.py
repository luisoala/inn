import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join('..'))
from IntervalModel_adaptive_beta import ConvNet_Interval

sys.path.append(os.path.join('..', '..', 'luis_messy', 'quick_and_dirty', 'conv_test_run'))
from conv_model import ConvNet_Drop, ConvNet_ErrorPred, ConvNet_Det
from utils import get_data, safe_mkdir

test_dict = {}

path = os.path.join('..', '..', '..', 'Uncertainty IP',
                    'n_512_dist_5_jumps_60_pow_2/')
X_pow2_train, Y_pow2_train, X_pow2_val, Y_pow2_val, X_pow2_test, Y_pow2_test = get_data(path)

path = os.path.join('..', '..', '..', 'Uncertainty IP',
                    'n_512_dist_5_jumps_60_pow_4/')
X_pow4_train, Y_pow4_train, X_pow4_val, Y_pow4_val, X_pow4_test, Y_pow4_test = get_data(path)

path = os.path.join('..', '..', '..', 'Uncertainty IP',
                    'n_512_dist_5_jumps_60_pow_8/')
X_pow8_train, Y_pow8_train, X_pow8_val, Y_pow8_val, X_pow8_test, Y_pow8_test = get_data(path)

X_pow2_test_noisy = X_pow2_test+0.02*np.random.standard_normal(X_pow2_test.shape)
X_pow4_test_noisy = X_pow4_test+0.02*np.random.standard_normal(X_pow4_test.shape)
X_pow8_test_noisy = X_pow8_test+0.02*np.random.standard_normal(X_pow8_test.shape)

Y_pow8_train_noisy = Y_pow8_train+0.04*np.random.standard_normal(Y_pow8_train.shape)
Y_pow8_val_noisy = Y_pow8_val+0.04*np.random.standard_normal(Y_pow8_val.shape)
Y_pow8_test_noisy = Y_pow8_test+0.04*np.random.standard_normal(Y_pow8_test.shape)

################################
###TEST PARAMETERS:###
total_number_of_epochs = 100    ###doesnt do anything atm
snapshot_epochs = [5,10,20,40,80,100]
batch_size = 256

interval_errorPred_epochs = 50
num_dropout_draws = 20

################################


def save_dict(save_path, matrices):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path,'wb') as handle:
        pickle.dump(matrices, handle)


def get_error_pred_data(convnet_pred, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    Preds_train = convnet_pred.convnet.predict(X_train)
    Preds_val = convnet_pred.convnet.predict(X_val)
    Preds_test = convnet_pred.convnet.predict(X_test)
    #get errors
    Error_train = np.abs(Y_train - Preds_train)
    Error_val = np.abs(Y_val - Preds_val)
    Error_test = np.abs(Y_test - Preds_test)

    #build error pred inputs from prediction model inputs and outputs
    disc_input_train = np.concatenate([X_train, Preds_train], axis=-1)
    disc_input_val = np.concatenate([X_val, Preds_val], axis=-1)
    disc_input_test = np.concatenate([X_test, Preds_test], axis=-1)
    return disc_input_train, Error_train, disc_input_val, Error_val, disc_input_test, Error_test


def run_dropout_test(convnet, X_test, Y_test):
    results = []
    for i in range(num_dropout_draws):
        result = convnet.convnet.predict(X_test, batch_size=1)
        results.append(result)
    results = np.array(results)

    mean = np.mean(results, axis=0)
    variance = np.var(results, axis=0)

    matrices = {
        'inputs': X_test[:,:,0],
        'targets': Y_test[:,:,0],
        'mean': mean[:,:,0],
        'variance': variance[:,:,0],
        'standard_deviation': np.sqrt(variance[:,:,0]),
    }
    return matrices


def run_interval_test(convnet, X_test, Y_test):
    out, min_out, max_out = convnet.sample_tests(X_test, Y_test)
    matrices = {
        'inputs': X_test[:,:,0],
        'targets': Y_test[:,:,0],
        'pred': out[:,:,0],
        'uncertainty': max_out[:,:,0]-min_out[:,:,0],
        'min': min_out[:,:,0],
        'max': max_out[:,:,0],
        'beta': convnet.initialized_beta,
    }
    return matrices


def run_error_pred_test(convnet_det, convnet_pred, X_test, Y_test):
    out = convnet_det.convnet.predict(X_test)
    X_Error_test = np.concatenate([X_test, out], axis=-1)
    error_pred = convnet_pred.convnet.predict(X_Error_test)
    matrices = {
        'inputs': X_test[:,:,0],
        'targets': Y_test[:,:,0],
        'pred': out[:,:,0],
        'error_prediction': error_pred[:,:,0],
    }
    return matrices


##############################
"""Power 2 tests"""
##############################

##########INTERVAL TRAINING################
convnet_interval = ConvNet_Interval()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_interval.single_training_run(num_epochs, batch_size, X_pow2_train, Y_pow2_train, X_pow2_val, Y_pow2_val)
    convnet_interval.interval_training_run(interval_errorPred_epochs, batch_size, X_pow2_train, Y_pow2_train, X_pow2_val, Y_pow2_val, beta=-1)

    matrices = run_interval_test(convnet_interval, X_pow2_test, Y_pow2_test)
    save_path = 'Power2Training/TestingOnPow2/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_interval_test(convnet_interval, X_pow8_test, Y_pow8_test)
    save_path = 'Power2Training/TestingOnPow8/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_interval_test(convnet_interval, X_pow2_test_noisy, Y_pow2_test)
    save_path = 'Power2Training/TestingOnPow2Noisy/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_interval
##########DROPOUT TRAINING#################
convnet_drop = ConvNet_Drop()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_drop.train(num_epochs, batch_size, num_epochs, X_pow2_train, Y_pow2_train, X_pow2_val, Y_pow2_val, None, None, False, False)

    matrices = run_dropout_test(convnet_drop, X_pow2_test, Y_pow2_test)
    save_path = 'Power2Training/TestingOnPow2/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_dropout_test(convnet_drop, X_pow8_test, Y_pow8_test)
    save_path = 'Power2Training/TestingOnPow8/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_dropout_test(convnet_drop, X_pow2_test_noisy, Y_pow2_test)
    save_path = 'Power2Training/TestingOnPow2Noisy/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_drop
##########ERROR PRED TRAINING##############
convnet_det = ConvNet_Det()
convnet_error_pred = ConvNet_ErrorPred()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_det.train(num_epochs, batch_size, num_epochs, X_pow2_train, Y_pow2_train, X_pow2_val, Y_pow2_val, None, None, False, False)
    X_Error_pow2_train, Y_Error_po2_train, X_Error_pow2_val, Y_Error_pow2_val, X_Error_pow2_test, Y_Error_pow2_test = get_error_pred_data(convnet_det, X_pow2_train, Y_pow2_train, X_pow2_val, Y_pow2_val, X_pow2_test, Y_pow2_test)
    convnet_error_pred.train(interval_errorPred_epochs, batch_size, interval_errorPred_epochs, X_Error_pow2_train, Y_Error_po2_train, X_Error_pow2_val, Y_Error_pow2_val, None, None, False, False, 0)
    del X_Error_pow2_train, Y_Error_po2_train, X_Error_pow2_val, Y_Error_pow2_val, X_Error_pow2_test, Y_Error_pow2_test

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow2_test, Y_pow2_test)
    save_path = 'Power2Training/TestingOnPow2/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow8_test, Y_pow8_test)
    save_path = 'Power2Training/TestingOnPow8/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow2_test_noisy, Y_pow2_test)
    save_path = 'Power2Training/TestingOnPow2Noisy/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)
del convnet_det, convnet_error_pred


##############################
"""Power 4 tests"""
##############################

##########INTERVAL TRAINING################
convnet_interval = ConvNet_Interval()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_interval.single_training_run(num_epochs, batch_size, X_pow4_train, Y_pow4_train, X_pow4_val, Y_pow4_val)
    convnet_interval.interval_training_run(interval_errorPred_epochs, batch_size, X_pow4_train, Y_pow4_train, X_pow4_val, Y_pow4_val, beta=-1)

    matrices = run_interval_test(convnet_interval, X_pow4_test, Y_pow4_test)
    save_path = 'Power4Training/TestingOnPow4/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_interval_test(convnet_interval, X_pow4_test_noisy, Y_pow4_test)
    save_path = 'Power4Training/TestingOnPow4Noisy/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_interval
##########DROPOUT TRAINING#################
convnet_drop = ConvNet_Drop()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_drop.train(num_epochs, batch_size, num_epochs, X_pow4_train, Y_pow4_train, X_pow4_val, Y_pow4_val, None, None, False, False)

    matrices = run_dropout_test(convnet_drop, X_pow4_test, Y_pow4_test)
    save_path = 'Power4Training/TestingOnPow4/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_dropout_test(convnet_drop, X_pow4_test_noisy, Y_pow4_test)
    save_path = 'Power4Training/TestingOnPow4Noisy/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_drop
##########ERROR PRED TRAINING##############
convnet_det = ConvNet_Det()
convnet_error_pred = ConvNet_ErrorPred()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_det.train(num_epochs, batch_size, num_epochs, X_pow4_train, Y_pow4_train, X_pow4_val, Y_pow4_val, None, None, False, False)
    X_Error_pow4_train, Y_Error_po4_train, X_Error_pow4_val, Y_Error_pow4_val, X_Error_pow4_test, Y_Error_pow4_test = get_error_pred_data(convnet_det, X_pow4_train, Y_pow4_train, X_pow4_val, Y_pow4_val, X_pow4_test, Y_pow4_test)
    convnet_error_pred.train(interval_errorPred_epochs, batch_size, interval_errorPred_epochs, X_Error_pow4_train, Y_Error_po4_train, X_Error_pow4_val, Y_Error_pow4_val, None, None, False, False, 0)
    del X_Error_pow4_train, Y_Error_po4_train, X_Error_pow4_val, Y_Error_pow4_val, X_Error_pow4_test, Y_Error_pow4_test

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow4_test, Y_pow4_test)
    save_path = 'Power4Training/TestingOnPow4/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow4_test_noisy, Y_pow4_test)
    save_path = 'Power4Training/TestingOnPow4Noisy/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)
del convnet_det, convnet_error_pred


##############################
"""Power 8 tests"""
##############################

##########INTERVAL TRAINING################
convnet_interval = ConvNet_Interval()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_interval.single_training_run(num_epochs, batch_size, X_pow8_train, Y_pow8_train, X_pow8_val, Y_pow8_val)
    convnet_interval.interval_training_run(interval_errorPred_epochs, batch_size, X_pow8_train, Y_pow8_train, X_pow8_val, Y_pow8_val, beta=-1)

    matrices = run_interval_test(convnet_interval, X_pow8_test, Y_pow8_test)
    save_path = 'Power8Training/TestingOnPow8/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_interval_test(convnet_interval, X_pow2_test, Y_pow2_test)
    save_path = 'Power8Training/TestingOnPow2/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_interval_test(convnet_interval, X_pow8_test_noisy, Y_pow8_test)
    save_path = 'Power8Training/TestingOnPow8Noisy/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_interval
##########DROPOUT TRAINING#################
convnet_drop = ConvNet_Drop()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_drop.train(num_epochs, batch_size, num_epochs, X_pow8_train, Y_pow8_train, X_pow8_val, Y_pow8_val, None, None, False, False)

    matrices = run_dropout_test(convnet_drop, X_pow8_test, Y_pow8_test)
    save_path = 'Power8Training/TestingOnPow8/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_dropout_test(convnet_drop, X_pow2_test, Y_pow2_test)
    save_path = 'Power8Training/TestingOnPow2/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_dropout_test(convnet_drop, X_pow8_test_noisy, Y_pow8_test)
    save_path = 'Power8Training/TestingOnPow8Noisy/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_drop
##########ERROR PRED TRAINING##############
convnet_det = ConvNet_Det()
convnet_error_pred = ConvNet_ErrorPred()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_det.train(num_epochs, batch_size, num_epochs, X_pow8_train, Y_pow8_train, X_pow8_val, Y_pow8_val, None, None, False, False)
    X_Error_pow8_train, Y_Error_po8_train, X_Error_pow8_val, Y_Error_pow8_val, X_Error_pow8_test, Y_Error_pow8_test = get_error_pred_data(convnet_det, X_pow8_train, Y_pow8_train, X_pow8_val, Y_pow8_val, X_pow8_test, Y_pow8_test)
    convnet_error_pred.train(interval_errorPred_epochs, batch_size, interval_errorPred_epochs, X_Error_pow8_train, Y_Error_po8_train, X_Error_pow8_val, Y_Error_pow8_val, None, None, False, False, 0)
    del X_Error_pow8_train, Y_Error_po8_train, X_Error_pow8_val, Y_Error_pow8_val, X_Error_pow8_test, Y_Error_pow8_test

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow8_test, Y_pow8_test)
    save_path = 'Power8Training/TestingOnPow8/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow2_test, Y_pow2_test)
    save_path = 'Power8Training/TestingOnPow2/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow8_test_noisy, Y_pow8_test)
    save_path = 'Power8Training/TestingOnPow8Noisy/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_det, convnet_error_pred


##############################
"""Power 8 Noisy output tests"""
##############################

##########INTERVAL TRAINING################
convnet_interval = ConvNet_Interval()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_interval.single_training_run(num_epochs, batch_size, X_pow8_train, Y_pow8_train_noisy, X_pow8_val, Y_pow8_val_noisy)
    convnet_interval.interval_training_run(interval_errorPred_epochs, batch_size, X_pow8_train, Y_pow8_train_noisy, X_pow8_val, Y_pow8_val_noisy, beta=-1)

    matrices = run_interval_test(convnet_interval, X_pow8_test, Y_pow8_test_noisy)
    save_path = 'Power8TrainingNoisyOutput/TestingOnPow8/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_interval_test(convnet_interval, X_pow8_test_noisy, Y_pow8_test_noisy)
    save_path = 'Power8TrainingNoisyOutput/TestingOnPow8Noisy/IntervalConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_interval
##########DROPOUT TRAINING#################
convnet_drop = ConvNet_Drop()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_drop.train(num_epochs, batch_size, num_epochs, X_pow8_train, Y_pow8_train_noisy, X_pow8_val, Y_pow8_val_noisy, None, None, False, False)

    matrices = run_dropout_test(convnet_drop, X_pow8_test, Y_pow8_test_noisy)
    save_path = 'Power8TrainingNoisyOutput/TestingOnPow8/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_dropout_test(convnet_drop, X_pow8_test_noisy, Y_pow8_test_noisy)
    save_path = 'Power8TrainingNoisyOutput/TestingOnPow8Noisy/DropoutConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_drop
##########ERROR PRED TRAINING##############
convnet_det = ConvNet_Det()
convnet_error_pred = ConvNet_ErrorPred()
for num_epochs, snapshot_epoch in [(ep1-ep0,ep1) for ep1,ep0 in zip(snapshot_epochs,[0]+snapshot_epochs[:-1])]:

    convnet_det.train(num_epochs, batch_size, num_epochs, X_pow8_train, Y_pow8_train_noisy, X_pow8_val, Y_pow8_val_noisy, None, None, False, False)
    X_Error_pow8_train, Y_Error_po8_train_noisy, X_Error_pow8_val, Y_Error_pow8_val_noisy, X_Error_pow8_test, Y_Error_pow8_test_noisy = get_error_pred_data(convnet_det, X_pow8_train, Y_pow8_train_noisy, X_pow8_val, Y_pow8_val_noisy, X_pow8_test, Y_pow8_test_noisy)
    convnet_error_pred.train(interval_errorPred_epochs, batch_size, interval_errorPred_epochs, X_Error_pow8_train, Y_Error_po8_train_noisy, X_Error_pow8_val, Y_Error_pow8_val_noisy, None, None, False, False, 0)
    del X_Error_pow8_train, Y_Error_po8_train_noisy, X_Error_pow8_val, Y_Error_pow8_val_noisy, X_Error_pow8_test, Y_Error_pow8_test_noisy

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow8_test, Y_pow8_test_noisy)
    save_path = 'Power8TrainingNoisyOutput/TestingOnPow8/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

    matrices = run_error_pred_test(convnet_det, convnet_error_pred, X_pow8_test_noisy, Y_pow8_test_noisy)
    save_path = 'Power8TrainingNoisyOutput/TestingOnPow8Noisy/ErrorPredConvNet/Epoch{}.pickle'.format(snapshot_epoch)
    save_dict(save_path, matrices)

del convnet_det, convnet_error_pred
