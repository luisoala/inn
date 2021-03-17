from IntervalModel_adaptive_beta import ConvNet_Interval

import os, sys
sys.path.append(os.path.join('..', 'luis_messy', 'quick_and_dirty', 'conv_test_run'))
from utils import get_data, safe_mkdir

safe_mkdir('matrices-train')
safe_mkdir('matrices-test')

DATA_PATH = sys.argv[1]     # should be a string
MODEL_TYPE = 'interval'     # sys.argv[2]  #'prob', 'drop', 'det', 'error_pred'
EPOCHS = int(sys.argv[3])   # should be integer
BATCH_SIZE = int(sys.argv[4])   # should be integer
SAMPLING_INTERVAL = int(sys.argv[5])    # should be integer dividing epochs

if MODEL_TYPE == 'interval':
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(DATA_PATH)
    convnet_interval = ConvNet_Interval()

    convnet_interval.train(
        EPOCHS,
        BATCH_SIZE,
        SAMPLING_INTERVAL,
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        True,
        True,
    )
