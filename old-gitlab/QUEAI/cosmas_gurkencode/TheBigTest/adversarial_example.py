import keras.backend as K
from keras.layers import Concatenate, Input, dot

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join('..'))
from IntervalModel_adaptive_beta import ConvNet_Interval

sys.path.append(os.path.join('..', '..', 'luis_messy', 'quick_and_dirty', 'conv_test_run'))
from conv_model import ConvNet_ErrorPred, ConvNet_Det
from utils import get_data, safe_mkdir

#path = "D:/Cosmas/Desktop/NN_uncertainty/GenuineTests/1DInverseProblem/n_512_dist_5_jumps_60_pow_8/"
path = os.path.join('..', '..', '..', 'Uncertainty IP',
                    'n_512_dist_5_jumps_60_pow_8/')
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_data(path)

#X_train += 0.01*np.random.randn(*X_train.shape)
#X_val += 0.01*np.random.randn(*X_val.shape)
#X_test += 0.01*np.random.randn(*X_test.shape)

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


num_epochs = 50
batch_size = 256
num_variance_epochs = 50
sample_idx = 1

convnet_interval = ConvNet_Interval()
convnet_det = ConvNet_Det()
convnet_error_pred = ConvNet_ErrorPred()

convnet_det.train(num_epochs, batch_size, num_epochs, X_train, Y_train, X_val, Y_val, None, None, False, False)
X_Error_train, Y_Error_train, X_Error_val, Y_Error_val, X_Error_test, Y_Error_test = get_error_pred_data(convnet_det, X_train, Y_train, X_val, Y_val, X_test, Y_test)
convnet_error_pred.train(num_variance_epochs, batch_size, num_variance_epochs, X_Error_train, Y_Error_train, X_Error_val, Y_Error_val, None, None, False, False, 0)

convnet_interval.single_training_run(num_epochs, batch_size, X_train, Y_train, X_val, Y_val)
convnet_interval.interval_training_run(num_variance_epochs, batch_size, X_train, Y_train, X_val, Y_val, beta=-1)

##########################
###INTERVAL
##########################
img = X_test[sample_idx:sample_idx+1, :, :]
pre_pred, pre_pre_min, pre_pre_max = convnet_interval.both_convnet.predict([img,img,img])

img_in = convnet_interval.interval_convnet.inputs[0]
gt_in = Input(img_in.shape.as_list()[1:])
out, out_min, out_max = convnet_interval.both_convnet([img_in, img_in, img_in])
loss = -K.mean(K.square(out_max-out_min))/np.mean((pre_pre_max-pre_pre_min)**2)# + K.mean(K.square(out-gt_in))

error_func = K.function([img_in, gt_in],[loss])
gradient = K.gradients(loss, img_in)[0]
gradient_func = K.function([img_in, gt_in], [gradient])

img = X_test[sample_idx:sample_idx+1, :, :]
gt = Y_test[sample_idx:sample_idx+1, :, :]


pre_pred, pre_min, pre_max = convnet_interval.both_convnet.predict([img,img,img])
pre_err = np.abs(pre_pred - gt)
pre_err_pred = pre_max-pre_min
pre_adv_error = error_func([img, gt])[0]

tmp_img = img
for i in range(200):
    adv_gradient = gradient_func([tmp_img, gt])[0]
    regularizer = img-tmp_img
    tmp_img = tmp_img + 1e-2 * adv_gradient/np.linalg.norm(adv_gradient) + 1e-2 * 3 * (regularizer/(np.linalg.norm(regularizer)+1e-1))
img_adv = tmp_img

post_pred, post_min, post_max = convnet_interval.both_convnet.predict([img_adv, img_adv, img_adv])
post_err = np.abs(post_pred - gt)
post_err_pred = post_max-post_min
post_adv_error = error_func([img_adv, gt])[0]


print('pre interval\t', pre_adv_error)
print('post interval\t', post_adv_error)
##########################

plt.figure('interval')
plt.subplot(4, 1, 1)
plt.plot(img.squeeze(), color='orange')
plt.plot(gt.squeeze(), color='black')
plt.plot(pre_pred.squeeze(), color='blue')

plt.subplot(4, 1, 2)
plt.plot(pre_err.squeeze(), color='red')
plt.plot(pre_err_pred.squeeze(), color='magenta')

plt.subplot(4, 1, 3)
plt.plot(img_adv.squeeze(), color='orange')
plt.plot(img.squeeze(), color='yellow')
plt.plot(gt.squeeze(), color='black')
plt.plot(post_pred.squeeze(), color='blue')

plt.subplot(4, 1, 4)
plt.plot(post_err.squeeze(), color='red')
plt.plot(post_err_pred.squeeze(), color='magenta')




##########################
###ERROR PRED
##########################
img = X_test[sample_idx:sample_idx+1, :, :]
pre_pre_pred = convnet_det.convnet.predict(img)
pre_pre_err = np.abs(pre_pre_pred - gt)
pre_pre_err_pred = convnet_error_pred.convnet.predict(np.concatenate([img, pre_pre_pred], axis=-1))

img_in = convnet_det.convnet.inputs[0]
gt_in = Input(img_in.shape.as_list()[1:])
img_out = convnet_det.convnet(img_in)
combined = Concatenate(axis=-1)([img_in, img_out])
error = K.abs(img_out-gt_in)
error_out = convnet_error_pred.convnet(combined)
# loss = K.mean(-dot([error_out, error], -2))
loss = -K.mean(K.square(error_out))/np.mean((pre_pre_err_pred)**2)# + K.mean(K.square(img_out - gt_in))
error_func = K.function([img_in, gt_in], [loss])
gradient = K.gradients(loss, img_in)[0]
gradient_func = K.function([img_in, gt_in], [gradient])


img = X_test[sample_idx:sample_idx+1, :, :]
gt = Y_test[sample_idx:sample_idx+1, :, :]

pre_pred = convnet_det.convnet.predict(img)
pre_err = np.abs(pre_pred - gt)
pre_err_pred = convnet_error_pred.convnet.predict(np.concatenate([img, pre_pred], axis=-1))
pre_adv_error = error_func([img, gt])[0]

tmp_img = img
for i in range(200):
    adv_gradient = gradient_func([tmp_img, gt])[0]
    regularizer = img-tmp_img
    tmp_img = tmp_img + 1e-2 * adv_gradient/np.linalg.norm(adv_gradient) + 1e-2 * 3 * (regularizer/(np.linalg.norm(regularizer)+1e-1))
img_adv = tmp_img


post_pred = convnet_det.convnet.predict(img_adv)
post_err = np.abs(post_pred - gt)
post_err_pred = convnet_error_pred.convnet.predict(np.concatenate([img_adv, post_pred], axis=-1))
post_adv_error = error_func([img_adv, gt])[0]

print('pre error pred\t', pre_adv_error)
print('post error pred\t', post_adv_error)
##########################


plt.figure()
plt.subplot(4, 1, 1)
plt.plot(img.squeeze(), color='orange')
plt.plot(gt.squeeze(), color='black')
plt.plot(pre_pred.squeeze(), color='blue')

plt.subplot(4, 1, 2)
plt.plot(pre_err.squeeze(), color='red')
plt.plot(pre_err_pred.squeeze(), color='magenta')

plt.subplot(4, 1, 3)
plt.plot(img_adv.squeeze(), color='orange')
plt.plot(img.squeeze(), color='yellow')
plt.plot(gt.squeeze(), color='black')
plt.plot(post_pred.squeeze(), color='blue')

plt.subplot(4, 1, 4)
plt.plot(post_err.squeeze(), color='red')
plt.plot(post_err_pred.squeeze(), color='magenta')

plt.show()
