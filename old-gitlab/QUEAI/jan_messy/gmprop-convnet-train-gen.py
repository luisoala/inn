import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import MAE
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping,
    TerminateOnNaN,
)

from gmprop.layers import Conv2D
from gmprop.losses import NLLPE

import parsedata

# # # # # Setup parameters # # # # #
BATCH_SIZE = 64
EPOCHS = 200
INIT_LEARNING_RATE = 1e-6

SCRIPTNAME = 'gmprop-convnet-train-gen'
LOGDIR = os.path.join('.', 'logs', SCRIPTNAME)
CKPTSDIR = os.path.join(LOGDIR, 'ckpts')

# # # # # Generate image data # # # # #
T_SIZE, V_SIZE = 10000, 300     # size of training and validation set
steps_per_epoch = T_SIZE//BATCH_SIZE
val_steps = V_SIZE//BATCH_SIZE
TN, TS = np.random.randn(T_SIZE, 2), np.random.randn(T_SIZE, 1) 
def train_generator():
    count = 0
    while count < T_SIZE:
        image = parsedata.generate_stripes(20*TN[count, :], 10*TS[count, :])
        masked = parsedata.immask(image)
        masked = np.expand_dims(masked, axis=2)
        image = np.expand_dims(image, axis=2)
        count += 1
        yield masked, image

VN, VS = np.random.randn(V_SIZE, 2), np.random.randn(V_SIZE, 1)         
def val_generator():
    count = 0
    while count < V_SIZE:
        image = parsedata.generate_stripes(20*VN[count, :], 10*VS[count, :])
        masked = parsedata.immask(image)
        masked = np.expand_dims(masked, axis=2)
        image = np.expand_dims(image, axis=2)
        count += 1
        yield masked, image
       
traindata = tf.data.Dataset.from_generator(
    train_generator,
    (tf.float32, tf.float32),
)
traindata = traindata.map(
    lambda x, y: parsedata.resize_images(x, y, size=(64, 64)),
    num_parallel_calls=3,
)
traindata = traindata.map(lambda img, target: (
        {
            'input_mean': img,
            'input_variance': 0.5*np.ones(img.shape.as_list()),
            'target': target,
        },
        {
            'predicted_mean': target,
        }
    )
)
traindata = traindata.repeat().shuffle(3000)
traindata = traindata.batch(BATCH_SIZE).prefetch(1)

valdata = tf.data.Dataset.from_generator(
    val_generator,
    (tf.float32, tf.float32),
)
valdata = valdata.map(
    lambda x, y: parsedata.resize_images(x, y, size=(64, 64)),
    num_parallel_calls=3,
)
valdata = valdata.map(lambda img, target: (
        {
            'input_mean': img,
            'input_variance': 0.5*np.ones(img.shape),
            'target': target,
        },
        {
            'predicted_mean': target,
        }
    )
)
valdata = valdata.repeat()
valdata = valdata.batch(BATCH_SIZE).prefetch(1)

input_shape = [64, 64, 1]

# # # # # Average Pooling ConvNet # # # # #
# input layers
input_mean = Input(input_shape, name='input_mean')
input_variance = Input(input_shape, name='input_variance')

# target layer for custom loss function
target = Input(input_shape, name='target')

# conv layers
conv1 = Conv2D(32, 31, activation='relu', padding='same',
               kernel_initializer='he_normal')([input_mean, input_variance])
               
conv2 = Conv2D(32, 23, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv1)
               
conv3 = Conv2D(32, 19, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv2)
               
conv4 = Conv2D(64, 15, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv3)

conv5 = Conv2D(64, 11, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv4)

conv6 = Conv2D(128, 7, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv5)

# pre skip output layer
pre_outputs = Conv2D(1, 7, activation=None, padding='same',
                     kernel_initializer='he_normal')(conv6)
                 

# power exponential outut layer
pred_mean = Lambda(lambda x: x, name='predicted_mean')(pre_outputs[0])
pred_variance = Lambda(lambda x: x, name='predicted_variance')(pre_outputs[1])

nllpe_loss = Lambda(lambda x: K.mean(NLLPE(x[0], x[1], x[2])), name='nllpe_loss')(
    [target, pred_mean, pred_variance]
)


# # # # # Average Pooling ConvNet Model # # # # #
model = Model(
    inputs=[input_mean, input_variance, target],
    outputs=[pred_mean, pred_variance],
)

model.add_loss(nllpe_loss)


# dummy loss adding nothing, as true loss is computed in nlld_loss
def dummyloss(y_true, y_pred):
    return 0.0*y_true


model.compile(
    # optimizer=SGD(INIT_LEARNING_RATE),
    optimizer=Adam(INIT_LEARNING_RATE),
    loss={
        'predicted_mean': dummyloss,
    },
    metrics={
        'predicted_mean': [MSE, MAE],
    },
)

model.summary()

# # # # # Report GPU status after building the model # # # # #
os.system('nvidia-smi')

# # # # # Training Phase # # # # #
model.fit(
    traindata,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valdata,
    validation_steps=val_steps,
    verbose=2,
    callbacks=[
        ModelCheckpoint(
            filepath=CKPTSDIR+'/convnet-model.{epoch:03d}-{val_loss:.2e}.hdf5',
            period=2
        ),
        TensorBoard(
            log_dir=LOGDIR
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.55,
            patience=5,
            min_lr=1e-8,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
        ),
        TerminateOnNaN(),
    ],
)

# # # # # Save final model # # # # #
model.save(os.path.join(CKPTSDIR, 'convnet-model-final.hdf5'))
