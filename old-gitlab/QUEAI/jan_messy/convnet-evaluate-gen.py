import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dropout

import parsedata

# # # # # Setup parameters # # # # #
NUM_PLOTS = 5
NUM_SAMPLES = 500

# # # # # Generate image data # # # # #
T_SIZE = 10    # size of test set
TN, TS = np.random.randn(T_SIZE, 2), np.random.randn(T_SIZE, 1) 
def test_generator():
    count = 0
    while count < T_SIZE:
        image = parsedata.generate_stripes(20*TN[count, :], 10*TS[count, :])
        masked = parsedata.immask(image)
        masked = np.expand_dims(masked, axis=2)
        image = np.expand_dims(image, axis=2)
        count += 1
        yield masked, image
testdata = tf.data.Dataset.from_generator(
    test_generator,
    (tf.float32, tf.float32),
)        
testdata = testdata.map(
    lambda x, y: parsedata.resize_images(x, y, size=(64, 64))
)
testdata = testdata.batch(NUM_PLOTS)

# # # # # Restore model from checkpoint # # # # #
#model = load_model('./logs/convnet-train-gen/ckpts/convnet-model-final.hdf5')
model = load_model('./logs/convnet-train-gen (no dropout)/ckpts/convnet-model-final.hdf5')
# model = load_model('./logs/convnet-train-gen/ckpts/convnet-model.005-2.05e-01.hdf5')

model.summary()

# # # # # Add dropout layers to model # # # # #
input_layer = Input(batch_shape=model.layers[0].input_shape, name='input')
prev_layer = input_layer
for cur_layer in model.layers[1:-1]:
    if not isinstance(cur_layer, Dropout):
        prev_layer = cur_layer(prev_layer)
        prev_layer = Dropout(0.2)(prev_layer, training=True)
output_layer = model.layers[-1](prev_layer)
dr_model = Model([input_layer], [output_layer])

dr_model.summary()


# # # # Evaluate # # # # #
session = tf.Session()
masked, images = session.run(testdata.make_one_shot_iterator().get_next())
predictions = model.predict(masked, batch_size=NUM_PLOTS)
dr_predictions = np.zeros((NUM_SAMPLES,)+predictions.shape)
for k in range(NUM_SAMPLES):
    dr_predictions[k, ...] = dr_model.predict(masked, batch_size=NUM_PLOTS)
dr_mean_predictions = np.mean(dr_predictions, axis=0)
dr_variance_predictions = np.var(dr_predictions, axis=0)

for i in range(predictions.shape[0]):
    plt.figure()

    plt.subplot(2, 4, 1)
    plt.imshow(masked[i, :, :, 0])
    plt.colorbar()
    plt.title('input')

    plt.subplot(2, 4, 2)
    plt.imshow(images[i, :, :, 0])
    plt.colorbar()
    plt.title('target')

    plt.subplot(2, 4, 3)
    plt.imshow(predictions[i, :, :, 0])
    plt.colorbar()
    plt.title('prediction')

    plt.subplot(2, 4, 4)
    plt.imshow(np.abs(predictions[i, :, :, 0]-images[i, :, :, 0]))
    plt.colorbar()
    plt.title('abs error')

    plt.subplot(2, 4, 5)
    plt.imshow(dr_predictions[0, i, :, :, 0])
    plt.colorbar()
    plt.title('dr prediction')

    plt.subplot(2, 4, 6)
    plt.imshow(np.abs(dr_predictions[0, i, :, :, 0]-images[i, :, :, 0]))
    plt.colorbar()
    plt.title('dr abs error')

    plt.subplot(2, 4, 7)
    plt.imshow(dr_mean_predictions[i, :, :, 0])
    plt.colorbar()
    plt.title('mean dr prediction')

    plt.subplot(2, 4, 8)
    plt.imshow(dr_variance_predictions[i, :, :, 0])
    plt.colorbar()
    plt.title('variance dr prediction')

plt.show()
