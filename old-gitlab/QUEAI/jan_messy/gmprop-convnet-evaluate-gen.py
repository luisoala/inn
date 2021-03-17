import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from gmprop.layers import Conv2D

import parsedata

# # # # # Setup parameters # # # # #
NUM_PLOTS = 5

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
testdata = testdata.map(lambda img, target: (
        {
            'input_mean': img,
            'input_variance': 0.2*np.ones(img.shape),
            'target': target,
        },
        {
            'predicted_mean': target,
        }
    )
)
testdata = testdata.batch(NUM_PLOTS)

# # # # # Restore model from checkpoint # # # # #
custom_objects = {
    'Conv2D': Conv2D,
}

#model = load_model(
#    './logs/gmprop-convnet-train-gen/ckpts/convnet-model-final.hdf5',
#    custom_objects=custom_objects,
#    compile=False
#)
model = load_model(
    './logs/gmprop-convnet-train-gen/ckpts/convnet-model.020--1.44e+00.hdf5',
    custom_objects=custom_objects,
    compile=False
)

# # # # Evaluate # # # # #
session = tf.Session()
indict, outdict = session.run(testdata.make_one_shot_iterator().get_next())
images = indict['input_mean']
targets = indict['target']
pred_means, pred_variances = model.predict(indict, batch_size=NUM_PLOTS)

for i in range(pred_means.shape[0]):
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.imshow(images[i, :, :, 0])
    plt.subplot(3, 2, 2)
    plt.imshow(targets[i, :, :, 0])
    plt.subplot(3, 2, 3)
    plt.imshow(pred_means[i, :, :, 0])
    plt.subplot(3, 2, 4)
    plt.imshow(np.abs(pred_means[i, :, :, 0]-targets[i, :, :, 0]))
    plt.subplot(3, 2, 5)
    plt.imshow(pred_variances[i, :, :, 0])
plt.show()
