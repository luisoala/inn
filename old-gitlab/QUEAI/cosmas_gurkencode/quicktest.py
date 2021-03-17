from keras.layers import Layer, Input
from keras.models import Model
from keras import backend as K

from keras.optimizers import SGD
from keras.losses import MSE, MAE

from keras_interval_networks.layers import Dense, Conv1D, Conv2D, Conv3D, Dropout

import numpy as np


# Test stuff
N, B = 8, 16
a, b = np.random.randn(B, N, N, 1), 1e-5*np.random.rand(B, N, N, 1)


def get_model_creator():
    layer1 = Conv2D(4, 2, padding='same', activation='relu')
    layer2 = Conv2D(4, 2, padding='same', activation='relu')
    layer3 = Conv2D(1, 2, padding='same')

    def _model(inputs):
        hidden1 = layer1(inputs)
        hidden2 = layer2(hidden1)
        outputs = layer3(Dropout(0.9)(hidden2))
        return Model(inputs, outputs)
    return _model


input1 = Input((N, N, 1))
input2 = Input((N, N, 1))
input3 = Input((N, N, 1))

creator = get_model_creator()

model1 = creator(input1)
model2 = creator([input1, input2])
model3 = creator([input1, input2, input3])

# model1.summary()
# model2.summary()
# model3.summary()


for l in model1.layers:
    if isinstance(l, Conv2D):
        l.set_single_trainable()
model1.compile(
    loss=MSE,
    optimizer=SGD(1e-1),
)

for l in model1.layers:
    if isinstance(l, Conv2D):
        l.set_minmax_trainable()
model2.compile(
    loss=[MAE, MAE],
    optimizer=SGD(1e-1),
)


for l in model1.layers:
    if isinstance(l, Conv2D):
        l.set_single_trainable()
model1.summary()
model1.fit(a, a, epochs=2)

for l in model1.layers:
    if isinstance(l, Conv2D):
        l.set_minmax_trainable()
# model2.compile(
#     loss=[MAE, MAE],
#     optimizer=SGD(1e-1),
# )
model2.summary()
print(K.get_value(model2.layers[-1].max_kernel))
model2.fit([a-b, a+b], [a-b, a+b], epochs=2)

for l in model1.layers:
    if isinstance(l, Conv2D):
        l.set_single_trainable()
# model1.compile(
#     loss=MSE,
#     optimizer=SGD(1e-1),
# )
model1.summary()
model1.fit(a, a, epochs=2)


pred1 = model1.predict(a)
pred2, pred3 = model2.predict([a-b, a+b])
pred4, pred5, pred6 = model3.predict([a, a-b, a+b])

print(np.all(pred2 <= pred3))
print(np.max(pred3 - pred2))


print(K.get_value(model2.layers[-1].max_kernel))
