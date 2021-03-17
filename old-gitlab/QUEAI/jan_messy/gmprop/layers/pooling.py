''' Moment propagating versions of several layers from
    tensorflow/python/keras/layers/pooling.py
    
    The layers behave like the corresponding standard layers, except
    that their call methods expect a list of two tensors as input (a mean
    and a variance tensor describing an input Gaussian distribution) and 
    return a list of two tensors as output (the mean and variance tensors of 
    the Gaussian distribution that matches the first and second moment of the
    output probability distribution according to the layer transformation.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn


class Pooling2D(Layer):
    """Pooling layer for arbitrary pooling functions,
    for 2D inputs (e.g. images).

    This class only exists for code reuse. It will never be an exposed API.

    Arguments:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool`.
    pool_size: An integer or tuple/list of 2 integers:
        (pool_height, pool_width) specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
    data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.

    """

    def __init__(self, pool_function, pool_size, strides, padding='valid',
                 data_format=None, name=None, **kwargs):
        super(Pooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def call(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        assert inputs[0].shape.is_compatible_with(inputs[1].shape)
        if self.data_format == 'channels_last':
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = [[], []]
        outputs[0] = self.pool_function(
            inputs[0],
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4)
        )
        outputs[1] = self.pool_function(
            inputs[1]/np.prod(pool_shape),
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format, 4)
        )
        return outputs

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        input_shape[0] = tensor_shape.TensorShape(input_shape[0])
        input_shape[1] = tensor_shape.TensorShape(input_shape[1])
        assert input_shape[0].is_compatible_with(input_shape[1])
        input_shape[0] = input_shape[0].as_list()
        input_shape[1] = input_shape[1].as_list()
        if self.data_format == 'channels_first':
            rows = [input_shape[0][2], input_shape[1][2]]
            cols = [input_shape[0][3], input_shape[1][3]]
        else:
            rows = [input_shape[0][1], input_shape[1][1]]
            cols = [input_shape[0][2], input_shape[1][2]]
        rows[0] = conv_utils.conv_output_length(rows[0], self.pool_size[0],
                                                self.padding, self.strides[0])
        rows[1] = conv_utils.conv_output_length(rows[1], self.pool_size[0],
                                                self.padding, self.strides[0])
        cols[0] = conv_utils.conv_output_length(cols[0], self.pool_size[1],
                                                self.padding, self.strides[1])
        cols[1] = conv_utils.conv_output_length(cols[1], self.pool_size[1],
                                                self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return [
                tensor_shape.TensorShape(
                    [input_shape[0][0], input_shape[0][1], rows[0], cols[0]]
                ),
                tensor_shape.TensorShape(
                    [input_shape[1][0], input_shape[1][1], rows[1], cols[1]]
                ),
            ]
        else:
            return [
                tensor_shape.TensorShape(
                    [input_shape[0][0], rows[0], cols[0], input_shape[0][3]]
                ),
                tensor_shape.TensorShape(
                    [input_shape[1][0], rows[1], cols[1], input_shape[1][3]]
                ),
            ]

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'strides': self.strides,
            'data_format': self.data_format
        }
        base_config = super(Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(Pooling2D):
    """Max pooling operation for spatial data.

    Arguments:
        pool_size: integer or tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the input in both spatial dimension.
            If only one integer is specified, the same window length
            will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None.
            Strides values.
            If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    Input shape:
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    Output shape:
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        raise NotImplementedError(
            'MaxPooling2D has not been implemented for Gaussian moment '
            'propagation yet.'
        )
    # super(MaxPooling2D, self).__init__(
    #     nn.max_pool,
    #     pool_size=pool_size, strides=strides,
    #     padding=padding, data_format=data_format, **kwargs)


class AveragePooling2D(Pooling2D):
    """Average pooling operation for spatial data.

    Arguments:
        pool_size: integer or tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the input in both spatial dimension.
            If only one integer is specified, the same window length
            will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None.
            Strides values.
            If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    Input shape:
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`

    Output shape:
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
        4D tensor with shape:
        `(batch_size, channels, pooled_rows, pooled_cols)`
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(AveragePooling2D, self).__init__(
            nn.avg_pool,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )
