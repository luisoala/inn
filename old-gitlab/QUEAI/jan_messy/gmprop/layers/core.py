''' Moment propagating versions of several layers from
    tensorflow/python/keras/layers/core.py
    
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

import copy
import numpy as np

from gmprop import activations

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn


class Reshape(Layer):
    """Reshapes an output to a certain shape.

    Arguments:
        target_shape: target shape. Tuple of integers,
            does not include the samples dimension (batch size).

    Input shape:
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        `(batch_size,) + target_shape`

    Example:

    ```python
        # as first layer in a Sequential model
        model = Sequential()
        model.add(Reshape((3, 4), input_shape=(12,)))
        # now: model.output_shape == (None, 3, 4)
        # note: `None` is the batch dimension

        # as intermediate layer in a Sequential model
        model.add(Reshape((6, 2)))
        # now: model.output_shape == (None, 6, 2)

        # also supports shape inference using `-1` as dimension
        model.add(Reshape((-1, 2, 2)))
        # now: model.output_shape == (None, 3, 2, 2)
    ```

    """

    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        Arguments:
            input_shape: shape of array being reshaped
            output_shape: desired shape of the array with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.

        Returns:
            The new output shape with a -1 replaced with its computed value.

            Raises a ValueError if the total array size of the output_shape is
            different then the input_shape, or more than one unknown dimension
            is specified.

        Raises:
            ValueError: in case of invalid values
                for `input_shape` or `input_shape`.

        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)
        return output_shape

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        output_shape = [[], []]
        if None in input_shape[0][1:]:
            output_shape[0] = [input_shape[0][0]]
            # input shape (partially) unknown? replace -1's with None's
            output_shape[0] += tuple(s if s != -1
                                     else None for s in self.target_shape)
        else:
            output_shape[0] = [input_shape[0][0]]
            output_shape[0] += self._fix_unknown_dimension(input_shape[0][1:],
                                                           self.target_shape)
        if None in input_shape[1][1:]:
            output_shape[1] = [input_shape[1][0]]
            # input shape (partially) unknown? replace -1's with None's
            output_shape[1] += tuple(s if s != -1
                                     else None for s in self.target_shape)
        else:
            output_shape[1] = [input_shape[1][0]]
            output_shape[1] += self._fix_unknown_dimension(input_shape[1][1:],
                                                           self.target_shape)
        return [
            tensor_shape.TensorShape(output_shape[0]),
            tensor_shape.TensorShape(output_shape[1]),
        ]

    def call(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        return [
            array_ops.reshape(
                inputs[0],
                (array_ops.shape(inputs[0])[0],) + self.target_shape
            ),
            array_ops.reshape(
                inputs[1],
                (array_ops.shape(inputs[1])[0],) + self.target_shape
            ),
        ]

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful for e.g. connecting RNNs and convnets together.

    Example:

    ```python
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        # now: model.output_shape == (None, 64, 10)
        # note: `None` is the batch dimension
    ```

    Arguments:
        dims: Tuple of integers. Permutation pattern, does not include the
            samples dimension. Indexing starts at 1.
            For instance, `(2, 1)` permutes the first and second dimension
            of the input.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.

    """

    def __init__(self, dims, **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.dims = tuple(dims)
        self.input_spec = [
            InputSpec(ndim=len(self.dims) + 1),
            InputSpec(ndim=len(self.dims) + 1),
        ]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = [input_shape[0][dim], input_shape[1][dim]]
            output_shape[0][i + 1] = target_dim[0]
            output_shape[1][i + 1] = target_dim[1]
        return [
          tensor_shape.TensorShape(output_shape[0]),
          tensor_shape.TensorShape(output_shape[1]),
        ]

    def call(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        return [
            array_ops.transpose(inputs[0], perm=(0,) + self.dims),
            array_ops.transpose(inputs[1], perm=(0,) + self.dims),
        ]

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(Permute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    Arguments:
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    Example:

    ```python
        model = Sequential()
        model.add(Convolution2D(64, 3, 3,
                                border_mode='same',
                                input_shape=(3, 32, 32)))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```

    """

    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]

    def call(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        if self.data_format == 'channels_first':
            permutation = [[0], [0]]
            permutation[0].extend([i for i in
                                  range(2, K.ndim(inputs[0]))])
            permutation[0].append(1)
            permutation[1].extend([i for i in
                                  range(2, K.ndim(inputs[1]))])
            permutation[1].append(1)
            inputs[0] = array_ops.transpose(inputs[0], perm=permutation[0])
            inputs[1] = array_ops.transpose(inputs[1], perm=permutation[1])
        outputs = []
        outputs[0] = array_ops.reshape(inputs[0],
                                       (array_ops.shape(inputs[0])[0], -1))
        outputs[1] = array_ops.reshape(inputs[1],
                                       (array_ops.shape(inputs[1])[0], -1))
        if not context.executing_eagerly():
            outputs[0].set_shape(
                self.compute_output_shape(inputs.get_shape())[0]
            )
            outputs[1].set_shape(
                self.compute_output_shape(inputs.get_shape())[1]
            )
        return outputs

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        input_shape[0] = tensor_shape.TensorShape(input_shape[0]).as_list()
        input_shape[1] = tensor_shape.TensorShape(input_shape[1]).as_list()
        output_shape = [[input_shape[0][0]], [input_shape[1][0]]]
        if all(input_shape[0][1:]):
            output_shape[0] += [np.prod(input_shape[0][1:])]
        else:
            output_shape[0] += [None]
        if all(input_shape[1][1:]):
            output_shape[1] += [np.prod(input_shape[1][1:])]
        else:
            output_shape[1] += [None]
        return [
            tensor_shape.TensorShape(output_shape[0]),
            tensor_shape.TensorShape(output_shape[1]),
        ]

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Example:

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        nD tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`.

    """

    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Dense, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs
        )
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        input_shape[0] = tensor_shape.TensorShape(input_shape[0])
        input_shape[1] = tensor_shape.TensorShape(input_shape[1])
        assert input_shape[0].is_compatible_with(input_shape[1])
        if (input_shape[0][-1].value is None
                or input_shape[1][-1].value is None):
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = input_shape[0][-1].value
        self.input_spec = [
            InputSpec(min_ndim=2, axes={-1: last_dim}),
            InputSpec(min_ndim=2, axes={-1: last_dim}),
        ]
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2
        assert inputs[0].shape.is_compatible_with(inputs[1].shape)
        means, variances = inputs
        means = ops.convert_to_tensor(means)
        variances = ops.convert_to_tensor(variances)
        shape = means.get_shape().as_list()
        rank = len(shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outmeans = standard_ops.tensordot(
                means,
                self.kernel,
                [[rank - 1], [0]]
            )
            outvariances = standard_ops.tensordot(
                variances,
                K.square(self.kernel),
                [[rank - 1], [0]]
            )
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                output_shape = shape[:-1] + [self.units]
                outmeans.set_shape(output_shape)
                outvariances.set_shape(output_shape)
        else:
            outmeans = gen_math_ops.mat_mul(means, self.kernel)
            outvariances = gen_math_ops.mat_mul(variances,
                                                K.square(self.kernel))
        if self.use_bias:
            outmeans = nn.bias_add(outmeans, self.bias)
        if self.activation is not None:
            return self.activation([outmeans, outvariances])
        return [outmeans, outvariances]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        input_shape[0] = tensor_shape.TensorShape(input_shape[0])
        input_shape[1] = tensor_shape.TensorShape(input_shape[1])
        input_shape[0] = input_shape[0].with_rank_at_least(2)
        input_shape[1] = input_shape[1].with_rank_at_least(2)
        assert input_shape[0].is_compatible_with(input_shape[1])
        if input_shape[0][-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, '
                'but saw: %s' % input_shape[0])
        if input_shape[1][-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, '
                'but saw: %s' % input_shape[1])
        return [
            input_shape[0][:-1].concatenate(self.units),
            input_shape[1][:-1].concatenate(self.units),
        ]

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
