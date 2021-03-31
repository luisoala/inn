import keras.backend as K
import numpy as np
from keras import activations, constraints, initializers, regularizers
from keras.engine.base_layer import Layer


class Dropout(Layer):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = min(1.0, max(0.0, rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if isinstance(inputs, list):
            raise ValueError("Hilfe bei get noise shape")
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0.0 < self.rate < 1.0:
            if self.seed is None:
                seed = np.random.randint(10e6)
                self.seed = seed
            else:
                seed = self.seed

            if isinstance(inputs, list):
                noise_shape = self._get_noise_shape(inputs[0])
                dropped_inputs = [
                    K.dropout(a, self.rate, noise_shape, seed=seed)
                    for a in inputs
                ]
                return [
                    K.in_train_phase(a, b, training=training)
                    for a, b in zip(dropped_inputs, inputs)
                ]
            else:
                noise_shape = self._get_noise_shape(inputs)
                dropped_inputs = K.dropout(
                    inputs, self.rate, noise_shape, seed=seed
                )
                return K.in_train_phase(
                    dropped_inputs, inputs, training=training
                )
        return inputs

    def get_config(self):
        config = {
            "rate": self.rate,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        }
        base_config = super(Dropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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
    # Example
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
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        minmax_kernel_regularizer=None,
        bias_regularizer=None,
        minmax_bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.minmax_kernel_regularizer = regularizers.get(
            minmax_kernel_regularizer
        )
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.minmax_bias_regularizer = regularizers.get(
            minmax_bias_regularizer
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.min_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="zeros",
            name="min_kernel",
            regularizer=self.minmax_kernel_regularizer,
            constraint=constraints.get("nonneg"),
            trainable=False,
        )

        self.max_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="zeros",
            name="max_kernel",
            regularizer=self.minmax_kernel_regularizer,
            constraint=constraints.get("nonneg"),
            trainable=False,
        )

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
            self.min_bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                name="min_bias",
                regularizer=self.minmax_bias_regularizer,
                constraint=constraints.get("nonneg"),
                trainable=False,
            )

            self.max_bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                name="max_bias",
                regularizer=self.minmax_bias_regularizer,
                constraint=constraints.get("nonneg"),
                trainable=False,
            )

            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.min_bias = None
            self.max_bias = None
            self.bias = None
        self.built = True

    def call(self, inputs):
        def _both_call(inputs,):
            x, lo, hi = inputs
            x = K.maximum(x, 0.0)
            lo = K.maximum(lo, 0.0)
            hi = K.maximum(hi, 0.0)
            lo_out = K.dot(
                lo, K.maximum(self.kernel - self.min_kernel, 0.0)
            ) + K.dot(hi, K.minimum(self.kernel - self.min_kernel, 0.0))
            hi_out = K.dot(
                lo, K.minimum(self.kernel + self.max_kernel, 0.0)
            ) + K.dot(hi, K.maximum(self.kernel + self.max_kernel, 0.0))
            output = K.dot(x, self.kernel)
            if self.use_bias:
                lo_out = K.bias_add(
                    lo_out,
                    self.bias - self.min_bias,
                    data_format="channels_last",
                )
                hi_out = K.bias_add(
                    hi_out,
                    self.bias + self.max_bias,
                    data_format="channels_last",
                )
                output = K.bias_add(
                    output, self.bias, data_format="channels_last"
                )
            if self.activation is not None:
                lo_out = self.activation(lo_out)
                hi_out = self.activation(hi_out)
                output = self.activation(output)
            output = [output, lo_out, hi_out]
            return output

        def _interval_call(inputs):
            lo, hi = inputs
            lo = K.maximum(lo, 0.0)
            hi = K.maximum(hi, 0.0)
            lo_out = K.dot(
                lo, K.maximum(self.kernel - self.min_kernel, 0.0)
            ) + K.dot(hi, K.minimum(self.kernel - self.min_kernel, 0.0))
            hi_out = K.dot(
                lo, K.minimum(self.kernel + self.max_kernel, 0.0)
            ) + K.dot(hi, K.maximum(self.kernel + self.max_kernel, 0.0))

            if self.use_bias:
                lo_out = K.bias_add(
                    lo_out,
                    self.bias - self.min_bias,
                    data_format="channels_last",
                )
                hi_out = K.bias_add(
                    hi_out,
                    self.bias + self.max_bias,
                    data_format="channels_last",
                )
            if self.activation is not None:
                lo_out = self.activation(lo_out)
                hi_out = self.activation(hi_out)
            output = [lo_out, hi_out]
            return output

        def _single_call(inputs):
            x = inputs
            x = K.maximum(x, 0.0)
            output = K.dot(x, self.kernel)
            if self.use_bias:
                output = K.bias_add(
                    output, self.bias, data_format="channels_last"
                )
            if self.activation is not None:
                output = self.activation(output)
            return output

        if isinstance(inputs, list):
            if len(inputs) == 2:
                return _interval_call(inputs)
            elif len(inputs) == 3:
                return _both_call(inputs)
        else:
            return _single_call(inputs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            if len(input_shape) == 2:
                assert input_shape[0] and len(input_shape[0]) >= 2
                assert input_shape[1] and len(input_shape[1]) >= 2
                assert input_shape[0][-1]
                assert input_shape[1][-1]
                output_shape = [list(a) for a in input_shape]
                output_shape[0][-1] = self.units
                output_shape[1][-1] = self.units
            elif len(input_shape) == 3:
                assert input_shape[0] and len(input_shape[0]) >= 2
                assert input_shape[1] and len(input_shape[1]) >= 2
                assert input_shape[2] and len(input_shape[2]) >= 2
                assert input_shape[0][-1]
                assert input_shape[1][-1]
                assert input_shape[2][-1]
                output_shape = [list(a) for a in input_shape]
                output_shape[0][-1] = self.units
                output_shape[1][-1] = self.units
                output_shape[2][-1] = self.units
            return [tuple(a) for a in output_shape]
        else:
            assert input_shape and len(input_shape) >= 2
            assert input_shape[-1]
            output_shape = list(input_shape)
            output_shape[-1] = self.units
            return tuple(output_shape)

    def set_minmax_trainable(self):
        if self.use_bias:
            self.trainable_weights = [
                self.min_kernel,
                self.max_kernel,
                self.min_bias,
                self.max_bias,
            ]
            self.non_trainable_weights = [self.kernel, self.bias]
        else:
            self.trainable_weights = [self.min_kernel, self.max_kernel]
            self.non_trainable_weights = [self.kernel]

    def set_single_trainable(self):
        if self.use_bias:
            self.non_trainable_weights = [
                self.min_kernel,
                self.max_kernel,
                self.min_bias,
                self.max_bias,
            ]
            self.trainable_weights = [self.kernel, self.bias]
        else:
            self.non_trainable_weights = [self.min_kernel, self.max_kernel]
            self.trainable_weights = [self.kernel]

    def reset_minmax(self):
        K.set_value(
            self.min_kernel, np.zeros_like(K.get_value(self.min_kernel))
        )
        K.set_value(
            self.max_kernel, np.zeros_like(K.get_value(self.max_kernel))
        )
        if self.use_bias:
            K.set_value(
                self.min_bias, np.zeros_like(K.get_value(self.min_bias))
            )
            K.set_value(
                self.max_bias, np.zeros_like(K.get_value(self.max_bias))
            )

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "minmax_kernel_regularizer": regularizers.serialize(
                self.minmax_kernel_regularizer
            ),
            "minmax_bias_regularizer": regularizers.serialize(
                self.minmax_bias_regularizer
            ),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            The purpose of this argument is to preserve weight
            ordering when switching a model from one data format
            to another.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Example
    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(3, 32, 32), padding='same',))
        # now: model.output_shape == (None, 64, 32, 32)
        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        def _compute_output_shape(input_shape):
            if not all(input_shape[1:]):
                raise ValueError(
                    'The shape of the input to "Flatten" '
                    "is not fully defined "
                    "(got " + str(input_shape[1:]) + ". "
                    'Make sure to pass a complete "input_shape" '
                    'or "batch_input_shape" argument to the first '
                    "layer in your model."
                )
            return (input_shape[0], np.prod(input_shape[1:]))

        if isinstance(input_shape, list):
            return [_compute_output_shape(a) for a in input_shape]
        else:
            return _compute_output_shape(input_shape)

    def call(self, inputs):
        def _single_call(inputs):
            if self.data_format == "channels_first":
                # Ensure works for any dim
                permutation = [0]
                permutation.extend([i for i in range(2, K.ndim(inputs))])
                permutation.append(1)
                inputs = K.permute_dimensions(inputs, permutation)

            return K.batch_flatten(inputs)

        if isinstance(inputs, list):
            return [_single_call(a) for a in inputs]
        else:
            return _single_call(inputs)

    def get_config(self):
        config = {"data_format": self.data_format}
        base_config = super(Flatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
