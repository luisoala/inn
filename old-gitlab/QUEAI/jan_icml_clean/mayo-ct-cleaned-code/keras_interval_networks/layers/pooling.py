import keras.backend as K
from keras.engine.base_layer import Layer
from keras.utils import conv_utils


class _Pooling2D(Layer):
    """Abstract class for different pooling 2D layers. """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(_Pooling2D, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, "pool_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        def _compute_output_shape(input_shape):
            if self.data_format == "channels_first":
                rows = input_shape[2]
                cols = input_shape[3]
            elif self.data_format == "channels_last":
                rows = input_shape[1]
                cols = input_shape[2]
            rows = conv_utils.conv_output_length(
                rows, self.pool_size[0], self.padding, self.strides[0]
            )
            cols = conv_utils.conv_output_length(
                cols, self.pool_size[1], self.padding, self.strides[1]
            )
            if self.data_format == "channels_first":
                return (input_shape[0], input_shape[1], rows, cols)
            elif self.data_format == "channels_last":
                return (input_shape[0], rows, cols, input_shape[3])

        if isinstance(input_shape, list):
            return [_compute_output_shape(a) for a in input_shape]
        else:
            return _compute_output_shape(input_shape)

    def _pooling_function(
        self, inputs, pool_size, strides, padding, data_format
    ):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(
            inputs=inputs,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return output

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaxPooling2D(_Pooling2D):
    """Max pooling operation for spatial data.
    # Arguments
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
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super(MaxPooling2D, self).__init__(
            pool_size, strides, padding, data_format, **kwargs
        )

    def _pooling_function(
        self, inputs, pool_size, strides, padding, data_format
    ):

        _pool_func = lambda x: K.pool2d(  # noqa: E731
            x, pool_size, strides, padding, data_format, pool_mode="max"
        )
        if isinstance(inputs, list):
            return [_pool_func(a) for a in inputs]
        else:
            return _pool_func(inputs)
