''' Moment propagating versions of several activation functions from
    tensorflow/python/keras/activations.py
    
    The activations behave like the corresponding standard functions, except
    that they expect a list of two tensors as input (a mean
    and a variance tensor describing an input Gaussian distribution) and 
    return a list of two tensors as output (the mean and variance tensors of 
    the Gaussian distribution that matches the first and second moment of the
    output probability distribution according to the transformation.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops


def softmax(x, axis=-1):
    """Softmax activation function.

    Arguments:
        x: List of mean and variance input tensor.
        axis: Integer, axis along which the softmax normalization is applied.

    Returns:
        List of new mean and variance tensors, according to
        the softmax transformation.

    Raises:
        ValueError: In case `dim(x) == 1`.

    """
    raise NotImplementedError(
        'The softmax activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def elu(x, alpha=1.0):
    """Exponential linear unit.

    Arguments:
        x: List of mean and variance input tensor.
        alpha: A scalar, slope of negative section.

    Returns:
        List of new mean and variance tensors, according to
        the exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.

    Reference:
        - [Fast and Accurate Deep Network Learning by Exponential
          Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

    """
    raise NotImplementedError(
        'The elu activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def selu(x):
    """Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are pre-defined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).

    Arguments:
        x: A tensor or variable to compute the activation function for.

    Returns:
        List of new mean and variance tensors, according to
        the scaled exponential unit activation: `scale * elu(x, alpha)`.

    # Note
        - To be used together with the initialization "lecun_normal".
        - To be used together with the dropout variant "AlphaDropout".

    References:
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

    """
    raise NotImplementedError(
        'The selu activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def softplus(x):
    """Softplus activation function.

    Arguments:
        x: List of mean and variance input tensor.

    Returns:
        List of new mean and variance tensors, according to
        the softplus activation: `log(exp(x) + 1)`.

    """
    raise NotImplementedError(
        'The softplus activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def softsign(x):
    """Softsign activation function.

    Arguments:
        x: List of mean and variance input tensor.

    Returns:
        List of new mean and variance tensors, according to
        the softplus activation: `x / (abs(x) + 1)`.
    """
    raise NotImplementedError(
        'The softsign activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


# private helper function for relu, pdf of standard normal distribution
def _gauss_density(x):
    return 1/np.sqrt(2*np.pi)*K.exp(-K.square(x)/2)


# private helper function for relu, cdf of standard normal distribution
def _gauss_cumulative(x):
    return 1/2*(1+math_ops.erf(x/np.sqrt(2)))


def relu(x, alpha=0., max_value=None, threshold=0):
    """Rectified Linear Unit.

    With default values, it returns element-wise `max(x, 0)`.

    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    Arguments:
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: float. Saturation threshold.
        threshold: float. Threshold value for thresholded activation.

    Returns:
        List of new mean and variance tensors, according to
        the rectified linear unit activation.

    """
    if max_value is not None:
        raise NotImplementedError(
            'The relu activation function with max_value other than None '
            'has not been implemented for Gaussian moment propagation yet.'
        )
    if not threshold == 0.0:
        raise NotImplementedError(
            'The relu activation function with threshold other than 0.0 '
            'has not been implemented for Gaussian moment propagation yet.'
        )
    if not isinstance(x, list) and len(x) == 2:
        raise ValueError('The relu activation function expects a list of '
                         'exactly two input tensors, but got: %s' % x)
    assert x[0].shape.is_compatible_with(x[1].shape)
    mean, variance = x
    # variance = variance + K.epsilon()
    variance = variance + 1e-4
    std = K.sqrt(variance)

    div = mean/std
    gd_div = _gauss_density(div)
    gc_div = _gauss_cumulative(div)
    new_mean = mean*gc_div + std*gd_div
    new_variance = (
        (K.square(mean)+variance)*gc_div + mean*std*gd_div - K.square(new_mean)
    )
    new_variance = K.maximum(0.0, new_variance)
    return new_mean, new_variance


def tanh(x):
    """Tanh activation function.

    Arguments:
        x: List of mean and variance input tensor.

    Returns:
        List of new mean and variance tensors, according to
        the tanh activation: `tanh(x)`.

    """
    raise NotImplementedError(
        'The tanh activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def sigmoid(x):
    """Sigmoid activation function.

    Arguments:
        x: List of mean and variance input tensor.

    Returns:
        List of new mean and variance tensors, according to
        the sigmoid activation: `1 / (1 + exp(-x))`.

    """
    raise NotImplementedError(
        'The sigmoid activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def exponential(x):
    """Exponential activation function.

    Arguments:
       x: List of mean and variance input tensor.

    Returns:
       List of new mean and variance tensors, according to
       the exponential activation: `exp(x)`.

    """
    raise NotImplementedError(
        'The exponential activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def hard_sigmoid(x):
    """Hard sigmoid activation function.

    Faster to compute than sigmoid activation.

    Arguments:
        x: List of mean and variance input tensor.

    Returns:
        List of new mean and variance tensors, according to
        the Hard sigmoid activation:
        - `0` if `x < -2.5`
        - `1` if `x > 2.5`
        - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.

    """
    raise NotImplementedError(
        'The hard sigmoid activation function has not been implemented '
        'for Gaussian moment propagation yet.'
    )


def linear(x):
    """Linear (identity) activation function.

    Arguments:
        x: List of mean and variance input tensor.

    Returns:
        List of new mean and variance tensors, according to
        the linear identity activation: `x`.

    """
    mean, variance = x
    return mean, variance


def serialize(activation):
    return activation.__name__


def deserialize(name, custom_objects=None):
    if custom_objects and name in custom_objects:
        fn = custom_objects.get(name)
    else:
        fn = globals().get(name)
        if fn is None:
            raise ValueError('Unknown activation function: ' + name)
    return fn


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret activation function identifier:',
                         identifier)
