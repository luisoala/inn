''' Moment propagating extensions of available losses from
    tensorflow/python/keras/losses.py
    
    The losses have no direct corresponding standard version. Instead of 
    a prediction and target they expect a target tensor and two prediction
    tensors as input (one mean and one variance tensor). These describe a
    predicted probability distribution. Losses are then derived as the 
    negative log-likelihood of the target under the predicted distribution.
    
    Because of the additional input tensor the probabilistic losses can
    not be used as standard losses in a Keras model. They have to be added
    as individual loss layers (see below for examples).
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops


def nll_gaussian(y_true, y_mean, y_variance):
    """Negative log-likelihood for Gaussian output units.

    Gaussian output units predict a Gaussian distribution for the output, by
    predicting a mean and variance for the output, instead of giving a point
    estimate. Used for regression type problems.

    This loss computes the negative log-likelihood of the true target tensor
    under the predicted Gaussian distribution.

    Important: This loss can not be directly compiled into a model,
    as model loss functions require exactly two inputs (y_true, y_pred).
    To use this loss it has to be wrapped in Lambda layer with access to the
    target tensor, which can be added to the model via add_loss(). Then a
    zero dummy loss can be passed to the model compilation instead.

    Example:

    ```python
        # define model layers
        some_input = Input(...)
        target = Input(...)
        ...
        mean_tensor = ...
        variance_tensor = ...

        # define loss given a target from within the model
        loss = Lambda(lambda x: nll_gaussian(x[0], x[1], x[2]))(
            [target, mean_tensor, variance_tensor]
        )

        # define the model using functional API
        model = Model(inputs=[some_input, target], outputs=mean_tensor)
        
        # add loss and define dummy loss
        mode.add_loss(loss)
        def dummyloss(y_true, y_pred):
            return 0.0*y_true

        # pass dummy loss to the model compilation
        model.compile(optimizer, loss={'some_input': dummyloss})
    ```

    See [Gast & Roth, 2018, Lightweight probabilistic deep networks] for more
    details on probabilistic output units and probabilistic losses.

    Arguments:
        y_true: tensor of true targets.
        y_mean: tensors of predicted mean.
        y_variance: tensors of predicted variance.

    Returns:
        Tensor with one scalar loss entry per sample.

    """
    # add small amount to variances to avoid numerical instabilities
    y_variance = y_variance + 1e-4
    return K.mean(
        math_ops.square(y_true-y_mean)/(2*y_variance)
        +1/2*math_ops.log(y_variance),
        axis=-1
    )


def nll_power_exponential(y_true, y_mean, y_variance, k=0.5):
    """Negative log-likelihood for generalized power exponential output units.

    Generalized power exponential output units predict a multivariate power
    exponential distribution for the output, by predicting a mean and variance
    instead of giving a point estimate. Used for regression type problems.

    This loss computes the negative log-likelihood of the true target tensor
    under the predicted distribution.

    Important: This loss can not be directly compiled into a model,
    as model loss functions require exactly two inputs (y_true, y_pred).
    To use this loss it has to be wrapped in Lambda layer with access to the
    target tensor, which can be added to the model via add_loss(). Then a
    zero dummy loss can be passed to the model compilation instead.

    Example:

    ```python
        # define model layers
        some_input = Input(...)
        target = Input(...)
        ...
        mean_tensor = ...
        variance_tensor = ...

        # define loss given a target from within the model
        loss = Lambda(lambda x: nll_power_exponential(x[0], x[1], x[2]))(
            [target, mean_tensor, variance_tensor]
        )

        # define the model using functional API
        model = Model(inputs=[some_input, target], outputs=mean_tensor)
        
        # add loss and define dummy loss
        mode.add_loss(loss)
        def dummyloss(y_true, y_pred):
            return 0.0*y_true

        # pass dummy loss to the model compilation
        model.compile(optimizer, loss={'some_input': dummyloss})
    ```

    See [Gomez et al., 1996, A multivariate generalization of the power
    exponential family of distributions] and [Gast & Roth, 2018, Lightweight
    probabilistic deep networks] for more details.

    Arguments:
        y_true: tensor of true targets.
        y_mean: tensors of predicted mean.
        y_variance: tensors of predicted variance.
        k: positive scalar specifying the power for the power exponential
            distribution (Defaults to 1/2, in which case the power exponential
            dsitribution is a Laplace distribution)

    Returns:
        Tensor with one scalar loss entry per sample.

    """
    # add small amount to variances to avoid numerical instabilities
    y_variance = y_variance + 1e-4
    return (
        1/2*tf.pow(
            K.sum(math_ops.square(y_true-y_mean)/y_variance, axis=-1),
            k,
        )
        + 1/2*K.sum(math_ops.log(y_variance), axis=-1)
    )


def nll_dirichlet(y_true, y_mean, y_variance, c1=1e-2, c2=1e-1):
    """Negative log-likelihood for Dirichlet categorical output units.

    Dirichlet output units predict a multivariate Dirichlet distribution
    for the output, by predicting the corresponding concentration parameters
    instead of giving a point estimate. Used for classification type problems.
    Inputs y_true and y_mean should be simplex vectors, e.g. one-hot encodings
    or outputs of a softmax activation.

    This loss computes the negative log-likelihood of the true target tensor
    under the predicted distribution.

    Important: This loss can not be directly compiled into a model,
    as model loss functions require exactly two inputs (y_true, y_pred).
    To use this loss it has to be wrapped in Lambda layer with access to the
    target tensor, which can be added to the model via add_loss(). Then a
    zero dummy loss can be passed to the model compilation instead.

    Example:

    ```python
        # define model layers
        some_input = Input(...)
        target = Input(...)
        ...
        mean_tensor = ...
        variance_tensor = ...

        # define loss given a target from within the model
        loss = Lambda(lambda x: nll_dirichlet(x[0], x[1], x[2]))(
            [target, mean_tensor, variance_tensor]
        )

        # define the model using functional API
        model = Model(inputs=[some_input, target], outputs=mean_tensor)
        
        # add loss and define dummy loss
        mode.add_loss(loss)
        def dummyloss(y_true, y_pred):
            return 0.0*y_true

        # pass dummy loss to the model compilation
        model.compile(optimizer, loss={'some_input': dummyloss})
    ```

    See [Gast & Roth, 2018, Lightweight probabilistic deep networks]
    for more details.

    Arguments:
        y_true: tensor of true targets.
        y_mean: tensors of predicted mean.
        y_variance: tensors of predicted variance.
        c1, c2: constants used to estimate Dirichlet concentration parameter.
            c1 controls how 'peaky' the distribution can become, and c2
            controls the amplification of variances.

    Returns:
        Tensor with one scalar loss entry per sample.

    """
    # add small amount to variances to avoid numerical instabilities
    y_variance = y_variance + 1e-3

    # Laplace smoothing of true targets to avoid log(0) terms
    y_true = (y_true + 1e-3)
    y_true = y_true / K.sum(y_true, axis=-1, keepdims=True)

    # estimate dirichlet parameters
    scale = c1 + c2*K.sqrt(K.sum(y_mean*y_variance, axis=-1, keepdims=True))
    alpha = y_mean / scale

    return (
        K.sum(tf.lgamma(alpha), axis=-1) - tf.lgamma(K.sum(alpha, axis=-1))
        + K.sum((1-alpha)*K.log(y_true), axis=-1)
    )


# Aliases.
nllg = NLLG = nll_gaussian
nllpe = NLLPE = nll_power_exponential
nlld = NLLD = nll_dirichlet


def serialize(loss):
    return loss.__name__


def deserialize(name, custom_objects=None):
    if custom_objects and name in custom_objects:
        fn = custom_objects.get(name)
    else:
        fn = globals().get(name)
        if fn is None:
            raise ValueError('Unknown loss function: ' + name)
    return fn


def get(identifier):
    print(identifier)
    print(isinstance(identifier, six.string_types))
    print(callable(identifier))
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret loss function identifier:',
                         identifier)
