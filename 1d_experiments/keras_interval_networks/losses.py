import keras.backend as K

def lower_interval_mean_squared_error(beta=1e-1):
    def _lower_interval_mean_squared_error(y_true, y_pred):
        mse = K.mean(K.square(K.maximum(y_pred - y_true, 0.0)), axis=-1)
        reg = -K.mean(y_pred, axis=-1)
        return mse + beta*reg
    return _lower_interval_mean_squared_error


def upper_interval_mean_squared_error(beta=1e-1):
    def _upper_interval_mean_squared_error(y_true, y_pred):
        mse = K.mean(K.square(K.maximum(y_true - y_pred, 0.0)), axis=-1)
        reg = K.mean(y_pred, axis=-1)
        return mse + beta*reg
    return _upper_interval_mean_squared_error

LIMSE = lower_interval_mean_squared_error
UIMSE = upper_interval_mean_squared_error
