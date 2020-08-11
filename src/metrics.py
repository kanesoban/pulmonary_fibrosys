import numpy as np
import tensorflow as tf


def laplace_log_likelihood(y_true, y_pred):
    uncertainty_clipped = 100.0
    delta = tf.minimum(tf.abs(y_true - y_pred), 1000.0)
    metric = -np.sqrt(2.0) * delta / uncertainty_clipped - np.log(np.sqrt(2.0) * uncertainty_clipped)
    return tf.reduce_mean(metric)


class LaplaceLogLikelihood(tf.keras.metrics.Metric):
    def __init__(self, name='laplace_log_likkelihood', **kwargs):
        super(LaplaceLogLikelihood, self).__init__(name=name, **kwargs)
        self.y_true = []
        self.y_pred = []

    def reset_states(self):
        self.y_true = []
        self.y_pred = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def result(self):
        y_true = tf.concat(self.y_true, axis=1)
        y_pred = tf.concat(self.y_pred, axis=1)
        uncertainty_clipped = tf.cast(tf.constant(100), tf.float32)
        delta = tf.minimum(tf.abs(y_true - y_pred), 1000.0)
        metric = -np.sqrt(2.0) * delta / uncertainty_clipped - np.log(np.sqrt(2.0) * uncertainty_clipped)
        return tf.reduce_mean(metric)
