import numpy as np
import tensorflow as tf


def laplace_log_likelihood_loss(y_true, y_pred):
    uncertainty_clipped = tf.maximum(y_pred[:, 1:2] * 1000.0, 70)
    prediction = y_pred[:, :1]
    delta = tf.minimum(tf.abs(y_true - prediction), 1000.0)
    metric = -np.sqrt(2.0) * delta / uncertainty_clipped - tf.math.log(np.sqrt(2.0) * uncertainty_clipped)
    return -tf.reduce_mean(metric)
