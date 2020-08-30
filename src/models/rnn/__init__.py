import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.dense.metrics import laplace_log_likelihood
from models.dense.losses import laplace_log_likelihood_loss
import models.cnn3d as cnn3d


def get_combined_model(sequence_length, learning_rate, width, height, depth):
    rnn_inputs = tf.keras.Input(shape=(sequence_length, 2))
    x = tf.keras.layers.Masking(mask_value=-1, input_shape=(sequence_length, 1))(rnn_inputs)
    rnn_out = tf.keras.layers.GRU(4)(x)
    rnn_out = tf.keras.layers.Reshape((-1,))(rnn_out)

    cnn3d_inputs, cnn3d_out = cnn3d.get_model(width, height, depth)
    cnn3d_out = tf.keras.layers.Reshape((-1,))(cnn3d_out)

    combined_out = tf.keras.layers.concatenate([rnn_out, cnn3d_out])

    prediction_output = tf.keras.layers.Dense(1)(combined_out)
    uncertainty_output = tf.keras.layers.Dense(1, activation='sigmoid')(combined_out)

    outputs = tf.keras.layers.concatenate([prediction_output, uncertainty_output])

    model = tf.keras.Model(inputs=[rnn_inputs, cnn3d_inputs], outputs=outputs)

    metrics = [laplace_log_likelihood]

    model.compile(loss=laplace_log_likelihood_loss, optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)

    return model


def get_model(sequence_length, learning_rate):
    inputs = tf.keras.Input(shape=(sequence_length, 2))
    x = tf.keras.layers.Masking(mask_value=-1, input_shape=(sequence_length, 1))(inputs)
    x = tf.keras.layers.GRU(4)(x)

    prediction_output = tf.keras.layers.Dense(1)(x)
    uncertainty_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    outputs = tf.keras.layers.concatenate([prediction_output, uncertainty_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    metrics = [laplace_log_likelihood]

    model.compile(loss=laplace_log_likelihood_loss, optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=metrics)

    return model


