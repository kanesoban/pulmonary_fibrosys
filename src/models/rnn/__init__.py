import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.dense.metrics import laplace_log_likelihood
from models.dense.losses import laplace_log_likelihood_loss


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


