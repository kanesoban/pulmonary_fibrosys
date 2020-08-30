import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


def get_model(width, height, depth):
    # Do we have to specify channels ?
    inputs = tf.keras.Input(shape=(width, height, depth))
    x = tf.keras.layers.Conv3D(32, kernel_size=(5, 5, 5), activation='relu')(inputs)
    x = tf.keras.layers.MaxPool3D()(x)

    x = tf.keras.layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPool3D()(x)

    return inputs, x
