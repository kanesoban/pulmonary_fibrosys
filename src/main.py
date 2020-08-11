import argparse
import os
from datetime import datetime
import random
import yaml


from tqdm import tqdm
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

from metrics import laplace_log_likelihood


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train_dataset, val_dataset = get_data(config['input_file'], config['batch_size'])

    model = get_model()

    log_dir = os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tensorboard_callback]

    model.fit(train_dataset, epochs=config['epochs'], validation_data=val_dataset, callbacks=callbacks)


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    metrics = [laplace_log_likelihood]

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=metrics)
    return model


def get_data(input_file, batch_size):
    train_data = pd.read_csv(input_file)
    min_week = train_data['Weeks'].min()
    train_data['Weeks'] = train_data['Weeks'] - min_week
    max_week = train_data['Weeks'].max()
    train_data['Weeks'] /= max_week
    max_FVC = train_data['FVC'].max()
    train_data['FVC'] /= max_FVC
    grouped = train_data.groupby(train_data.Patient)
    patient_dfs = []
    for patient in tqdm(train_data['Patient']):
        patient_df = grouped.get_group(patient)

        FVC_next = patient_df['FVC'].iloc[1:]
        FVC_next = FVC_next.to_list()

        # This will only work until i am not using
        weeks = patient_df['Weeks']

        Weeks_next = weeks.iloc[1:]
        Weeks_next = Weeks_next.to_list()

        converted_data = {'Weeks': weeks.iloc[:-1], 'FVC': patient_df['FVC'].iloc[:-1], 'FVC_next': FVC_next,
                          'Weeks_next': Weeks_next}
        converted_df = pd.DataFrame.from_dict(converted_data)

        patient_dfs.append(converted_df)
    result = pd.concat(patient_dfs)
    num_data = len(result)
    indexes = list(range(num_data))
    random.shuffle(indexes)
    data = result.iloc[indexes]
    split = int(num_data * 0.8)
    training_data = data.iloc[:split]
    val_data = data.iloc[split:]
    input_columns = ['Weeks', 'FVC', 'Weeks_next']
    target_column = ['FVC_next']
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(training_data[input_columns], tf.float32),
                tf.cast(training_data[target_column], tf.float32)
            )
        )
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(val_data[input_columns], tf.float32),
                tf.cast(val_data[target_column], tf.float32)
            )
        )
    )
    return train_dataset.batch(batch_size), val_dataset.batch(batch_size)


if __name__ == "__main__":
    main()
