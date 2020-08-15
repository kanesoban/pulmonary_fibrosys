import os
from collections import Counter

import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.dense.metrics import laplace_log_likelihood
from models.dense.losses import laplace_log_likelihood_loss


def get_model(sequence_length):
    inputs = tf.keras.Input(shape=(sequence_length, 1))
    x = tf.keras.layers.Masking(mask_value=-1, input_shape=(sequence_length, 1))(inputs)
    x = tf.keras.layers.LSTM(4)(x)
    prediction_output = tf.keras.layers.Dense(1)(x)
    uncertainty_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    outputs = tf.keras.layers.concatenate([prediction_output, uncertainty_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    metrics = [laplace_log_likelihood]

    model.compile(loss=laplace_log_likelihood_loss, optimizer=tf.optimizers.Adam(), metrics=metrics)

    return model


def get_data(input_file, batch_size, sequence_length, data_dir=None, split=0.8):
    saved_data_path = os.path.join(data_dir, 'data.npy')

    if os.path.isfile(saved_data_path):
        with open(saved_data_path, 'rb') as f:
            data = np.load(f)
            targets = np.load(f)
    else:
        data, targets = process_data(input_file, sequence_length, saved_data_path)

    n_data = data.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(data, tf.float32),
            tf.cast(targets, tf.float32)
        )
    ).shuffle(n_data)

    train_size = int(split * n_data)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset.batch(batch_size), val_dataset.batch(batch_size)


def process_data(input_file, sequence_length, saved_data_path):
    train_data = pd.read_csv(input_file)
    grouped = train_data.groupby(train_data.Patient)
    data = []
    targets = []
    for patient in tqdm(train_data['Patient'].unique()):
        patient_df = grouped.get_group(patient)
        week_counts = Counter(patient_df['Weeks'])
        for week in week_counts:
            if week_counts[week] > 1:
                patient_df = aggregate_duplicate_weeks(patient_df, week)
        patient_df = patient_df.set_index('Weeks')
        min_idx = int(min(patient_df.index))
        max_idx = int(max(patient_df.index))
        indexes = sorted(list(patient_df.index))
        missing_indexes = list(set(list(range(int(min_idx), int(max_idx + 1)))) - set(patient_df.index))
        for idx in missing_indexes:
            patient_df.loc[idx] = -1
        patient_df = patient_df.sort_index()

        patient_data = []
        patient_targets = []
        for idx in indexes:
            prev_indexes = sorted(list(range(int(idx - sequence_length), int(idx))))
            if len(set(indexes).intersection(set(prev_indexes))):
                sequence = np.empty(sequence_length)
                for i, prev_idx in enumerate(prev_indexes):
                    if prev_idx in patient_df['FVC'].index:
                        sequence[i] = patient_df['FVC'].loc[prev_idx]
                    else:
                        sequence[i] = -1
                patient_data.append(sequence)
                patient_targets.append(patient_df['FVC'].loc[idx])
        patient_data = np.stack(patient_data).reshape((-1, sequence_length, 1))
        patient_targets = np.stack(patient_targets).reshape((-1, 1))
        data.append(patient_data)
        targets.append(patient_targets)
    data = np.concatenate(data, axis=0)
    targets = np.concatenate(targets, axis=0)

    with open(saved_data_path, 'wb') as f:
        np.save(f, data)
        np.save(f, targets)

    return data, targets


def aggregate_duplicate_weeks(patient_df, week):
    old_rows = patient_df[patient_df['Weeks'] == week]
    fvc = np.mean(old_rows['FVC'])
    percent = np.mean(old_rows['Percent'])
    old_row = old_rows.iloc[0]
    row = [old_row['Patient'], week, fvc, percent, old_row['Age'], old_row['Sex'], old_row['SmokingStatus']]
    patient_df = patient_df.drop(old_rows.index)
    old_index = old_rows.index[0]
    patient_df.loc[old_index, ['Patient', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus']] = row
    return patient_df
