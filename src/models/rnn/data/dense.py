import os

import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from utils import get_patient_scan

MIN_WEEK = -12
MAX_WEEK = 133


def get_combined_data(input_file, batch_size, sequence_length, scans_root, n_depth, rows, columns, split=0.8):
    train_data = pd.read_csv(input_file)
    n_features = 0
    train_data['Weeks'] = train_data['Weeks']
    n_features += 1
    max_FVC = train_data['FVC'].max()
    train_data['FVC'] /= max_FVC
    n_features += 1
    grouped = train_data.groupby(train_data.Patient)
    n_data = len(train_data)

    def gen():
        for patient in tqdm(train_data['Patient'].unique()):
            patient_df = grouped.get_group(patient)
            FVC = patient_df['FVC'].iloc[:-1].tolist()
            weeks = patient_df['Weeks']
            Weeks = weeks.iloc[:-1]
            Weeks_next = weeks.iloc[1:]
            week_diff = (np.array(Weeks_next.tolist()) - np.array(Weeks.tolist())) / (MAX_WEEK - MIN_WEEK)
            converted_data = {'FVC': FVC, 'week_diff': week_diff}
            converted_df = pd.DataFrame.from_dict(converted_data)
            indexes = sorted(list(converted_df.index))
            patient_dir = os.path.join(scans_root, patient)
            img = np.expand_dims(get_patient_scan(patient_dir, n_depth=n_depth, rows=rows, columns=columns), axis=-1)

            for idx in indexes:
                prev_indexes = sorted(list(range(int(idx - sequence_length), int(idx))))
                if len(set(indexes).intersection(set(prev_indexes))):
                    sequence = np.empty((sequence_length, n_features))
                    for i, prev_idx in enumerate(prev_indexes):
                        if prev_idx in converted_df['FVC'].index:
                            sequence[i] = [converted_df['FVC'].loc[prev_idx], converted_df['week_diff'].loc[prev_idx]]
                        else:
                            sequence[i] = [-1, -1]
                    yield ((sequence, img), converted_df['FVC'].loc[idx])

    dataset = tf.data.Dataset.from_generator(gen, output_types=((tf.float32, tf.float32), tf.float32)).repeat(None).shuffle(n_data)

    train_size = int(split * n_data)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset.batch(batch_size), val_dataset.batch(batch_size)


def get_data(input_file, batch_size, sequence_length, split=0.8):
    train_data = pd.read_csv(input_file)
    n_features = 0
    train_data['Weeks'] = train_data['Weeks']
    n_features += 1
    max_FVC = train_data['FVC'].max()
    train_data['FVC'] /= max_FVC
    n_features += 1
    grouped = train_data.groupby(train_data.Patient)
    data = []
    targets = []

    for patient in tqdm(train_data['Patient'].unique()):
        patient_df = grouped.get_group(patient)
        FVC = patient_df['FVC'].iloc[:-1].tolist()
        weeks = patient_df['Weeks']
        Weeks = weeks.iloc[:-1]
        Weeks_next = weeks.iloc[1:]
        week_diff = (np.array(Weeks_next.tolist()) - np.array(Weeks.tolist())) / (MAX_WEEK - MIN_WEEK)
        converted_data = {'FVC': FVC, 'week_diff': week_diff}
        converted_df = pd.DataFrame.from_dict(converted_data)
        indexes = sorted(list(converted_df.index))

        patient_data = []
        patient_targets = []
        for idx in indexes:
            prev_indexes = sorted(list(range(int(idx - sequence_length), int(idx))))
            if len(set(indexes).intersection(set(prev_indexes))):
                sequence = np.empty((sequence_length, n_features))
                for i, prev_idx in enumerate(prev_indexes):
                    if prev_idx in converted_df['FVC'].index:
                        sequence[i] = [converted_df['FVC'].loc[prev_idx], converted_df['week_diff'].loc[prev_idx]]
                    else:
                        sequence[i] = [-1, -1]
                patient_data.append(sequence)
                patient_targets.append(converted_df['FVC'].loc[idx])
        patient_data = np.stack(patient_data)
        patient_targets = np.stack(patient_targets).reshape((-1, 1))
        data.append(patient_data)
        targets.append(patient_targets)
    data = np.concatenate(data, axis=0)
    targets = np.concatenate(targets, axis=0)

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
