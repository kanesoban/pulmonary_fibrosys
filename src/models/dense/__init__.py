import random

import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.dense.metrics import laplace_log_likelihood
from models.dense.losses import laplace_log_likelihood_loss

MIN_WEEK = -12
MAX_WEEK = 133


def get_model():
    inputs = tf.keras.Input(shape=(11,))
    prediction_output = tf.keras.layers.Dense(1)(inputs)
    uncertainty_output = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)

    outputs = tf.keras.layers.concatenate([prediction_output, uncertainty_output])

    metrics = [laplace_log_likelihood]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=laplace_log_likelihood_loss, optimizer=tf.optimizers.Adam(), metrics=metrics)
    return model


def get_data(input_file, batch_size, split=0.8):
    train_data = pd.read_csv(input_file)
    train_data['Weeks'] = train_data['Weeks']
    max_FVC = train_data['FVC'].max()
    train_data['FVC'] /= max_FVC
    train_data['Percent'] /= train_data['Percent'].max()
    train_data['Age'] /= train_data['Age'].max()
    sex_df = pd.get_dummies(train_data['Sex'])
    train_data['Female'] = sex_df['Female']
    train_data['Male'] = sex_df['Male']
    smoking_df = pd.get_dummies(train_data['SmokingStatus'])
    train_data['Ex-smoker'] = smoking_df['Ex-smoker']
    train_data['Never smoked'] = smoking_df['Never smoked']
    train_data['Currently smokes'] = smoking_df['Currently smokes']

    grouped = train_data.groupby(train_data.Patient)
    patient_dfs = []
    for patient in tqdm(train_data['Patient'].unique()):
        patient_df = grouped.get_group(patient)

        FVC = patient_df['FVC'].iloc[:-1].tolist()
        FVC_next = patient_df['FVC'].iloc[1:].tolist()

        # This will only work until i am not using
        weeks = patient_df['Weeks']

        Weeks = weeks.iloc[:-1]
        Weeks_next = weeks.iloc[1:]
        week_diff = (np.array(Weeks_next.tolist()) - np.array(Weeks.tolist())) / (MAX_WEEK - MIN_WEEK)
        Weeks -= MIN_WEEK
        Weeks /= (MAX_WEEK - MIN_WEEK)
        Weeks = Weeks.tolist()
        Weeks_next -= MIN_WEEK
        Weeks_next /= (MAX_WEEK - MIN_WEEK)
        Weeks_next = Weeks_next.tolist()
        percent = patient_df['Percent'].iloc[:-1].tolist()
        age = patient_df['Age'].iloc[:-1].tolist()
        female = patient_df['Female'].iloc[:-1].tolist()
        male = patient_df['Male'].iloc[:-1].tolist()
        ex_smoker = patient_df['Ex-smoker'].iloc[:-1].tolist()
        never_smoked = patient_df['Never smoked'].iloc[:-1].tolist()
        currently_smokes = patient_df['Currently smokes'].iloc[:-1].tolist()

        converted_data = {'FVC': FVC, 'FVC_next': FVC_next,
                          'week_diff': week_diff, 'Weeks': Weeks, 'Weeks_next': Weeks_next,
                          'Percent': percent, 'Age': age, 'Female': female, 'Male': male, 'Ex-smoker': ex_smoker,
                          'Never smoked': never_smoked, 'Currently smokes': currently_smokes}
        converted_df = pd.DataFrame.from_dict(converted_data)

        patient_dfs.append(converted_df)
    result = pd.concat(patient_dfs)
    num_data = len(result)
    indexes = list(range(num_data))
    random.shuffle(indexes)
    data = result.iloc[indexes]
    split = int(num_data * split)
    training_data = data.iloc[:split]
    val_data = data.iloc[split:]
    input_columns = ['Weeks', 'FVC', 'Weeks_next', 'week_diff', 'Percent', 'Age', 'Female', 'Male', 'Ex-smoker',
                     'Never smoked', 'Currently smokes']
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
