import argparse
import os
import yaml

import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from utils import get_patient_scan
from models.rnn.data.dense import get_combined_test_data, get_combined_data
from models.rnn.losses import laplace_log_likelihood_loss
from models.rnn.metrics import laplace_log_likelihood


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--test_file')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    MAX_FVC = 4000

    model = tf.keras.models.load_model(config['model_save_path'], custom_objects={'laplace_log_likelihood_loss': laplace_log_likelihood_loss, 'laplace_log_likelihood': laplace_log_likelihood})

    test_data = pd.read_csv(config['test_input_file'])
    n_features = 0
    test_data['Weeks'] = test_data['Weeks']
    n_features += 1
    test_data['FVC'] /= MAX_FVC
    n_features += 1
    grouped = test_data.groupby(test_data.Patient)

    for patient in tqdm(test_data['Patient'].unique()):
        patient_df = grouped.get_group(patient)
        FVC = patient_df['FVC'].iloc[:1].tolist()
        week_diff = np.nan
        converted_data = {'FVC': FVC, 'week_diff': 0.1}
        converted_df = pd.DataFrame.from_dict(converted_data)
        patient_dir = os.path.join(config['test_scans_root'], patient)
        img = np.expand_dims(get_patient_scan(patient_dir, n_depth=config['depth'], rows=config['height'], columns=config['width']), axis=-1)
        sequence = np.empty((1, n_features))
        sequence[0] = [converted_df['FVC'].loc[0], converted_df['week_diff'].loc[0]]
        data = (np.expand_dims(sequence, axis=0), np.expand_dims(img, axis=0))
        model.predict([data])


if __name__ == "__main__":
    main()
