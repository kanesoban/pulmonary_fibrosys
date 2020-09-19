import argparse
import yaml

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.rnn.data.dense import get_combined_test_data
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

    test_dataset = get_combined_test_data(config['test_input_file'], config['test_scans_root'], config['depth'], config['height'], config['width'], max_fvc=MAX_FVC)
    model.predict(test_dataset)


if __name__ == "__main__":
    main()
