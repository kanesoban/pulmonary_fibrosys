import argparse
import yaml

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.rnn import get_combined_model
from models.rnn.data.dense import get_combined_data


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

    train_dataset, val_dataset = get_combined_data(config['input_file'], config['batch_size'], config['sequence_length'], config['scans_root'], config['depth'], config['height'], config['width'])

    model = get_combined_model(config['sequence_length'], config['learning_rate'], config['width'], config['height'], config['depth'])

    model.fit(train_dataset, epochs=config['epochs'], validation_data=val_dataset)


'''
def main():
    args = parse_args()
    with open(args.config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    #train_dataset, val_dataset = get_data(config['input_file'], config['batch_size'], config['sequence_length'], data_dir=config['data_dir'])
    train_dataset, val_dataset = get_data(config['input_file'], config['batch_size'], config['sequence_length'])

    model = get_model(config['sequence_length'], config['learning_rate'])

    model.fit(train_dataset, epochs=config['epochs'], validation_data=val_dataset)
'''

'''

log_dir = os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [tensorboard_callback]

model.fit(train_dataset, epochs=config['epochs'], validation_data=val_dataset)
'''


if __name__ == "__main__":
    main()
