import argparse
import yaml

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

from models.rnn import get_combined_model
from models.rnn.data.dense import get_combined_data, get_combined_test_data


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

    train_dataset, val_dataset = get_combined_data(config['train_input_file'], config['batch_size'],
                                                   config['sequence_length'], config['train_scans_root'],
                                                   config['depth'], config['height'], config['width'], max_fvc=MAX_FVC)

    model = get_combined_model(config['sequence_length'], config['learning_rate'], config['width'], config['height'],
                               config['depth'])

    steps_per_epoch = 1500 // config['batch_size']
    validation_steps = 300 // config['batch_size']
    model.fit(train_dataset, epochs=config['epochs'], validation_data=val_dataset, steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps)
    model.save(config['model_save_path'], save_format='h5')


'''

log_dir = os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [tensorboard_callback]

model.fit(train_dataset, epochs=config['epochs'], validation_data=val_dataset)
'''

if __name__ == "__main__":
    main()
