import argparse
import json
import pyzshcomplete

class DictToClass:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToClass(value))
            else:
                setattr(self, key, value)
                
def parse_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

def get_config():
    parser = argparse.ArgumentParser(description='Config Parser')
    parser.add_argument('--config', '-c', help='Path to the config JSON file')
    parser.add_argument('--data_folder', '-d', help='Data folder path')
    parser.add_argument('--image_size', '-i', nargs=2, type=int, help='Image size (width height)')
    parser.add_argument('--checkpoint', '-ckpt', help='Checkpoint path')
    parser.add_argument('--batch_size', '-b', type=int, help='Batch size')

    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs')
    parser.add_argument('--lr', '-l', type=float, help='Learning rate')
    parser.add_argument('--exp_group_name', '-egn', help='Experiment group name')
    pyzshcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.config:
        config = parse_config(args.config)
    else:
        config = {}

    if args.data_folder:
        config['data_folder'] = args.data_folder

    if args.image_size:
        config['image_size'] = args.image_size
    if args.checkpoint:
        config['checkpoint'] = args.checkpoint
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr
    if args.exp_group_name:
        config['exp_group_name'] = args.exp_group_name


    assert config, "No configurations provided."

    required_keys = ['data_folder', 'image_size', 'checkpoint', 'batch_size',  'epochs', 'lr',  'exp_group_name']
    missing_keys = [key for key in required_keys if key not in config]

    assert not missing_keys, f"Missing configuration keys: {', '.join(missing_keys)}"
    return DictToClass(config)
    