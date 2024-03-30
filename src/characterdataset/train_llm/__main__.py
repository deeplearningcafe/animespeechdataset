import argparse
from .train_conversational import main
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = "default_config.toml"
CONFIG_PATH = os.path.join(current_dir, CONFIG_FILE)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='LLM instruction tuning'
    )
    
    parser.add_argument('--config_file', default=CONFIG_PATH, type=str,
        required=False, help='File with the configuration for training')
    
    args = parser.parse_args()
    parser.print_help()
    main(args)