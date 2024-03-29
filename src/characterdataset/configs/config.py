import toml
import munch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PATH = os.path.join(current_dir, "default_config.toml")
# open config files
def load_global_config( filepath : str = DEFAULT_PATH):
    return munch.munchify( toml.load( filepath ) )

def save_global_config( new_config , filepath : str = DEFAULT_PATH):
    with open( filepath , "w", encoding="utf-8") as file:
        toml.dump( new_config , file )
