import toml
import munch

# open config files
def load_global_config( filepath : str = "configs\default_config.toml" ):
    return munch.munchify( toml.load( filepath ) )

def save_global_config( new_config , filepath : str = "configs\default_config.toml" ):
    with open( filepath , "w" ) as file:
        toml.dump( new_config , file )
