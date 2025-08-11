import os
from pathlib import Path
from omegaconf import OmegaConf
from platformdirs import user_cache_dir
from joblib import Memory

# The default
config_names = os.getenv('CONFIG_NAMES', None)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_MODELS_PATH = os.getenv('VIPER_MODELS_DIR', f"{ROOT}/pretrained_models")

OmegaConf.register_new_resolver("root", lambda: str(ROOT))
OmegaConf.register_new_resolver("pretrained_models_path", lambda: str(PRETRAINED_MODELS_PATH))

configs = [OmegaConf.load(f'{SCRIPT_DIR}/base_config.yaml')]

if config_names is not None:
    for config_name in config_names.split(','):
        configs.append(OmegaConf.load(f'{SCRIPT_DIR}/{config_name.strip()}.yaml'))

# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

# Set the cache path
config.cache_path = os.getenv("VIPER_CACHE_DIR") or user_cache_dir("vipermcp")

