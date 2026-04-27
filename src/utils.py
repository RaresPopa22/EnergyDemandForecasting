import numpy as np
import torch
import yaml


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def deep_merge(base, override):
    result = {**base}

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def read_configs(base, override):
    base_config = read_config(base)
    override_config = read_config(override)

    return deep_merge(base_config, override_config)


def setup_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    

def set_random_seeds(config):
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
