import yaml
import argparse

def read_config(cfg_file):
    with open(cfg_file, 'r') as f:
        config = yaml.safe_load(f)

    return dict2namespace(config)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace

def namespace2dict(namespace):
    config = {}
    for key, value in vars(namespace).items():
        if isinstance(value, argparse.Namespace):
            value = namespace2dict(value)
        config[key] = value
    return config