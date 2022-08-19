import os
import yaml
import threading
import numpy as np
import argparse


def merge_config(config,args):
    for key_1 in config.keys():
        if(isinstance(config[key_1],dict)):
            for key_2 in config[key_1].keys():
                if(key_2) in dir(args):
                    config[key_1][key_2] = getattr(args,key_2)
    return config

def load_yaml(yaml_name, args):
    config = yaml.load(open(yaml_name, 'r', encoding='utf-8'),Loader=yaml.FullLoader)
    config = merge_config(config, args)
    return config

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)