import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)

import argparse
import logging
from pprint import pformat

from src.utils.config import read_config, dict2namespace, namespace2dict
from src.fast_agents import get_runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    hps = read_config(args.cfg_file)

    logging.basicConfig(filename=hps.running.exp_dir + f'/{args.mode}.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')
    logging.info('\n\n Config:\n\n'+pformat(namespace2dict(hps)))

    runner = get_runner(hps)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'test':
        runner.test()
    else:
        raise ValueError()

if __name__ == "__main__":
    main()