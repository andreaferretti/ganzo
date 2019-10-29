import argparse
import json
import os
from hashlib import sha1

import torch

from data import Data
from generator import Generator
from discriminator import Discriminator
from loss import Loss
from noise import Noise
from hook import Hook
from statistics import Statistics
from snapshot import Snapshot
from evaluation import Evaluation
from game import Game
from registry import Registry


def _name(options):
    return sha1(json.dumps(vars(options), sort_keys=True).encode('utf8')).hexdigest()

class Options:
    def __init__(self, train=True):
        self.parser = argparse.ArgumentParser(description='GANzo')
        self.parser.add_argument('--from-json', help='load configuration from this JSON file')
        self.parser.add_argument('--experiment', help='experiment name, leave blank to autogenerate')
        self.parser.add_argument('--device', help='device name, leave blank to autodetect')
        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
        self.parser.add_argument('--model-dir', default='models', help='directory where to store the models')
        self.parser.add_argument('--restore', action='store_true', help='restart training from the saved models')
        self.parser.add_argument('--delete', action='store_true', help='delete saved models without asking')
        self.parser.add_argument('--seed', type=int, help='random number generator seed')

        Data.add_options(self.parser)
        Generator.add_options(self.parser)
        Discriminator.add_options(self.parser)
        Noise.add_options(self.parser)
        Snapshot.add_options(self.parser)
        if train:
            Loss.add_options(self.parser)
            Hook.add_options(self.parser)
            Statistics.add_options(self.parser)
            Evaluation.add_options(self.parser)
            Game.add_options(self.parser)
        for f in Registry.option_functions():
            f(self.parser)

    def from_command_line(self):
        options = self.parser.parse_args()
        if options.experiment is None:
            options.experiment = _name(options)
        if options.device is None:
            if torch.cuda.is_available():
                options.device = 'cuda'
            else:
                options.device = 'cpu'

        return options

    def from_json(self, path, parent=None):
        n = argparse.Namespace()
        with open(path, 'r') as f:
            dictionary = json.load(f)
            if parent is not None:
                dictionary = {**vars(parent), **dictionary}
            n.__dict__ = dictionary
        return n

    def from_json_and_command_line(self, path):
        n = argparse.Namespace()
        with open(path, 'r') as f:
            dictionary = json.load(f)
            n.__dict__ = dictionary
        return self.parser.parse_args(namespace=n)

    def save_as_json(self, options):
        experiment_dir = os.path.join(options.model_dir, options.experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        path = os.path.join(experiment_dir, 'options.json')
        with open(path, 'w') as f:
            json.dump(vars(options), f, indent=4)