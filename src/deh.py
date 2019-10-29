import os
import sys

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
from options import Options


if __name__ == '__main__':
    option_loader = Options(train=False)
    option_loader.parser.add_argument('--num-samples', type=int, default=1, help='number of samples to be generated')

    options = option_loader.from_command_line()
    model_dir = os.path.join(options.model_dir, options.experiment)
    json_path = os.path.join(model_dir, 'options.json')
    if os.path.exists(model_dir):
        options = option_loader.from_json_and_command_line(json_path)
    else:
        print('Missing model directory: {model_dir}')
        sys.exit(1)
    options.restore = True
    options.sample_every = 1
    print(options)

    if options.seed is not None:
        torch.random.manual_seed(options.seed)

    data = Data.from_options(options)
    generator = Generator.from_options(options)
    noise = Noise.from_options(options)
    snapshot = Snapshot.from_options(options)

    for _ in range(options.num_samples):
        snapshot.save(data, noise, generator)