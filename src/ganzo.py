import os
import sys

import torch

from data import Data
from generator import Generator
from discriminator import Discriminator
from loss import Loss
from noise import Noise
from statistics import Statistics
from snapshot import Snapshot
from game import Game
from options import Options


if __name__ == '__main__':
    option_loader = Options()
    options = option_loader.from_command_line()
    print(options)

    model_dir = os.path.join(options.model_dir, options.experiment)
    json_path = os.path.join(model_dir, 'options.json')
    if os.path.exists(model_dir):
        if not options.delete:
            print(f'Directory {model_dir} already exists, what do you want to do?')
            choice = None
            while choice not in ['r', 'n', 'd']:
                try:
                    print('r) restore training from the saved models, keeping existing options')
                    print('n) restore training from the saved models, using newly provided options')
                    print('d) delete existing models and options, and start from scratch')
                    choice = input().strip()
                except EOFError:
                    sys.exit(0)
                if choice == 'r':
                    options = option_loader.from_json(json_path)
                    options.restore = True
                elif choice == 'n':
                    options.restore = True
                else:
                    options.restore = False
        else:
            options.restore = False
    option_loader.save_as_json(options)

    if options.seed is not None:
        torch.random.manual_seed(options.seed)

    data = Data.from_options(options)
    generator = Generator.from_options(options)
    discriminator = Discriminator.from_options(options)
    loss = Loss.from_options(options, discriminator)
    noise = Noise.from_options(options)
    statistics = Statistics.from_options(options)
    snapshot = Snapshot.from_options(options)
    game = Game.from_options(options, generator, discriminator, loss)

    for _ in range(options.epochs):
        losses = game.run_epoch(data, noise)
        statistics.log(losses)
        snapshot.save(data, noise, generator)