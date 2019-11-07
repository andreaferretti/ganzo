import os
import sys

import torch

from data import Data
from generator import Generator
from discriminator import Discriminator
from replica import Replica
from loss import Loss
from noise import Noise
from hook import Hook
from statistics import Statistics
from snapshot import Snapshot
from evaluation import Evaluation
from game import Game
from options import Options


if __name__ == '__main__':
    option_loader = Options()
    options = option_loader.from_command_line()
    restored_from_json = False
    if options.from_json is not None:
        options = option_loader.from_json_and_command_line(options.from_json)
        restored_from_json = True
    print(options)

    model_dir = os.path.join(options.model_dir, options.experiment)
    json_path = os.path.join(model_dir, 'options.json')
    if os.path.exists(model_dir):
        if options.restore:
            if not restored_from_json:
                options = option_loader.from_json_and_command_line(json_path)
        elif options.delete:
            options.restore = False
        else:
            print(f'Directory {model_dir} already exists, what do you want to do?')
            choice = None
            while choice not in ['r', 'n', 'd', 'a']:
                try:
                    print('r) restore training from the saved models, keeping existing options')
                    print('n) restore training from the saved models, using newly provided options')
                    print('d) delete existing models and options, and start from scratch')
                    print('a) abort')
                    choice = input().strip()
                except EOFError:
                    sys.exit(1)
                if choice == 'r':
                    options = option_loader.from_json_and_command_line(json_path)
                    options.restore = True
                elif choice == 'n':
                    options.restore = True
                elif choice == 'a':
                    sys.exit(2)
                else:
                    options.restore = False
    option_loader.save_as_json(options)

    if options.seed is not None:
        torch.random.manual_seed(options.seed)

    data = Data.from_options(options)
    replica = Replica.from_options(options)
    generators = replica.generators(options)
    discriminators = replica.generators(options)
    loss = Loss.from_options(options, discriminator)
    noise = Noise.from_options(options)
    hooks = Hook.from_options(options)
    statistics = Statistics.from_options(options)
    snapshot = Snapshot.from_options(options)
    evaluation = Evaluation.from_options(options)
    game = Game.from_options(options, generators, discriminators, loss, hooks)

    for _ in range(options.epochs):
        losses = game.run_epoch(data, noise)
        statistics.log(losses)
        snapshot.save(data, noise, generators)
        if evaluation.has_improved(losses):
            torch.save(generator.state_dict(), os.path.join(options.model_dir, options.experiment, 'generator.pt'))
            torch.save(discriminator.state_dict(), os.path.join(options.model_dir, options.experiment, 'discriminator.pt'))