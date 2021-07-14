# Copyright 2020 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import torch

from ganzo.data import Data
from ganzo.generator import Generator
from ganzo.discriminator import Discriminator
from ganzo.loss import Loss
from ganzo.noise import Noise
from ganzo.hook import Hook
from ganzo.statistics import Statistics
from ganzo.snapshot import Snapshot
from ganzo.evaluation import Evaluation
from ganzo.game import Game
from ganzo.options import Options


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
    generator = Generator.from_options(options)
    discriminator = Discriminator.from_options(options)
    loss = Loss.from_options(options, discriminator)
    noise = Noise.from_options(options)
    hooks = Hook.from_options(options)
    statistics = Statistics.from_options(options)
    snapshot = Snapshot.from_options(options)
    evaluation = Evaluation.from_options(options)
    game = Game.from_options(options, generator, discriminator, loss, hooks)

    for _ in range(options.epochs):
        losses = game.run_epoch(data, noise)
        statistics.log(losses)
        snapshot.save(data, noise, generator)
        if evaluation.has_improved(losses):
            torch.save(generator.state_dict(), os.path.join(options.model_dir, options.experiment, 'generator.pt'))
            torch.save(discriminator.state_dict(), os.path.join(options.model_dir, options.experiment, 'discriminator.pt'))