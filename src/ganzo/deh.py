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
    options.parallel = False
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