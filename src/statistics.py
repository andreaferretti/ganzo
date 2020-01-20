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
import importlib
from timeit import default_timer as timer

from registry import Registry, RegistryError, register, with_option_parser


@register('log', 'none')
class NoStatistics:
    def __init__(self, options):
        pass

    def log(self, losses):
        pass

@register('log', 'console', default=True)
class ConsoleStatistics:
    def __init__(self, options):
        self.last_time = timer()
        self.epoch = options.start_epoch

    def log(self, losses):
        now = timer()
        print(f'=== Epoch {self.epoch} ===')
        print(f'Time since last iteration: {now - self.last_time}')
        self.last_time = now
        for k, v in losses.items():
            print(f'  {k}: {v}')
        self.epoch += 1

@register('log', 'file')
class FileStatistics:
    def __init__(self, options):
        self.last_time = timer()
        self.epoch = options.start_epoch
        if options.log_file is not None:
            self.log_file = options.log_file
        else:
            self.log_file = os.path.join(options.output_dir, options.experiment, 'statistics.log')

    def log(self, losses):
        now = timer()
        with open(self.log_file, 'a') as f:
            f.write(f'=== Epoch {self.epoch} ===\n')
            f.write(f'Time since last iteration: {now - self.last_time}\n')
            self.last_time = now
            for k, v in losses.items():
                f.write(f'  {k}: {v}\n')
            self.epoch += 1

tensorboard_enabled = importlib.util.find_spec('tensorboardX') is not None

if tensorboard_enabled:
    from tensorboardX import SummaryWriter

    @register('log', 'tensorboard')
    class TensorBoardStatistics:
        def __init__(self, options):
            self.writer = SummaryWriter(os.path.join(options.output_dir, options.experiment))
            self.epoch = options.start_epoch

        def log(self, losses):
            for k, v in losses.items():
                self.writer.add_scalar(k, v, self.epoch)
            self.epoch += 1

class Statistics:
    @staticmethod
    def from_options(options):
        if options.log in Registry.keys('log'):
            cls = Registry.get('log', options.log)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.log} for namespace `log`')

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        if train:
            group = parser.add_argument_group('logging')
            group.add_argument('--log', choices=Registry.keys('log'), default=Registry.default('log'), help='logging format')
            group.add_argument('--log-file', help='file to log statistics')