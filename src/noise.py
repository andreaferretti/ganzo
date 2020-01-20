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
import torch

from registry import Registry, RegistryError, register, with_option_parser


@register('noise', 'gaussian', default=True)
class GaussianNoise:
    '''
    A generator of Gaussian noise. Yields batches of noise having shape

        B x S

    where
    * B is the batch size
    * S is the state size

    Generated data is already moved to the appropriate device.

    The following options can be used to configure it:

    --state-size: the dimensionality of the noise space.
    '''
    def __init__(self, options):
        self.state_size = options.state_size
        self.batch_size = options.batch_size
        self.device = options.device

    def next(self):
        '''
        Yields a batch of noise.
        '''
        return torch.randn(self.batch_size, self.state_size).to(self.device)

class Noise:
    @staticmethod
    def from_options(options):
        if options.noise in Registry.keys('noise'):
            cls = Registry.get('noise', options.noise)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.noise} for namespace `noise`')

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        group = parser.add_argument_group('noise generation')
        group.add_argument('--noise', choices=Registry.keys('noise'), default=Registry.default('noise'), help='type of noise')
        group.add_argument('--state-size', type=int, default=128, help='state size')