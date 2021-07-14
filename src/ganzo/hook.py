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

from ganzo.registry import Registry, RegistryError, register, with_option_parser


@register('hook', 'weight-clipper')
class WeightClipper:
    def __init__(self, options):
        self.clip_to = options.clip_to

    def clip(self, layer):
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            w.clamp_(-self.clip_to, self.clip_to)

    def apply(self, model):
        '''
        Applies weight clipping to all layers of a model.
        '''
        model.apply(self.clip)

class Hook:
    @staticmethod
    def from_options(options):
        hooks = {
            'generator': None,
            'discriminator': None
        }
        if options.generator_hook is not None:
            if options.generator_hook in Registry.keys('hook'):
                cls = Registry.get('hook', options.generator_hook)
                hooks['generator'] = cls(options)
            else:
                raise RegistryError(f'missing value {options.generator_hook} for namespace `hook`')
        if options.discriminator_hook is not None:
            if options.discriminator_hook in Registry.keys('hook'):
                cls = Registry.get('hook', options.discriminator_hook)
                hooks['discriminator'] = cls(options)
            else:
                raise RegistryError(f'missing value {options.discriminator_hook} for namespace `hook`')
        return hooks

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        if train:
            group = parser.add_argument_group('generator and discriminator hooks')
            group.add_argument('--generator-hook', choices=Registry.keys('hook'), help='type of hook to apply to the discriminator')
            group.add_argument('--discriminator-hook', choices=Registry.keys('hook'), help='type of hook to apply to the generator')
            group.add_argument('--clip-to', type=float, default=0.001, help='amount to clip weights')