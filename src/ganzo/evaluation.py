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
import math

from ganzo.registry import Registry, RegistryError, register, with_option_parser


@register('evaluation', 'latest', default=True)
class Latest:
    def __init__(self, options):
        pass

    def has_improved(self, losses):
        return True

@register('evaluation', 'generator-loss')
class GeneratorLoss:
    def __init__(self, options):
        self.best_loss = math.inf

    def has_improved(self, losses):
        loss = losses['generator']
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        else:
            return False

class Evaluation:
    @staticmethod
    def from_options(options):
        if options.evaluation_criterion in Registry.keys('evaluation'):
            cls = Registry.get('evaluation', options.evaluation_criterion)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.evaluation_criterion} for namespace `evaluation`')

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        if train:
            group = parser.add_argument_group('model evaluation')
            group.add_argument('--evaluation-criterion', choices=Registry.keys('evaluation'), default=Registry.default('evaluation'), help='the criterion to evaluate model improvement')