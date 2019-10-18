import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registry, RegistryError, register


@register('generator', 'fc', default=True)
class FCGenerator(nn.Module):
    def __init__(self, options):
        super(FCGenerator, self).__init__()
        self.image_size = options.image_size
        self.dropout = options.generator_dropout
        size = options.state_size
        self.linear = []
        for i in range(options.generator_layers - 1):
            self.linear.append(nn.Linear(size, size * 2))
            self.add_module(f'linear_{i}', self.linear[i])
            size *= 2
        self.linear.append(nn.Linear(size, self.image_size * self.image_size))
        self.add_module(f'linear_{options.generator_layers - 1}', self.linear[-1])

    def forward(self, x):
        for layer in self.linear[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.linear[-1](x)
        return torch.tanh(x)

class Generator:
    @staticmethod
    def from_options(options):
        if options.generator in Registry.keys('generator'):
            cls = Registry.get('generator', options.generator)
            generator = cls(options)
        else:
            raise RegistryError(f'missing value {options.generator} for namespace `generator`')
        if options.restore:
            state_dict = torch.load(os.path.join(options.model_dir, options.experiment, 'generator.pt'))
            generator.load_state_dict(state_dict)
        generator = generator.to(options.device)
        return generator

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('generator')
        group.add_argument('--generator', choices=Registry.keys('generator'), default=Registry.default('generator'), help='type of generator')
        group.add_argument('--generator-dropout', type=float, help='dropout coefficient in generator layers')
        group.add_argument('--generator-layers', type=int, default=4, help='number of generator layers')