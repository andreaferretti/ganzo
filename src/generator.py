import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registry, RegistryError, register


@register('generator', 'fc', default=True)
class FCGenerator(nn.Module):
    def __init__(self, options):
        super(FCGenerator, self).__init__()
        self.dropout = options.generator_dropout
        size = options.state_size
        self.linear = []
        for i in range(options.generator_layers - 1):
            self.linear.append(nn.Linear(size, size * 2))
            self.add_module(f'linear_{i}', self.linear[i])
            size *= 2
        self.linear.append(nn.Linear(size, options.image_size * options.image_size))
        self.add_module(f'linear_{options.generator_layers - 1}', self.linear[-1])

    def forward(self, x):
        for layer in self.linear[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.linear[-1](x)
        return torch.tanh(x)

@register('generator', 'conv')
class ConvGenerator(nn.Module):
    def __init__(self, options):
        super(ConvGenerator, self).__init__()
        self.dropout = options.generator_dropout

        self.conv = []
        self.batch_norm = []
        d = 8 * 8

        self.conv.append(nn.ConvTranspose2d(options.state_size, d, kernel_size=4, stride=1, padding=0))
        self.batch_norm.append(nn.BatchNorm2d(d))
        self.add_module('conv_0', self.conv[0])
        self.add_module('batch_norm_0', self.batch_norm[0])
        for i in range(1, options.generator_layers - 1):
            self.conv.append(nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1))
            self.batch_norm.append(nn.BatchNorm2d(d // 2))
            self.add_module(f'conv_{i}', self.conv[i])
            self.add_module(f'batch_norm_{i}', self.batch_norm[i])
            d //= 2
        self.conv.append(nn.ConvTranspose2d(d, options.image_colors, kernel_size=4, stride=2, padding=1))
        self.add_module(f'conv_{options.generator_layers - 1}', self.conv[-1])

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1, 1, 1)
        for conv, batch_norm in zip(self.conv, self.batch_norm):
            x = batch_norm(conv(x))
            x = F.relu(x)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.conv[-1](x)
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