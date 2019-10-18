import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registry, RegistryError, register


@register('discriminator', 'fc', default=True)
class FCDiscriminator(nn.Module):
    def __init__(self, options):
        super(FCDiscriminator, self).__init__()
        self.dropout = options.discriminator_dropout

        size = options.state_size * (2 ** (options.discriminator_layers - 1))
        self.linear = []
        self.linear.append(nn.Linear(options.image_size * options.image_size, size))
        self.add_module('linear_0', self.linear[0])
        for i in range(1, options.discriminator_layers - 1):
            self.linear.append(nn.Linear(size, size // 2))
            self.add_module(f'linear_{i}', self.linear[i])
            size //= 2
        self.linear.append(nn.Linear(size, 1))
        self.add_module(f'linear_{options.discriminator_layers - 1}', self.linear[-1])

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        for layer in self.linear[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.linear[-1](x)
        return torch.sigmoid(x)

@register('discriminator', 'conv')
class ConvDiscriminator(nn.Module):
    def __init__(self, options):
        super(ConvDiscriminator, self).__init__()
        self.dropout = options.discriminator_dropout

        self.conv = []
        self.batch_norm = []
        d = 8

        self.conv.append(nn.Conv2d(options.image_colors, d, kernel_size=4, stride=2, padding=1))
        self.batch_norm.append(nn.BatchNorm2d(d))
        self.add_module('conv_0', self.conv[0])
        self.add_module('batch_norm_0', self.batch_norm[0])
        for i in range(1, options.discriminator_layers - 1):
            self.conv.append(nn.Conv2d(d, 2 * d, kernel_size=4, stride=2, padding=1))
            self.batch_norm.append(nn.BatchNorm2d(2 * d))
            self.add_module(f'conv_{i}', self.conv[i])
            self.add_module(f'batch_norm_{i}', self.batch_norm[i])
            d *= 2
        self.conv.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))
        self.add_module(f'conv_{options.discriminator_layers - 1}', self.conv[-1])

    def forward(self, x):
        batch_size = x.size()[0]
        for conv, batch_norm in zip(self.conv, self.batch_norm):
            x = batch_norm(conv(x))
            x = F.leaky_relu(x, 0.2)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.conv[-1](x)
        return torch.sigmoid(x).view(batch_size, -1)

class Discriminator:
    @staticmethod
    def from_options(options):
        if options.discriminator in Registry.keys('discriminator'):
            cls = Registry.get('discriminator', options.discriminator)
            discriminator = cls(options)
        else:
            raise RegistryError(f'missing value {options.discriminator} for namespace `discriminator`')
        if options.restore:
            state_dict = torch.load(os.path.join(options.model_dir, options.experiment, 'discriminator.pt'))
            discriminator.load_state_dict(state_dict)
        discriminator = discriminator.to(options.device)
        return discriminator

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('discriminator')
        group.add_argument('--discriminator', choices=Registry.keys('discriminator'), default=Registry.default('discriminator'), help='type of discriminator')
        group.add_argument('--discriminator-dropout', type=float, help='dropout coefficient in discriminator layers')
        group.add_argument('--discriminator-layers', type=int, default=4, help='number of discriminator layers')