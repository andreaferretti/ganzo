import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registry, RegistryError, register


@register('discriminator', 'fc', default=True)
class FCDiscriminator(nn.Module):
    def __init__(self, options):
        super(FCDiscriminator, self).__init__()
        self.dropout = options.discriminator_dropout
        self.batch_size = options.batch_size

        size = options.state_size * (2 ** (options.discriminator_layers - 1))
        self.linear = []
        self.linear.append(nn.Linear(options.image_size * options.image_size, size))
        self.add_module(f'linear_0', self.linear[0])
        for i in range(1, options.generator_layers - 1):
            self.linear.append(nn.Linear(size, size // 2))
            self.add_module(f'linear_{i}', self.linear[i])
            size //= 2
        self.linear.append(nn.Linear(size, 1))
        self.add_module(f'linear_{options.discriminator_layers - 1}', self.linear[-1])

    def forward(self, x):
        x = x.view(self.batch_size, -1)
        for layer in self.linear[:-1]:
            x = F.leaky_relu(layer(x), 0.2)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.linear[-1](x)
        return torch.sigmoid(x)

class Discriminator:
    @staticmethod
    def from_options(options):
        if options.discriminator in Registry.keys('discriminator'):
            cls = Registry.get('discriminator', options.discriminator)
            discriminator = cls(options)
        else:
            raise RegistryError(f'missing value {options.discriminator} for namespace `discriminator`')
        if options.restore:
            pass
        discriminator = discriminator.to(options.device)
        return discriminator

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('discriminator')
        group.add_argument('--discriminator', choices=Registry.keys('discriminator'), default=Registry.default('discriminator'), help='type of discriminator')
        group.add_argument('--discriminator-dropout', type=float, help='dropout coefficient in discriminator layers')
        group.add_argument('--discriminator-layers', type=int, default=4, help='number of discriminator layers')