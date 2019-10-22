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

def _init_kaiming(m):
    if m.weight is not None:
        nn.init.kaiming_uniform_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)

def _init_xavier(m):
    if m.weight is not None:
        nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.0)

def _mean_pool(x):
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=int((kernel_size - 1) / 2),
            bias=True
        )

    def forward(self, x):
        return _mean_pool(self.conv(x))

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=int((kernel_size - 1) / 2),
            bias=True
        )

    def forward(self, x):
        return self.conv(_mean_pool(x))

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hw):
        super(ResidualBlock, self).__init__()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm([input_dim, hw, hw])
        self.layer_norm2 = nn.LayerNorm([input_dim, hw, hw])
        self.shortcut = MeanPoolConv(input_dim, output_dim, 1)
        self.conv = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=int((kernel_size - 1) / 2),
            bias=False
        )
        self.conv_mean_pool = ConvMeanPool(input_dim, output_dim, kernel_size)

        _init_xavier(self.shortcut.conv)
        _init_kaiming(self.conv)
        _init_kaiming(self.conv_mean_pool.conv)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        x = self.conv(x)
        x = self.layer_norm2(x)
        x = self.relu2(x)
        x = self.conv_mean_pool(x)

        return shortcut + x

@register('discriminator', 'good')
class GoodDiscriminator(nn.Module):
    '''
    GoodDiscriminator, using the 64x64 architecture described in
        Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
        Improved Training of Wasserstein GANs
        https://arxiv.org/abs/1704.00028

    Dropout is not used and the number of layers is fixed.
    Image size must be a multiple of 16.
    '''
    def __init__(self, options):
        super(GoodDiscriminator, self).__init__()


        if options.image_size % 16 != 0:
            raise ValueError('Image size must be multiple of 16')
        start = options.image_size // 16
        side = options.image_size

        self.image_size = options.image_size
        self.start = start

        self.conv = nn.Conv2d(3, side, kernel_size=3, stride=1, padding=1, bias=True)
        self.rb1 = ResidualBlock(1 * side, 2 * side, 3, hw=side) # <--- check whether to use `side` or `64`
        self.rb2 = ResidualBlock(2 * side, 4 * side, 3, hw=int(side / 2))
        self.rb3 = ResidualBlock(4 * side, 8 * side, 3, hw=int(side / 4))
        self.rb4 = ResidualBlock(8 * side, 8 * side, 3, hw=int(side / 8))
        self.fc = nn.Linear(start * start * 8 * side, 1)

        _init_xavier(self.fc)
        _init_xavier(self.conv)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.contiguous().view(batch_size, 3, self.image_size, self.image_size)
        x = self.conv(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = x.view(batch_size, self.start * self.start * 8 * self.image_size)
        return self.fc(x).view(-1)

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