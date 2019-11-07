import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry import Registry, RegistryError, register, with_option_parser


@register('discriminator', 'fc', default=True)
class FCDiscriminator(nn.Module):
    def __init__(self, options):
        '''
        The fully connected generator is initialized by creating a chain of
        fully connected layers that perform transformations

            n * n -> 2^(k - 1) * d -> ... -> 2 * d -> 1

        where
            d = options.state_size
            k = options.discriminator_layers
            n = options.image_size
        '''
        super(FCDiscriminator, self).__init__()
        self.dropout = options.discriminator_dropout
        self.layers = options.discriminator_layers

        sizes = [options.image_size * options.image_size]
        size = options.state_size * (2 ** (options.discriminator_layers - 1))
        for i in range(options.discriminator_layers - 1):
            sizes.append(size)
            size //=2
        sizes.append(1)

        # Notice that the number of layers is variable, hence we cannot
        # register them as fields on the module itself. The layers are
        # explicitly registered calling `.add_module()`, and later retrieved
        # by name. See `FCGenerator` to understand the reason why.
        for i in range(options.discriminator_layers):
            layer = nn.Linear(sizes[i], sizes[i + 1])
            self.add_module(f'linear_{i}', layer)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        layers = {}
        for name, module in self.named_children():
            layers[name] = module
        for i in range(self.layers):
            layer = layers[f'linear_{i}']
            x = layer(x)
            if i < self.layers - 1:
                x = F.leaky_relu(x, 0.2)
                if self.dropout is not None:
                    x = F.dropout(x, self.dropout)
        return torch.sigmoid(x)

@register('discriminator', 'conv')
class ConvDiscriminator(nn.Module):
    def __init__(self, options):
        super(ConvDiscriminator, self).__init__()
        self.dropout = options.discriminator_dropout
        self.layers = options.discriminator_layers

        d = 8

        # Notice that the number of layers is variable, hence we cannot
        # register them as fields on the module itself. The layers are
        # explicitly registered calling `.add_module()`, and later retrieved
        # by name. See `FCGenerator` to understand the reason why.
        self.add_module('conv_0', nn.Conv2d(options.image_colors, d, kernel_size=4, stride=2, padding=1))
        self.add_module('batch_norm_0', nn.BatchNorm2d(d))
        for i in range(1, options.discriminator_layers - 1):
            self.add_module(f'conv_{i}', nn.Conv2d(d, 2 * d, kernel_size=4, stride=2, padding=1))
            self.add_module(f'batch_norm_{i}', nn.BatchNorm2d(2 * d))
            d *= 2
        self.add_module(f'conv_{options.discriminator_layers - 1}', nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        batch_size = x.size()[0]
        layers = {}
        for name, module in self.named_children():
            layers[name] = module
        for i in range(self.layers - 1):
            conv = layers[f'conv_{i}']
            batch_norm = layers[f'batch_norm_{i}']
            x = batch_norm(conv(x))
            x = F.leaky_relu(x, 0.2)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        last_conv = layers[f'conv_{self.layers - 1}']
        x = last_conv(x)
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

@register('discriminator', 'patch-gan')
class PatchGANDiscriminator(nn.Module):
    '''
    PatchGAN from
        Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
        Image-to-Image Translation with Conditional Adversarial Networks
        https://arxiv.org/abs/1611.07004

    The patch size is 64.
    '''
    def __init__(self, options):
        super(PatchGANDiscriminator, self).__init__()
        d = options.discriminator_channels
        self.conv1 = self.conv(6, d)
        self.conv2 = self.conv(d, 2 * d)
        self.batch_norm2 = nn.BatchNorm2d(2 * d)
        self.conv3 = self.conv(2 * d, 4 * d)
        self.batch_norm3 = nn.BatchNorm2d(4 * d)
        self.conv4 = self.conv(4 * d, 8 * d)
        self.batch_norm4 = nn.BatchNorm2d(8 * d)
        self.conv5 = nn.Conv2d(8 * d, 1, kernel_size=4, stride=1, padding=0)

    def conv(self, i, o):
        return nn.Conv2d(i, o, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batch_norm2(self.conv2(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch_norm3(self.conv3(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch_norm4(self.conv4(x))
        x = F.leaky_relu(x, 0.2)
        x = F.sigmoid(self.conv5(x))
        return torch.mean(torch.mean(x, dim=3), dim=2).squeeze()

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
        if options.parallel:
            discriminator = nn.DataParallel(discriminator)
        return discriminator

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        group = parser.add_argument_group('discriminator')
        group.add_argument('--discriminator', choices=Registry.keys('discriminator'), default=Registry.default('discriminator'), help='type of discriminator')
        group.add_argument('--discriminator-dropout', type=float, help='dropout coefficient in discriminator layers')
        group.add_argument('--discriminator-layers', type=int, default=4, help='number of discriminator layers')
        group.add_argument('--discriminator-channels', type=int, default=4, help='number of channels for the discriminator')