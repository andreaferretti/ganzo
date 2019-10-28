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

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, output_height, output_width, output_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bias=True):
        super(UpSampleConv, self).__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=int((kernel_size - 1) / 2),
            bias=bias
        )
        self.depth_to_space = DepthToSpace(2)

    def forward(self, x):
        x = self.depth_to_space(torch.cat((x, x, x, x), 1))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hw=64):
        super(ResidualBlock, self).__init__()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(input_dim)
        self.batch_norm2 = nn.BatchNorm2d(output_dim)
        self.shortcut = UpSampleConv(input_dim, output_dim, 1)
        self.upsample_conv = UpSampleConv(input_dim, output_dim, kernel_size, bias=False)
        self.conv = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=int((kernel_size - 1)/2),
            bias=True
        )

        _init_xavier(self.shortcut.conv)
        _init_kaiming(self.upsample_conv.conv)
        _init_kaiming(self.conv)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.upsample_conv(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv(x)

        return shortcut + x

@register('generator', 'good')
class GoodGenerator(nn.Module):
    '''
    GoodGenerator, using the 64x64 architecture described in
        Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
        Improved Training of Wasserstein GANs
        https://arxiv.org/abs/1704.00028

    Dropout is not used and the number of layers is fixed.
    Image size must be a multiple of 16.
    '''
    def __init__(self, options):
        super(GoodGenerator, self).__init__()

        if options.image_size % 16 != 0:
            raise ValueError('Image size must be multiple of 16')
        start = options.image_size // 16
        side = options.image_size

        self.image_size = options.image_size
        self.start = start

        self.fc = nn.Linear(options.state_size, start * start * 8 * side)
        self.rb1 = ResidualBlock(8 * side, 8 * side, 3)
        self.rb2 = ResidualBlock(8 * side, 4 * side, 3)
        self.rb3 = ResidualBlock(4 * side, 2 * side, 3)
        self.rb4 = ResidualBlock(2 * side, 1 * side, 3)
        self.batch_norm = nn.BatchNorm2d(side)
        self.conv = nn.Conv2d(side, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        _init_xavier(self.fc)
        _init_kaiming(self.conv)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.fc(x.contiguous()).view(-1, 8 * self.image_size, self.start, self.start)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x.view(batch_size, 3, self.image_size, self.image_size)

@register('generator', 'u-gen')
class UGenerator(nn.Module):
    '''
    UGenerator from
        Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
        Image-to-Image Translation with Conditional Adversarial Networks
        https://arxiv.org/abs/1611.07004

    Dropout is not used.
    Image size must be a multiple of 2^layers.
    '''
    def __init__(self, options):
        super(UGenerator, self).__init__()
        d = options.generator_channels
        self.layers = options.generator_layers
        self.conv_in = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)
        self.convs = []
        self.deconvs = []
        self.bns = []
        self.debns = []
        for i in range(options.generator_layers):
            c = 2 ** i
            self.convs.append(nn.Conv2d(c * d, 2 * c * d, kernel_size=4, stride=2, padding=1))
            self.deconvs.append(nn.ConvTranspose2d(4 * c * d, c * d, kernel_size=4, stride=2, padding=1))
            self.bns.append(nn.BatchNorm2d(2 * c * d))
            self.debns.append(nn.BatchNorm2d(c * d))
            self.add_module(f'conv_{i}', self.convs[i])
            self.add_module(f'bn_{i}', self.bns[i])
            self.add_module(f'deconv_{i}', self.deconvs[i])
            self.add_module(f'debn_{i}', self.debns[i])
        self.deconv_out = nn.ConvTranspose2d(2 * d, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x0):
        xs = []
        x_in = self.conv_in(x0)
        x = x_in
        for i in range(self.layers):
            conv = self.convs[i]
            bn = self.bns[i]
            x = conv(F.leaky_relu(x, 0.2))
            if i < self.layers - 1:
                x = bn(x)
            xs.append(x)
        y = x
        for i in range(self.layers, 0, -1):
            deconv = self.deconvs[i-1]
            debn = self.debns[i-1]
            y_ = torch.cat((y, xs[i - 1]), dim=1)
            y = debn(deconv(F.relu(y_)))
        y_out = self.deconv_out(F.relu(torch.cat((y, x_in), dim=1)))
        return F.tanh(y_out)

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
        group.add_argument('--generator-channels', type=int, default=8, help='number of channels for the generator')