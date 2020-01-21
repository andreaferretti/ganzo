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

import torch.nn as nn
import torch.nn.functional as F

class Upsampler(nn.Module):
    '''
    Upsampler from
        David Berthelot, Thomas Schumm, Luke Metz
        BEGAN: Boundary Equilibrium GenerativeAdversarial Networks
        https://arxiv.org/abs/1703.10717
    '''
    def __init__(self, layers, channels, upsamples, image_size, input_size, dropout):
        super().__init__()
        scale_factor = 2 ** upsamples
        if image_size % scale_factor != 0:
            raise ValueError(f'Image size must be multiple of 2^{upsamples}')
        if (layers - 1) % (upsamples + 1) != 0:
            raise ValueError(f'Number of layers - 1 must be multiple of {upsamples + 1}')
        size = image_size // scale_factor
        layers_per_upsample = (layers - 1) // (upsamples + 1) - 1

        self.initial_size = size
        self.channels = channels
        self.dropout = dropout
        self.fc = nn.Linear(input_size, size * size * channels)
        self.layers = []
        for i in range(upsamples + 1):
            for j in range(layers_per_upsample):
                layer = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                self.add_module(f'conv_{i}_{j}', layer)
                self.layers.append(layer)
            if i < upsamples:
                layer = nn.Upsample(scale_factor=2, mode='nearest')
                self.add_module(f'upsample_{i}', layer)
                self.layers.append(layer)

        self.final_conv = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.channels, self.initial_size, self.initial_size)
        for layer in self.layers:
            x = layer(x)
            x = F.elu(x)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = self.final_conv(x)
        return x

class Downsampler(nn.Module):
    '''
    Downsampler from
        David Berthelot, Thomas Schumm, Luke Metz
        BEGAN: Boundary Equilibrium GenerativeAdversarial Networks
        https://arxiv.org/abs/1703.10717
    '''
    def __init__(self, layers, channels, downsamples, image_size, target_size, dropout):
        super().__init__()
        scale_factor = 2 ** downsamples
        if image_size % scale_factor != 0:
            raise ValueError(f'Image size must be multiple of 2^{downsamples}')
        if (layers - 1) % (downsamples + 1) != 0:
            raise ValueError(f'Number of layers - 1 must be multiple of {downsamples + 1}')
        size = image_size // scale_factor
        layers_per_downsample = (layers - 1) // (downsamples + 1) - 1

        self.channels = channels
        self.dropout = dropout
        self.intial_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.layers = []
        for i in range(downsamples + 1):
            for j in range(layers_per_downsample):
                if (j < layers_per_downsample - 1) or (i == downsamples):
                    layer = nn.Conv2d((i + 1) * channels, (i + 1) * channels, kernel_size=3, stride=1, padding=1)
                else:
                    layer = nn.Conv2d((i + 1) * channels, (i + 2) * channels, kernel_size=3, stride=1, padding=1)
                self.add_module(f'conv_{i}_{j}', layer)
                self.layers.append(layer)
            if i < downsamples:
                layer = nn.Conv2d((i + 2) * channels, (i + 2) * channels, kernel_size=2, stride=2, padding=0)
                self.add_module(f'downsample_{i}', layer)
                self.layers.append(layer)

        self.fc = nn.Linear(size * size * channels * (downsamples + 1), target_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.intial_conv(x)
        for layer in self.layers:
            x = layer(x)
            x = F.elu(x)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x