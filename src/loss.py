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
import torch.nn as nn
from torch import autograd

from registry import Registry, RegistryError, register, with_option_parser
from utils import YesNoAction


class LossWithLabels:
    def __init__(self, options):
        self.batch_size = options.batch_size
        self.soft_labels = options.soft_labels
        self.noisy_labels = options.noisy_labels
        self.noisy_labels_frequency = options.noisy_labels_frequency
        self.device = torch.device(options.device)

    def _real_labels(self):
        if self.soft_labels:
            labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1)
        else:
            labels = torch.ones(self.batch_size, 1)
        return labels.to(self.device)

    def _fake_labels(self):
        if self.soft_labels:
            labels = torch.FloatTensor(self.batch_size, 1).uniform_(0, 0.1)
        else:
            labels = torch.ones(self.batch_size, 1)
        return labels.to(self.device)

    def real_labels(self):
        if self.noisy_labels and self.noisy_labels_frequency > torch.rand(1).item():
            return self._fake_labels()
        else:
            return self._real_labels()

    def fake_labels(self):
        if self.noisy_labels and self.noisy_labels_frequency > torch.rand(1).item():
            return self._real_labels()
        else:
            return self._fake_labels()

@register('loss', 'gan', default=True)
class GANLoss(LossWithLabels):
    def __init__(self, options, discriminator):
        super().__init__(options)
        self.criterion = nn.BCELoss()
        self.discriminator = discriminator

    def for_generator(self, fake_data, target=None):
        real_labels = self._real_labels()
        return self.criterion(self.discriminator(fake_data), real_labels)

    def for_discriminator(self, real_data, fake_data, target=None):
        real_labels = self.real_labels()
        fake_labels = self.fake_labels()
        real = self.criterion(self.discriminator(real_data), real_labels)
        fake = self.criterion(self.discriminator(fake_data), fake_labels)
        return real + fake

@register('loss', 'wgan')
class WGANLoss:
    def __init__(self, options, discriminator):
        self.device = torch.device(options.device)
        self.discriminator = discriminator

    def for_generator(self, fake_data, target=None):
        return -self.discriminator(fake_data).mean()

    def for_discriminator(self, real_data, fake_data, target=None):
        real = self.discriminator(real_data).mean()
        fake = self.discriminator(fake_data).mean()
        return fake - real

@register('loss', 'wgan-gp')
class WGANGPLoss:
    def __init__(self, options, discriminator):
        self.device = torch.device(options.device)
        self.gradient_penalty_factor = options.gradient_penalty_factor
        self.discriminator = discriminator

    def gradient_penalty(self, real_data, fake_data):
        n_elements = real_data.nelement()
        batch_size, colors, width, height = real_data.size()

        alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
        alpha = alpha.view(batch_size, colors, width, height).to(self.device)

        fake_data = fake_data.view(batch_size, colors, width, height)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)
        critic_interpolates = self.discriminator(interpolates)

        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(critic_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gradient_penalty_factor
        return gradient_penalty

    def for_generator(self, fake_data, target=None):
        return -self.discriminator(fake_data).mean()

    def for_discriminator(self, real_data, fake_data, target=None):
        real = self.discriminator(real_data).mean()
        fake = self.discriminator(fake_data).mean()
        penalty = self.gradient_penalty(real_data, fake_data)
        return fake - real + penalty

@register('loss', 'pix2pix')
class Pix2PixLoss(LossWithLabels):
    def __init__(self, options, discriminator):
        super().__init__(options)
        self.discriminator = discriminator
        self.cross_entropy = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.l1_weight = options.l1_weight

    def for_generator(self, fake_data, target=None):
        real_labels = self._real_labels()
        return self.cross_entropy(self.discriminator(fake_data), real_labels) + self.l1_weight * self.l1(fake_data, target)

    def for_discriminator(self, real_data, fake_data, target=None):
        real_labels = self.real_labels()
        fake_labels = self.fake_labels()
        real = self.cross_entropy(self.discriminator(real_data), real_labels)
        fake = self.cross_entropy(self.discriminator(fake_data), fake_labels)
        return real + fake

@register('loss', 'ebgan')
class EBGANLoss:
    '''
    EBGAN loss from
        Junbo Zhao, Michael Mathieu and Yann LeCun
        Energy-based Generative Adversarial Networks
        https://arxiv.org/abs/1609.03126
    '''
    def __init__(self, options, discriminator):
        self.device = torch.device(options.device)
        self.discriminator = discriminator
        self.threshold = options.ebgan_threshold

    def for_generator(self, fake_data, target=None):
        return self.discriminator(fake_data).mean()

    def for_discriminator(self, real_data, fake_data, target=None):
        real = self.discriminator(real_data).mean()
        fake = (self.threshold - self.discriminator(fake_data)).clamp_min(0).mean()
        return real + fake

class Loss:
    @staticmethod
    def from_options(options, discriminator):
        if options.loss in Registry.keys('loss'):
            cls = Registry.get('loss', options.loss)
            return cls(options, discriminator)
        else:
            raise RegistryError(f'missing value {options.loss} for namespace `loss`')

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        if train:
            group = parser.add_argument_group('loss computation')
            group.add_argument('--loss', choices=Registry.keys('loss'), default=Registry.default('loss'), help='GAN loss')
            group.add_argument('--gradient-penalty-factor', type=float, default=10, help='gradient penalty factor (lambda in WGAN-GP)')
            group.add_argument('--soft-labels', action=YesNoAction, help='use soft labels in GAN loss')
            group.add_argument('--noisy-labels', action=YesNoAction, help='use noisy labels in GAN loss')
            group.add_argument('--noisy-labels-frequency', type=float, default=0.1, help='how often to use noisy labels in GAN loss')
            group.add_argument('--l1-weight', type=float, default=1, help='weight of the L1 distance contribution to the GAN loss')
            group.add_argument('--ebgan-threshold', type=float, default=1, help='threshold (m) in the EBGAN loss')