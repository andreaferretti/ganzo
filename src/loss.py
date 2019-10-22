import torch
import torch.nn as nn
from torch import autograd

from registry import Registry, RegistryError, register


@register('loss', 'gan', default=True)
class GANLoss:
    def __init__(self, options, discriminator):
        self.criterion = nn.BCELoss()
        self.discriminator = discriminator
        self.batch_size = options.batch_size
        self.soft_labels = options.soft_labels
        self.noisy_labels = options.noisy_labels
        self.noisy_labels_frequency = options.noisy_labels_frequency
        self.device = torch.device(options.device)

    def real_labels(self):
        if self.soft_labels:
            labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1)
        else:
            labels = torch.ones(self.batch_size, 1)
        return labels.to(self.device)

    def fake_labels(self):
        if self.soft_labels:
            labels = torch.FloatTensor(self.batch_size, 1).uniform_(0, 0.1)
        else:
            labels = torch.ones(self.batch_size, 1)
        return labels.to(self.device)

    def for_generator(self, fake_data, labels=None):
        real_labels = self.real_labels()
        return self.criterion(self.discriminator(fake_data), real_labels)

    def for_discriminator(self, real_data, fake_data, labels=None):
        if self.noisy_labels and self.noisy_labels_frequency > torch.rand(1).item():
            real_labels = self.fake_labels()
        else:
            real_labels = self.real_labels()
        if self.noisy_labels and self.noisy_labels_frequency > torch.rand(1).item():
            fake_labels = self.real_labels()
        else:
            fake_labels = self.fake_labels()
        real = self.criterion(self.discriminator(real_data), real_labels)
        fake = self.criterion(self.discriminator(fake_data), fake_labels)
        return real + fake

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

    def for_generator(self, fake_data, labels=None):
        return -self.discriminator(fake_data).mean()

    def for_discriminator(self, real_data, fake_data, labels=None):
        real = self.discriminator(real_data).mean()
        fake = self.discriminator(fake_data).mean()
        penalty = self.gradient_penalty(real_data, fake_data)
        return fake - real + penalty

class Loss:
    @staticmethod
    def from_options(options, discriminator):
        if options.loss in Registry.keys('loss'):
            cls = Registry.get('loss', options.loss)
            return cls(options, discriminator)
        else:
            raise RegistryError(f'missing value {options.loss} for namespace `loss`')

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('loss computation')
        group.add_argument('--loss', choices=Registry.keys('loss'), default=Registry.default('loss'), help='GAN loss')
        group.add_argument('--gradient-penalty-factor', type=float, default=10, help='gradient penalty factor (lambda in WGAN-GP)')
        group.add_argument('--soft-labels', action='store_true', help='use soft labels in GAN loss')
        group.add_argument('--noisy-labels', action='store_true', help='use noisy labels in GAN loss')
        group.add_argument('--noisy-labels-frequency', type=float, default=0.1, help='how often to use noisy labels in GAN loss')