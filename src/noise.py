import torch

from registry import Registry, RegistryError, register


@register('noise', 'gaussian', default=True)
class GaussianNoise:
    def __init__(self, options):
        self.state_size = options.state_size
        self.batch_size = options.batch_size
        self.device = options.device

    def next(self):
        return torch.randn(self.batch_size, self.state_size).to(self.device)

class Noise:
    @staticmethod
    def from_options(options):
        if options.noise in Registry.keys('noise'):
            cls = Registry.get('noise', options.noise)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.noise} for namespace `noise`')

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('noise generation')
        group.add_argument('--noise', choices=Registry.keys('noise'), default=Registry.default('noise'), help='type of noise')
        group.add_argument('--state-size', type=int, default=128, help='state size')