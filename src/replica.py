from registry import Registry, RegistryError, register
from generator import Generator
from discriminator import Discriminator


@register('replica', 'single', default=True)
class SingleReplica:
    def __init__(self, options):
        pass

    def generators(self, options):
        return {
            'generator': Generator.from_options(options)
        }

    def discriminators(self, options):
        return {
            'discriminator': Discriminator.from_options(options)
        }

@register('replica', 'equal')
class EqualReplicas:
    def __init__(self, options):
        self.num_replicas = options.num_replicas

    def generators(self, options):
        result = {}
        for i in range(1, self.num_replicas + 1):
            result[f'generator_{i}'] = Generator.from_options(options)
        return result

    def discriminators(self, options):
        result = {}
        for i in range(1, self.num_replicas + 1):
            result[f'discriminator_{i}'] = Discriminator.from_options(options)
        return result

class Replica:
    @staticmethod
    def from_options(options):
        if options.replica in Registry.keys('replica'):
            cls = Registry.get('replica', options.replica)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.replica} for namespace `replica`')

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('replication logic')
        group.add_argument('--replica', choices=Registry.keys('replica'), default=Registry.default('replica'), help='type of replication')
        group.add_argument('--num-replicas', type=int, default=2, help='number of replicas')