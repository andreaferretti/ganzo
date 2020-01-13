from registry import Registry, RegistryError, register, with_option_parser
from generator import Generator
from discriminator import Discriminator


@register('replica', 'single', default=True)
class SingleReplica:
    def __init__(self, options):
        pass

    def generators(self, options):
        return [Generator.from_options(options)]

    def discriminators(self, options):
        return [Discriminator.from_options(options)]

@register('replica', 'equal')
class EqualReplicas:
    def __init__(self, options):
        self.num_replicas = options.num_replicas

    def generators(self, options):
        return [Generator.from_options(options, i) for i in range(self.num_replicas)]

    def discriminators(self, options):
        return [Discriminator.from_options(options, i) for i in range(self.num_replicas)]

class Replica:
    @staticmethod
    def from_options(options):
        if options.replica in Registry.keys('replica'):
            cls = Registry.get('replica', options.replica)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.replica} for namespace `replica`')

    @staticmethod
    @with_option_parser
    def add_options(parser, train):
        group = parser.add_argument_group('replication logic')
        group.add_argument('--replica', choices=Registry.keys('replica'), default=Registry.default('replica'), help='type of replication')
        group.add_argument('--num-replicas', type=int, default=2, help='number of replicas')