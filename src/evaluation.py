import math

from registry import Registry, RegistryError, register


@register('evaluation', 'latest', default=True)
class Latest:
    def __init__(self, options):
        pass

    def has_improved(self, losses):
        return True

@register('evaluation', 'generator-loss')
class GeneratorLoss:
    def __init__(self, options):
        self.best_loss = math.inf

    def has_improved(self, losses):
        loss = losses['generator']
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        else:
            return False

class Evaluation:
    @staticmethod
    def from_options(options):
        if options.evaluation_criterion in Registry.keys('evaluation'):
            cls = Registry.get('evaluation', options.evaluation_criterion)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.evaluation_criterion} for namespace `evaluation`')

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('model evaluation')
        group.add_argument('--evaluation-criterion', choices=Registry.keys('evaluation'), default=Registry.default('evaluation'), help='the criterion to evaluate model improvement')