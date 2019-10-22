from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from registry import Registry, RegistryError, register


@register('data', 'single-image', default=True)
class SingleImage:
    def __init__(self, options):
        transform_list = []
        if options.image_size is not None:
            transform_list.append(transforms.Resize((options.image_size, options.image_size)))
            # transform_list.append(transforms.CenterCrop(options.image_size))
        transform_list.append(transforms.ToTensor())
        if options.image_colors == 1:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        elif options.image_colors == 3:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        transform = transforms.Compose(transform_list)

        if options.dataset == 'mnist':
            dataset = datasets.MNIST(options.data_dir, train=True, download=True, transform=transform)
        elif options.dataset == 'lsun':
            training_class = options.image_class + '_train'
            dataset =  datasets.LSUN(options.data_dir, classes=[training_class], transform=transform)
        else:
            dataset = datasets.ImageFolder(root=options.data_dir, transform=transform)

        self.dataloader = DataLoader(
            dataset,
            batch_size=options.batch_size,
            num_workers=options.loader_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        self.iterator = iter(self.dataloader)

    def next(self):
        return next(self.iterator, None)

    def reset(self):
        self.iterator = iter(self.dataloader)

class Data:
    @staticmethod
    def from_options(options):
        if options.data_format in Registry.keys('data'):
            cls = Registry.get('data', options.data_format)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.data_format} for namespace `data`')

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('data loading')
        group.add_argument('--data-format', choices=Registry.keys('data'), default=Registry.default('data'), help='type of dataset')
        group.add_argument('--data-dir', default='data', help='directory with the images')
        group.add_argument('--dataset', choices=['folder', 'mnist', 'lsun'], default='folder', help='source of the dataset')
        group.add_argument('--image-class', default='bedroom', help='class to train on, only for LSUN')
        group.add_argument('--image-size', type=int, default=64, help='image dimension')
        group.add_argument('--image-colors', type=int, default=3, help='image colors')
        group.add_argument('--batch-size', type=int, default=64, help='batch size')
        group.add_argument('--loader-workers', type=int, default=4, help='number of threads loading data')