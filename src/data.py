import os

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from registry import Registry, RegistryError, register


@register('data', 'single-image', default=True)
class SingleImage:
    '''
    Loads datasets of single images, possibly matched with a corresponding label.

    Takes care of resizing/cropping images, batching, shuffling and distributing
    the work of data loading across a number of workers. Each batch has shape

        B x C x W x H

    where
    * B is the batch size
    * C is the number of channels (1 for B/W images, 3 for colors)
    * W is the image width
    * H is the image height

    The following options can be used to configure it:

    --data-dir: directory storing the dataset
    --dataset: how the underlying dataset is stored
        * `mnist` expects images in the standard MNIST format (idx-ubyte). This
            dataset is introduced in

            Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner
            Gradient-based learning applied to document recognition
            Proceedings of the IEEE, 86(11):2278-2324, November 1998
            http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

            and available on http://yann.lecun.com/exdb/mnist/
        * `lsun` expects the LSUN database. This dataset is introduced in

            Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, Jianxiong Xiao
            LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop
            https://arxiv.org/abs/1506.03365

            and available on https://www.yf.io/p/lsun
        * `folder` expects a folder of images structured as follows

            base-folder
            ├── class1
            │   ├── image1.jpg
            │   ├── image2.png
            │   ├── image3.jpg
            │   └── ...
            ├── class2
            │   └── ...
            └── ...

    --image-size SIZE: crop or resize images to SIZE
    --image-colors: 1 for B/W images, 3 for colors
    --loader-workers: number of threads loading data
    --pin-memory: whether to pin memory to CPU cores for loading data
    --batch-size: number of images inside each batch (the last batch is
        discarded if the dataset size is not divisible evenly by batch_size)
    --image-class: if present, filter the dataset to keep only images with this
        label
    '''
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
            pin_memory=options.pin_memory
        )
        self.iterator = iter(self.dataloader)

    def next(self):
        '''
        Yields the next batch of data and labels. Returns a pair
        (images, labels), where

            * images has shape B x C x W x H
            * labels has shape B x 1
        '''
        return next(self.iterator, None)


    def reset(self):
        '''
        Resets the state of the dataset, call this between epochs.
        '''
        del self.iterator
        self.iterator = iter(self.dataloader)

class ImagePairs(Dataset):
    def __init__(self, basedir, split, transform):
        allowed_extensions = ['.jpg', '.jpeg', '.png']
        self.basedir = basedir
        files = os.listdir(basedir)
        files = [f for f in files if os.path.splitext(f)[1].lower() in datasets.folder.IMG_EXTENSIONS]
        self.files = [os.path.join(basedir, f) for f in files]
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = datasets.folder.default_loader(self.files[index])
        w, h = image.size
        if self.split == 'horizontal':
            image1 = image.crop((0, 0, w // 2, h))
            image2 = image.crop((w // 2, 0, w, h))
            return self.transform(image1), self.transform(image2)
        elif self.split == 'vertical':
            image1 = image.crop((0, 0, w, h // 2))
            image2 = image.crop((0, h // 2, w, h))
            return self.transform(image1), self.transform(image2)

@register('data', 'pair-of-images')
class PairOfImages:
    '''
    Loads datasets of pairs of images.

    Takes care of resizing/cropping images, batching, shuffling and distributing
    the work of data loading across a number of workers. Each batch is made of two
    copies, each having shape

        B x C x W x H

    where
    * B is the batch size
    * C is the number of channels (1 for B/W images, 3 for colors)
    * W is the image width
    * H is the image height

    The following options can be used to configure it:

    --data-dir: directory storing the dataset
    --image-size SIZE: crop or resize images to SIZE
    --image-colors: 1 for B/W images, 3 for colors
    --split: how to split images into pairs
        * `horizontal` expects a folder of images side by side
        * `vertical` expects a folder of images one over the other
    --loader-workers: number of threads loading data
    --pin-memory: whether to pin memory to CPU cores for loading data
    --batch-size: number of images inside each batch (the last batch is
        discarded if the dataset size is not divisible evenly by batch_size)
    '''
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

        dataset = ImagePairs(options.data_dir, split=options.split, transform=transform)

        self.dataloader = DataLoader(
            dataset,
            batch_size=options.batch_size,
            num_workers=options.loader_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=options.pin_memory
        )
        self.iterator = iter(self.dataloader)

    def next(self):
        '''
        Yields the next batch of data and labels. Returns a pair
        (imagesA, imagesB), both having shape

            B x C x W x H
        '''
        batch = next(self.iterator, None)
        return batch

    def reset(self):
        '''
        Resets the state of the dataset, call this between epochs.
        '''
        del self.iterator
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
        group.add_argument('--split', choices=['horizontal', 'vertical'], help='how to split an image pair')
        group.add_argument('--batch-size', type=int, default=64, help='batch size')
        group.add_argument('--loader-workers', type=int, default=4, help='number of threads loading data')
        group.add_argument('--pin-memory', action='store_true', help='pin memory to CPU cores for loading data')