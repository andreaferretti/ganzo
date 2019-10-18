import os

import torchvision

from registry import Registry, RegistryError, register


@register('snapshot', 'folder', default=True)
class FolderSnapshot:
    def __init__(self, options):
        self.base_dir = os.path.join(options.output_dir, options.experiment)
        os.makedirs(self.base_dir, exist_ok=True)
        self.batch_size = options.batch_size
        self.image_size = options.image_size
        self.image_colors = options.image_colors
        self.sample_every = options.sample_every
        self.snapshot_size = options.snapshot_size
        self.epoch = 1
        self.noise = None
        self.sample_from_fixed_noise = options.sample_from_fixed_noise

    def save(self, dataloder, noiseloader, generator):
        if self.sample_from_fixed_noise and self.noise is None:
            self.noise = noiseloader.next()[:self.snapshot_size]

        if self.epoch % self.sample_every == 0:
            if self.sample_from_fixed_noise:
                noise = self.noise
            else:
                noise = noiseloader.next()[:self.snapshot_size]
            samples = generator(noise).view(self.batch_size, self.image_colors, self.image_size, self.image_size)
            samples = samples * 0.5 + 0.5

            torchvision.utils.save_image(samples, os.path.join(self.base_dir, f'epoch_{self.epoch}.png'), nrow=8, padding=2)
        self.epoch += 1

class Snapshot:
    @staticmethod
    def from_options(options):
        if options.save_images_as in Registry.keys('snapshot'):
            cls = Registry.get('snapshot', options.save_images_as)
            return cls(options)
        else:
            raise RegistryError(f'missing value {options.save_images_as} for namespace `snapshot`')

    @staticmethod
    def add_options(parser):
        group = parser.add_argument_group('result snapshotting')
        group.add_argument('--save-images-as', choices=Registry.keys('snapshot'), default=Registry.default('snapshot'), help='how to save the output images')
        group.add_argument('--output-dir', default='output', help='directory where to store the generated images')
        group.add_argument('--snapshot-size', type=int, default=16, help='how many images to generate for each sample (must be <= batch-size)')
        group.add_argument('--sample-every', type=int, default=10, help='how often to sample images (in epochs)')
        group.add_argument('--sample-from-fixed-noise', action='store_true', help='always use the same input noise when sampling')
