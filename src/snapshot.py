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
import os
import math
import importlib

import torch
import torchvision

from registry import Registry, RegistryError, register, with_option_parser
from utils import YesNoAction

class BaseSnapshot:
    def __init__(self, options):
        self.device = torch.device(options.device)
        self.batch_size = options.batch_size
        self.image_size = options.image_size
        self.image_colors = options.image_colors
        self.sample_every = options.sample_every
        self.snapshot_size = options.snapshot_size
        self.epoch = options.start_epoch
        self.noise = None
        self.sample_from_fixed_noise = options.sample_from_fixed_noise
        self.snapshot_translate = options.snapshot_translate
        self.nrows = int(math.sqrt(self.snapshot_size))

    def _samples_from_noise(self, noiseloader, generator):
        if self.sample_from_fixed_noise and self.noise is None:
            self.noise = noiseloader.next()[:self.snapshot_size]

        if self.sample_from_fixed_noise:
            noise = self.noise
        else:
            noise = noiseloader.next()[:self.snapshot_size]
        samples = generator(noise).view(self.snapshot_size, self.image_colors, self.image_size, self.image_size)
        return samples * 0.5 + 0.5

    def _samples_from_data(self, dataloader, generator):
        minibatch = dataloader.next()
        if minibatch is None: # end of batch
            dataloder.reset()
            minibatch = dataloader.next()
        inputs, outputs = minibatch
        inputs = inputs[:self.snapshot_size].to(self.device)
        outputs = outputs[:self.snapshot_size].to(self.device)
        generated = generator(inputs)
        samples = torch.cat((generated, inputs, outputs), dim=3)
        return samples * 0.5 + 0.5

    def _samples(self, dataloder, noiseloader, generator):
        if self.snapshot_translate:
            return self._samples_from_data(dataloder, generator)
        else:
            return self._samples_from_noise(noiseloader, generator)

@register('snapshot', 'folder', default=True)
class FolderSnapshot(BaseSnapshot):
    def __init__(self, options):
        super().__init__(options)
        self.base_dir = os.path.join(options.output_dir, options.experiment)
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, dataloder, noiseloader, generators):
        if self.epoch % self.sample_every == 0:
            samples = None
            for generator in generators:
                more_samples = self._samples(dataloder, noiseloader, generator)
                if samples is None:
                    samples = more_samples
                else:
                    samples = torch.cat((samples, more_samples), dim=0)

            torchvision.utils.save_image(samples, os.path.join(self.base_dir, f'epoch_{self.epoch}.png'), nrow=self.nrows, padding=2)
        self.epoch += 1

tensorboard_enabled = importlib.util.find_spec('tensorboardX') is not None

if tensorboard_enabled:
    from tensorboardX import SummaryWriter

    @register('snapshot', 'tensorboard')
    class TensorBoardSnapshot(BaseSnapshot):
        def __init__(self, options):
            super().__init__(options)
            self.writer = SummaryWriter(os.path.join(options.output_dir, options.experiment))
            self.epoch = options.start_epoch

        def save(self, dataloder, noiseloader, generator):
            if self.epoch % self.sample_every == 0:
                samples = None
                for generator in generators:
                    more_samples = self._samples(dataloder, noiseloader, generator)
                    if samples is None:
                        samples = more_samples
                    else:
                        samples = torch.cat((samples, more_samples), dim=0)

                grid = torchvision.utils.make_grid(samples, nrow=self.nrows, padding=2)
                self.writer.add_image('images', grid, self.epoch)
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
    @with_option_parser
    def add_options(parser, train):
        group = parser.add_argument_group('result snapshotting')
        group.add_argument('--save-images-as', choices=Registry.keys('snapshot'), default=Registry.default('snapshot'), help='how to save the output images')
        group.add_argument('--output-dir', default='output', help='directory where to store the generated images')
        group.add_argument('--snapshot-size', type=int, default=16, help='how many images to generate for each sample (must be <= batch-size)')
        group.add_argument('--sample-every', type=int, default=10, help='how often to sample images (in epochs)')
        group.add_argument('--sample-from-fixed-noise', action=YesNoAction, help='always use the same input noise when sampling')
        group.add_argument('--snapshot-translate', action=YesNoAction, help='generate snapshots for an image translation task')