Ganzo
=====

Ganzo is a framework to implement, train and run different types of
[GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network),
based on [PyTorch](https://pytorch.org/).

It aims to unify different types of GAN architectures, loss functions and
generator/discriminator game strategies, as well as offer a collection of
building blocks to reproduce popular GAN papers.

The guiding principles are:

* be fully runnable and configurable from the command line or from a JSON file
* be usable as a library
* allow for reproducible experiments

## Installing

The only hard dependencies for Ganzo are PyTorch and TorchVision. For instance,
in Conda you can create a dedicated environment with:

```
conda create -n ganzo python=3.7 pytorch torchvision -c pytorch
```

If available, Ganzo supports [TensorBoardX](https://github.com/lanpa/tensorboardX).
This is detected at runtime, so Ganzo can run with or without it.

If you want to use [LMDB]https://lmdb.readthedocs.io/) datasets such as
[LSUN](https://github.com/fyu/lsun), you will also need that dependency:

```
conda install python-lmdb
```

To download the LSUN image dataset, use the instructions in the linked
repository.

## Running

Ganzo can be used either from the command line or as a library. Each
component in Ganzo can be imported and used separately, but everything is
written in a way that allows for full customization on the command line or
through a JSON file.

To see all available options, run `python src/ganzo.py --help`. Most options
are relative to a component of Ganzo (data loading, generator models, discriminator
models, loss functions, logging and so on) and are explained in detail together
with the relative component.

Some options are global in nature:

* `experiment` this is the name of the experiment. Models, outputs and so on
  are saved using this name. You can choose a custom experiment name, or let
  Ganzo use a hash generated based on the options passed.
* `device` this is the device name (for instance `cpu` or `cuda`). If left
  unspecified, Ganzo will autodetect the presence of a GPU and use it if
  available.
* `epochs` number of epochs for training.
* `model-dir` path to the directory where models are saved. Models are further
  namespaced according to the experiment name.
* `restore` if this flag is set, and an experiment with this name has already
  been run, Ganzo will reload existing models and keep running from there.
* `delete` if this flag is set, and an experiment with this name has already
  been run, Ganzo will delete everything there and start from scratch. Note that
  by default Ganzo will ask on the command line what to do, unless at least one
  flag among `delete` and `restore` is active.
* `seed` this is the seed for PyTorch random number generator. This is used
  in order to reproduce results.

## Architecture

Ganzo is structured into modules that handles different concerns: data loading,
generators, discriminators, loss functions and so on. Each of these modules
defines some classes that can be exported and used on their own, or can be
used fully through configuration, for instance from a JSON file.

To do so, each module defines a main class (for instance, `loss.Loss`, or
`data.Data`), that has two static methods:

* `from_options(options, *args)` initializes a class from an object representing
  the options. This object is a [argparse.Namespace](https://docs.python.org/3/library/argparse.html#the-namespace-object) object that is obtained by parsing
  command line options or JSON configuration files. Most classes only require
  the `options` object in order to be instantiated, but in some cases, other
  arguments are passed in as well -- for instance the loss function requires
  the discriminator in order to compute generator loss.
* `add_options(parser)` takes as input an [argparse.ArgumentParser](https://docs.python.org/3/library/argparse.html#argumentparser-objects)
  object, and adds a set of arguments that are relevant to the specific module.
  This is typically done by adding an argument group. Of course, options are
  not constrained to be used by the module that introduces them: for instance,
  `data.Data` adds the argument `batch-size`, that is used by many other modules.

Having each module defined by configuration allows Ganzo to wire them without
knowing much of the specifics. The main loop run at training time looks like
this:

```python
data = Data.from_options(options)
generator = Generator.from_options(options)
discriminator = Discriminator.from_options(options)
loss = Loss.from_options(options, discriminator)
noise = Noise.from_options(options)
statistics = Statistics.from_options(options)
snapshot = Snapshot.from_options(options)
evaluation = Evaluation.from_options(options)
game = Game.from_options(options, generator, discriminator, loss)

for _ in range(options.epochs):
    losses = game.run_epoch(data, noise)
    statistics.log(losses)
    snapshot.save(data, noise, generator)
    if evaluation.has_improved(losses):
        # save models
```

You can write your own training script by adapting this basic structure. It
should be easy: the source of `ganzo.py` is less than 100 lines, most of which
deal with handling the case of restoring a training session.

## Components

The following goes in detail about the various components defined by Ganzo:
their role, the classes that they export, and the options that they provide
for configuration.

### Data

This module handles the loading of the image datasets.

Datasets can come in various formats: single image datasets (with or without a
labeling) and datasets of input/output pairs. A single image dataset can be used
to train a standard GAN, while having the labels can be used for conditioned
GANs. Datasets of pairs can be used for tasks of image to image translation,
such as super resolution or colorization.

Also, datasets can be stored in different ways. The simplest one is a folder
containing images, possibly split into subfolders representing categories. But
some datasets are stored in a custom way - for instance MNIST or LSUN.

The module `data` defines the following classes:

* `SingleImageData` TODO describe it

The module `data` exports the following options:

* `data-format`: the format of the dataset, such as `single-image`
* `data-dir`: the (input) directory where the images are stored
* `dataset`: the type of dataset, such as `folder`, `mnist` or `lsun`
* `image-class`: this can be used to filter the dataset, by restricting it to the
  images having this label
* `image-size`: if present, images are resized to this size
* `image-colors`: can be 1 or 3 for B/W or color images
* `batch-size`: how many images to consider in each minibatch
* `loader-workers`: how many workers to use to load data in background

### Generator

### Discriminator

### Loss

### Noise

### Statistics

### Snapshot

### Evaluation

### Game

## Extending Ganzo

Ganzo can be extended by defining your custom modules. To do this, you do not
need to write your training script (although this is certainly doable). If
you want to define custom components and let Ganzo take advantage of them,
you need to follow four steps:

* write your custom component (this can a be a data loader, a generator, a
  loss function...). You will need to make sure that it can be initialized
  via an `option` object and that it exposes the same public methods
  as the other classes (for instance, a loss function exposes two public methods
  `def for_generator(self, fake_data, labels=None)` and `def for_discriminator(self, real_data, fake_data, labels=None)`)
* let Ganzo be aware of your component by registering it
* add an enviroment variable to make Ganzo find your module
* optionally, add your custom options to the argument parser.

To make this possible, Ganzo uses a registry, that can be found under
`registry.py`. This exports the `Registry` singleton and the `register`
decorator function (also, the `RegistryException` class, which you should not
need).

### Custom components and the registry

When you write your custom component, you need to register it in the
correct namespace. This can be done with the `Registry.add` function, or
more simply with the `register` decorator. In both cases you will need to
provide the namespace (e.g. `loss` for a loss function) and the name of your
component (this is your choice, just make sure not to collide with existing
names). For instance, registering a `CustomLoss` class can be done like this:

```python
from registry import register

@register('loss', 'custom-loss')
class CustomLoss:
    # make sure that your __init__ method has the same signature
    # as the existing components
    def __init__(self, options, discriminator):
        # initialization logic here

    # more class logic here, for instance the public API
    def for_generator(self, fake_data, labels=None):
        pass

    def for_discriminator(self, real_data, fake_data, labels=None):
        pass
```

This can also be done more explicitly by adding your class to the registry:

```python
from registry import Registry

class CustomLoss:
    # make sure that your __init__ method has the same signature
    # as the existing components
    def __init__(self, options, discriminator):
        # initialization logic here

    # more class logic here, for instance the public API
    def for_generator(self, fake_data, labels=None):
        pass

    def for_discriminator(self, real_data, fake_data, labels=None):
        pass

Registry.add('loss', 'custom-loss', CustomLoss)
```

You will then be able to select your custom loss function by passing the
command line argument `--loss custom-loss`. Both `register` and `Registry.add`
take a flag `default` which means that the registered component will be
selected as default when the corresponding command line option is missing.
This can be used only once, though, and it is already taken by the default
components of Ganzo, so you should **not** pass `default=True` while
registering your component.

### Finding your module

Of course, at this point Ganzo is not aware that your module exists, or that
it defines and register new components. You need to make sure that Ganzo
actually imports your module, so that your custom logic is run. This cannot
be done with a flag on the command line: the reason is that you are able to
add custom options to the argument parser, so your modules must be found
**before** reading the command line options.

To do this, we use an environment variable `GANZO_LOAD_MODULES`, which
should contain a comma-separated list of python modules that you want to
import before Ganzo starts. These modules should be on the Python path, so
that they can be imported by name. For instance, if you have defined your loss
inside `custom.py`, you can call Ganzo like this:

```
GANZO_LOAD_MODULES=custom python src/ganzo.py # more options here
```

### Customizing options

Probably, your custom component will need some degrees of freedom that are not
applicable in general, and this means that you need to be able to extend
the command line parser. To do this, import the decorator `with_option_parser`
from the `registry` module and define a function that takes a parser argument
and extends it.

It is advised to namespace you options into their own argument group, in order
to make the help message more understandable. An example would be

```python
from registry import with_option_parser

@with_option_parser
def add_my_custom_options(parser):
    group = parser.add_argument_group('custom')
    group.add_argument('foos', type=int, default=3, help='the number of foos')
```

## Using Ganzo as a library