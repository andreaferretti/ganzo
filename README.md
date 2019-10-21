# Ganzo

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
game = Game.from_options(options, generator, discriminator, loss)

for _ in range(options.epochs):
    losses = game.run_epoch(data, noise)
    statistics.log(losses)
    snapshot.save(data, noise, generator)
```

You can write your own training script by adapting this basic structure. It
should be easy: the source of `ganzo.py` is less than 100 lines, most of which
deal with handling the case of restoring a training session.

## Components

### Data

### Generator

### Discriminator

### Loss

### Noise

### Statistics

### Snapshot

### Game

## Extending Ganzo

## Using Ganzo as a library