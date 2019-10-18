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

## Components

### Data

### Generator

### Discriminator

### Loss

### Noise

### Statistics

### Snapshot

### Game

## Custom components

## Using Ganzo as a library