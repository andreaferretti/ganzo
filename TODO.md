# Models

* Pix2pix
* CycleGAN
* StyleGAN
* StackGAN

# Experiments

For all existing experiments:

* Find correct hyperparameters and settings as used in papers
* Add a markdown file describing the experiment
* Commit example of generated output

# Components

* PairOfImages data loader
* Colorization data loader (transform to luminance space etc)
* New component type: Alternator - to decide when to switch between G and D
* Logging to file
* Make FC and Conv Gnerator and Discriminator more flexible
* Energy based losses
* Add FashionMNIST to data loading
* Add a new component type to decide how to initialize layers?

# Features

* Support multiple generators/discriminators (for instance for CycleGAN)
* Find a way to add more metrics to the `losses` dict in order to write more
  complex or specific Evaluation classes (e.g. monitor the W1 distance)
* Make sure that Tensorboard logs and snapshots can be used simultaneously
* Conditioned GAN
* New script to generate images from trained models
* Make it run under Pipenv
* Store the current epoch somewhere, so that if we restore an experiment, the
  generated images do not overlap

# Optimization

* Make use of multiple GPUs
* Understand why we take so many CPUs (relevant: https://github.com/pytorch/pytorch/issues/22866)
* Saturate GPUs

# Documentation

* Add documentation comments to all classes/methods
* Describe models
* Finish README with list of components
* Document the public API used for each type of component