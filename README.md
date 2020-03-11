# ncsn-pytorch

This repository contains an implementation of [Noise Conditional Score Networks (Song, Ermon 2019)](https://arxiv.org/abs/1907.05600) on MNIST. Among unconditional generative models, NCSN achieves the state-of-the-art inception score and a competitive FID score on the CIFAR-10 dataset.

Some direction was taken from the official implementation by the authors provided at [this repository](https://github.com/ermongroup/ncsn). In particular, the neural network architecture is taken from the original repository.

## Overview

Generative models attempt to generate samples from the true data distribution. Noise conditional score networks do this by modeling the score function of a perturbed distribution very similar to the data distribution. Langevin dynamics is then used to obtain samples from the perturbed distribution.

Specifially, the perturbed distribution is the data distribution plus i.i.d. Gaussian noise with fixed variance. (For very small variance, the perturbed distribution is very similar to the original data distribution.) To facilitate learning, the network learns the perturbed distribution's score function for multiple different variances by minimizing the following loss function

where

