# ncsn-pytorch

This repository contains an implementation of [Noise Conditional Score Networks (Song, Ermon 2019)](https://arxiv.org/abs/1907.05600) on MNIST. Among unconditional generative models, NCSN achieves the state-of-the-art inception score and a competitive FID score on the CIFAR-10 dataset.

Some direction was taken from the official implementation by the authors provided at [this repository](https://github.com/ermongroup/ncsn). In particular, the neural network architecture is taken from the original repository.

## Overview

Generative models attempt to generate samples from the true data distribution. Noise conditional score networks do this by modeling the score function of a perturbed distribution very similar to the data distribution. Langevin dynamics is then used to obtain samples from the perturbed distribution via the score function.

Specifically, the perturbed distribution is the data distribution plus i.i.d. Gaussian noise with fixed variance. (For very small variance, the perturbed distribution is visually very similar to the original data distribution.) To facilitate learning, the network learns the perturbed distribution's score function for multiple different variances by minimizing the following loss function 

![](https://github.com/mcclow12/ncsn-pytorch/blob/master/.img/loss.png)

where

![](https://github.com/mcclow12/ncsn-pytorch/blob/master/.img/loss_summand.png)

The function that minimizes this loss has been shown to equal to the score function (for each variance respectively) almost everywhere. Once the network is trained, annealed Langevin dynamics is used to obtain samples from the perturbed distribution with high levels of Gaussian noise, and then the noise is gradually annealed to near zero. In the end, samples from the true data distribution plus an invisible amount of Gussian noise are obtained.

## Sample Images
