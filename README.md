# DELIMIT

This repository contains the [PyTorch](https://pytorch.org) implementation of [our Paper "DELIMIT PyTorch - An extension for Deep Learning in Diffusion Imaging"](#our-paper).

## Contents
* [Installation](#installation)
* [Usage](#usage)
  * [By Scripts](#by-scripts)
  * [From Python](#from-python)
  * [Pretrained Weights](#pretrained-weights)
 * [Our Paper](#our-paper)

## Installation
`pip install git+https://github.com/justusschock/shapenet` 

## Usage
### By Scripts
For simplicity a simple example script is provided within the github repository

#### Data Preprocessing
In order to apply Spherical Harmonic Transformations, or any kind of convolution, the diffusion signal needs to be divided by its b=0 measurement.

## Our Paper
If you use our Code for your own research, please cite our paper:
```
@article{Koppers2018,
title = {DELIMIT PyTorch - An extension for Deep Learning in Diffusion Imaging},
author = {Simon Koppers and Dorit Merhof},
year = {2018},
journal = {arXiV preprint}
}
```
The Paper is available as [PDF on arXiv](https://arxiv.org/abs/1808.01517).