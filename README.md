# DELIMIT

This repository contains the [PyTorch](https://pytorch.org) implementation of [our Paper "DELIMIT PyTorch - An extension for Deep Learning in Diffusion Imaging"](#our-paper).

## Contents
* [Installation](#installation)
* [Requirements](#requirements)
* [Usage](#usage)
  * [By Scripts](#by-scripts)
  * [Data Preprocessing](#preprocessing)
 * [Our Paper](#our-paper)

## Requirements
required packages are:
scipy, numpy, torch, pyquaternion

## Installation
`pip install git+https://github.com/SimonKoppers/DELIMIT` 

## Usage
### By Scripts
For simplicity a simple [example jupyter notebook](https://github.com/SimonKoppers/DELIMIT/blob/master/example.ipynb) is provided within the github repository.

### Data Preprocessing
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