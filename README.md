# Dirichlet Diffusion Score Model for Biological Sequence 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/MaxViT/blob/master/LICENSE)

Unofficial **PyTorch** reimplementation of the
paper [Dirichlet Diffusion Score Model for Biological Sequence Generation](https://arxiv.org/pdf/2305.10699.pdf)
by Avdeyev et al.

<p align="center">
  <img src="ddsm.webp"  alt="1" width = 640px height = 514px >
</p>

Figure taken from [paper](https://arxiv.org/pdf/2305.10699.pdf).

## Usage

This implementation provides example notebooks for training evaluation DDSM models for Bin-MNIST and MNIST data. In these notebooks you can simply load our provided configs train or retrain your models with them as follow.

```python
import os
from lib.models.networks import MNISTScoreNet
import lib.utils.bookkeeping as bookkeeping
from lib.config.config_bin_mnist import get_config

train_resume = False
train_resume_path = 'path/to/saved/models'

if not train_resume:
    config = get_config()
    bookkeeping.save_config(config, config.saving.save_location)

else:
    path = train_resume_path
    date = "date"
    config_name = "config_001.yaml"
    config_path = os.path.join(path, date, config_name)
    config = bookkeeping.load_config(config_path)

model = MNISTScoreNet(ch=config.model.ch, ch_mult=config.model.ch_mult, attn=config.model.attn, num_res_blocks=config.model.num_res_blocks, dropout=0.1, time_dependent_weights=time_dependent_weights)
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
optimizer = Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
n_iter = 0
state = {"model": model, "optimizer": optimizer, "n_iter": 0}

if train_resume:
    checkpoint_path = config.saving.checkpoint_path
    model_name = 'model_name.pt'
    checkpoint_path = os.path.join(path, date, model_name)
    state = bookkeeping.load_state(state, checkpoint_path, device)
    config.training.n_iter = 100000
    config.sampler.sampler_freq = 5000
    config.saving.checkpoint_freq = 1000

```
Further, I provide a notebook to presample noise and speed up the computation.

## Note
I additionally included the U-Net model from paper [Dirichlet Diffusion Score Model for Biological Sequence Generation](https://arxiv.org/pdf/2107.03006.pdf) which could be more suitable to the MNIST data.

## Reference

```bibtex
@article{avdeyev2023dirichlet,
  title={Dirichlet Diffusion Score Model for Biological Sequence Generation},
  author={Avdeyev, Pavel and Shi, Chenlai and Tan, Yuhao and Dudnyk, Kseniia and Zhou, Jian},
  journal={arXiv preprint arXiv:2305.10699},
  year={2023}
}
```
