# DDSM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ChristophReich1996/MaxViT/blob/master/LICENSE)

Unofficial **PyTorch** reimplementation of the
paper [Dirichlet Diffusion Score Model for Biological Sequence Generation](https://arxiv.org/pdf/2305.10699.pdf)
by Avdeyev et al.

<p align="center">
  <img src="maxvit.png"  alt="1" width = 674px height = 306px >
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
In addition it provides a notebook to presample noise and speed up the computation.

To accesses the named weights of the network which are not recommended being used with weight decay
call `nwd: Set[str] = network.no_weight_decay()`.

In case you want to use a custom configuration you can use the `MaxViT` class. The constructor method takes the
following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| in_channels | Number of input channels to the convolutional stem. Default 3 | int, optional |
| depths | Depth of each network stage. Default (2, 2, 5, 2) | Tuple[int, ...], optional |
| channels | Number of channels in each network stage. Default (64, 128, 256, 512) | Tuple[int, ...], optional |
| num_classes | Number of classes to be predicted. Default 1000 | int, optional |
| embed_dim | Embedding dimension of the convolutional stem. Default 64 | int, optional |
| num_heads | Number of attention heads. Default 32 | int, optional |
| grid_window_size | Grid/Window size to be utilized. Default (7, 7) | Tuple[int, int], optional |
| attn_drop | Dropout ratio of attention weight. Default: 0.0 | float, optional |
| drop | Dropout ratio of output. Default: 0.0 | float, optional |
| drop_path | Dropout ratio of path. Default: 0.0 | float, optional |
| mlp_ratio | Ratio of mlp hidden dim to embedding dim. Default: 4.0 | float, optional |
| act_layer | Type of activation layer to be utilized. Default: nn.GELU | Type[nn.Module], optional |
| norm_layer | Type of normalization layer to be utilized. Default: nn.BatchNorm2d | Type[nn.Module], optional |
| norm_layer_transformer | Normalization layer in Transformer. Default: nn.LayerNorm | Type[nn.Module], optional |
| global_pool | Global polling type to be utilized. Default "avg" | str, optional |

## Reference

```bibtex
@article{avdeyev2023dirichlet,
  title={Dirichlet Diffusion Score Model for Biological Sequence Generation},
  author={Avdeyev, Pavel and Shi, Chenlai and Tan, Yuhao and Dudnyk, Kseniia and Zhou, Jian},
  journal={arXiv preprint arXiv:2305.10699},
  year={2023}
}
```
