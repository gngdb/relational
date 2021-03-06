# A Minimal Relational Function
> PyTorch/Numpy agnostic function implementing the relational block from "A simple neural network module for relational reasoning".


I wrote this to learn how to use [nbdev][]. I'm pretty sure it's correct but it only implements the core function for using relational networks and none of the other stuff (such as `nn.Module` classes etc) that [Kai included in the pull request][kai].

The original paper can be found [here](https://arxiv.org/abs/1706.01427).

[kai]: https://github.com/pytorch/pytorch/pull/2105
[nbdev]: https://nbdev.fast.ai

## Install

`pip install relational`

## How to use

This can be used to implement a relational network in PyTorch. An example would be something like:

```python
from relational.core import relation
```

```python
import torch
import torch.nn as nn
```

```python
class SetNet(nn.Module):
    def __init__(self, datadim, n_hidden):
        super(SetNet, self).__init__()
        self.n_hidden = n_hidden
        self.g = nn.Sequential(nn.Linear(datadim*2, n_hidden), 
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))
        self.f = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, n_hidden))

    def forward(self, x):
        n, t, d = x.size()
        x = relation(x, self.g, reduction='mean')
        return self.f(x)
```

```python
x = torch.randn(4, 8, 16)
setnet = SetNet(x.size(2), 10)
setnet(x).size()
```




    torch.Size([4, 10])



## Citation

The original NeurIPS paper can be found [here](https://papers.nips.cc/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html) and here's the bibtex so you can copy it:

```
@inproceedings{santoro2017simple,
 author = {Santoro, Adam and Raposo, David and Barrett, David G and Malinowski, Mateusz and Pascanu, Razvan and Battaglia, Peter and Lillicrap, Timothy},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {A simple neural network module for relational reasoning},
 url = {https://proceedings.neurips.cc/paper/2017/file/e6acf4b0f69f6f6e60e9a815938aa1ff-Paper.pdf},
 volume = {30},
 year = {2017}
}
```
