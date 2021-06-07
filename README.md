# Quasi-Global Momentum
This repository is the official implementation of the paper: [Quasi-Global Momentum: Accelerating Decentralized Deep Learning on Heterogeneous Data](https://arxiv.org/abs/2102.04761), appeared in ICML 2021.

## Abstract
Decentralized training of deep learning models
is a key element for enabling data privacy and on-device learning over networks.
In realistic learning scenarios,
the presence of heterogeneity across different clients' local datasets
poses an optimization challenge and may severely deteriorate the generalization performance.

In this paper, we investigate and identify the limitation of several decentralized optimization algorithms
for different degrees of data heterogeneity.
We propose a novel momentum-based method
to mitigate this decentralized training difficulty.
We show in extensive empirical experiments
on various CV/NLP datasets (CIFAR-10, ImageNet, and AG News)
and several network topologies (Ring and Social Network) that
our method is much more robust to the heterogeneity of clients' data than other existing methods,
by a significant improvement in test performance (1% - 20%).


## References
If you use the code, please cite the following paper:

```
@inproceedings{lin2021quasi,
  title={Quasi-global momentum: Accelerating decentralized deep learning on heterogeneous data},
  author={Lin, Tao and Karimireddy, Sai Praneeth and Stich, Sebastian U and Jaggi, Martin},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

## Examples
Please refer to `code/README.md`