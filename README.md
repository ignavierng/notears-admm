# Federated Structure Learning with Continuous Optimization

This repository contains an implementation of the structure learning methods described in ["Towards Federated Bayesian Network Structure Learning with Continuous Optimization"](https://arxiv.org/abs/2110.09356). 

If you find it useful, please consider citing:
```bibtex
@inproceedings{Ng2022federated,
  author = {Ng, Ignavier and Zhang, Kun},
  title = {Towards Federated Bayesian Network Structure Learning with Continuous Optimization},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  year = {2022},
}
```

## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `python-igraph`
- `torch`

## Running NOTEARS(-MLP) with ADMM
- See [examples/linear.ipynb](https://github.com/ignavierng/notears-admm/blob/master/examples/linear.ipynb) and [examples/nonlinear.ipynb](https://github.com/ignavierng/notears-admm/blob/master/examples/nonlinear.ipynb) for a demo in the linear and nonlinear cases, respectively.


## Acknowledgments
- A large part of the code, including some helper functions, is obtained and modified from the implementation of [NOTEARS](https://github.com/xunzheng/notears), and we are grateful to the authors of NOTEARS for releasing their code.
- The code to post-process the output is modified and obtained from the implementation of [GOLEM](https://github.com/ignavierng/golem).