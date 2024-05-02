# DCT: A Conditional Independence Test in the Presence of Discretization
This repository contains implementation for paper : **A Condiional Independence Test in the Presence of Discretization** \[[arXiv\]](https://arxiv.org/abs/2404.17644)

DCT is a conditional independence test specifically designed for the scenario that only discretized version of variables available. Specifically, DCT tries to recover the covariance matrix $\Sigma$ of the original continous variables and construct the relationship $\hat{\Sigma} - \Sigma$, which corresponds to the independence relationship. Correspondingly, DCT uses nodewise regression to construct $\hat{\Omega} - \Omega$, the conditional independence relationship.

## How to Install 
run the code 

`conda env create -f environment.yml`

Then you will have a conda environment named 'causal'. You can further activate the environment by running

`conda activate causal`

## How to Use 

We provide two examples running the test in `example_to_use,ipynb` and running the PC algorithm with DCT as the test in `example_to_use_pc.ipynb`. Our core algorithm is implemented at `causal_learn.causallearn.utils.DisTestUtil.py`.


