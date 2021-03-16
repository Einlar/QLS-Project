# QLS Project - A. A. 2020/21
## A bio-inspired learning rule for deep architectures

Project inspired by [1]: "Unsupervised learning by competing hidden units" (2019), D. Krotov & J. Hopfield.

The code is organized into 4 Jupyter Notebooks:
- `00_Differential_Algorithm.ipynb` explores a direct numerical simulation of the differential equations in [1], which was not present in the original paper. 

    The point is to observe empirically that they do indeed agree with the "fast approximate algorithm" which is then introduced.
- `01_Fast_algorithm_GPU.ipynb` and `02_SupervisedPhase.ipynb` reproduce the main results from the paper. Namely, the first notebook trains a first layer of weights using the local, unsupervised learning rule of [1] on the MNIST dataset. Then, the second notebook loads these weights and trains a "top" supervised layer to perform classification. This is done by using Pytorch libraries such as Pytorch Lightning and Pytorch Ignite, which allow scaling the model on powerful hardware (Pytorch Lightning, for instance, is immediately compatible with clusters of GPUs).
- `03_CNN_Prototype.ipynb` shows a possible way to generalize the rule found in [1] to Convolutional Neural Networks. Learned weights are visualized and show promise, but more work is needed before this could be used for classification.

Finally, `packages.json` provides a list of all the packages (with their version) which were part of the environment where these simulations have been made.