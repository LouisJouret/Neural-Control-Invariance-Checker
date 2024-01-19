# Neural Control Invariance Checker

The **Neural Control Invariance Checker** is a Python package designed to assess the invariance properties of a controlled linear system when governed by a neural network controller. This package provides tools to analyze and evaluate the behavior of the system-controller combination in various scenarios. The theory behind the code has been established in a paper called "Safety Verification of Neural-Network-Based Controllers: A Set Invariance Approach". It can be found here: https://ieeexplore.ieee.org/document/10354438. You can also find the pre-print version on arXiv: https://arxiv.org/abs/2312.11352.

## Features

- Divide the state space into convex polytopes representing the regions in which the controller behaves linearly.
- Evaluate the invariance of a set (the set has to be defined as S\O where S and O are both convex polytopic sets) for a linear system controlled with a neural network controller.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LouisJouret/Neural-Control-Invariance-Checker.git
   ```

2. Navigate to the package directory:

    `cd neural-control-invariance-checker`

3. Install the dependencies:
    
    `pip install -r requirements.txt`

4. Install the package:

    `python setup.py install`

## Usage

Three linear dynamical systems are provided on which you can test your neural network controller. The neural networks can be trained in the tests/ directory by either approximating classical controllers like MPC or by reinforcement learning like DDPG. Finally the linear regions and invariance check can by used as follow:
```
import neural_control_invariance_checker as nci

linear_regions = nci.linear_regions(nn_controller, S, O)
linear_regions, good_vertices, bad_vertices = nci.check_controller_safety(S, O, nn_controller, A_sys, B_sys)
```

## Examples
Visit the tests/test.py file for an inuitive example of how the package should be used.
