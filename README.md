# Neural Control Invariance Checker

The **Neural Control Invariance Checker** is a Python package designed to assess the invariance properties of a controlled linear system when governed by a neural network controller. This package provides tools to analyze and evaluate the behavior of the system-controller combination in various scenarios.

## Features

- Divide the state space into convex polytopes representing the regions in which the controller behaves linearly.
- Evaluate the invariance of a set (the set has to be defined as S\O where S and O are both convex polytopic sets) for a linear system controlled with a neural network controller.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LouisJouret/neural-control-invariance-checker.git
   ```

2. Navigate to the package directory:

    'cd neural-control-invariance-checker'

3. Install the dependencies:
    
    'pip install -r requirements.txt'

4. Install the package:

    'python setup.py install'

## Usage
```
import neural_control_invariance_checker as nci

linear_regions = nci.linear_regions(nn_controller, S, O)
linear_regions, good_vertices, bad_vertices = nci.check_controller_safety(S, O, nn_controller, A_sys, B_sys)
```

## Examples
Visit the tests/ directory in this repository for examples and demonstrations of how to use the package to analyze different controlled linear systems with neural network controllers.
