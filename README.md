# Redy
A research framework for finding redundancies in neural networks.

See the paper for more information ([Pruning and Slicing Neural Networks using Formal Verification, 2021, Ori Lahav and Guy Katz](https://arxiv.org/abs/2105.13649)).

Uses Marabou and Maraboupy as backend. See https://github.com/NeuralNetworkVerification/Marabou. Note that many other verification engines can be used as an alternative backends.

This framework is for research purposes and not intended for production. Use with caution, and make sure to read everything below before usage.

## Architecture
The general usage flow of this framework:
 1. Load a network into Redy representation
    - There are two representations ("Views")
        - `ViewIO` - Generic model with nodes, equations, inputs and outputs
        - `ViewNetwork` - More strict model with layers and without equations
 2. Modify the network and create redundancy queries (using `features.redundancy`)
 3. Export the query into one of the following -
    - `evaluate` - used for simulations, allows for modified network evaluation
    - `marabou` - for running queries on Marabou

## Usage
Read paper for terminology, and see examples of usage:
 - `example_basics.py`
    - Modify networks
    - Evaluate networks
    - Query neurons for redundancy (phase-redundancy, forward-redundancy and result-preserving redundancy) using Marabou
    - Restrict networks to sub-domains
 - `example_relax.py`
    - Example of creating a relax-redundant neuron
    - Functions for `l_m` computation (see paper)
 - `example_milp.py`
    - Instructions for running Gurobi Marabou MILP implementation
    - Code for parsing the output and extracting neurons bounds

## Marabou
 - We used commit `a771a89ba56991b62dd4644386a6460339a60243` of Marabou.
 - Before running queries on networks created using this framework, apply `redy/marabou_patches/nlr_bug_fix.patch`.
 - Make sure to enable Gurobi properly. See [Marabou](https://github.com/NeuralNetworkVerification/Marabou) for more information.
 - For MILP bounds calculation another patch should be applied. See `example_milp.py`.

