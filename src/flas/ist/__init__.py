"""Inverse Semantic Transport (IST): explain response differences as sparse
compositions of FLAS concept transports, plus graded strength<->flow-time
training.

Modules:
    activations       layer-l activation extraction + point-cloud distances
    inverse           TransportMixture + sparse / greedy inverse solvers
    mixture_generate  multi-concept steered generation (causal validation)
    train_graded      graded strength<->flow-time supervised training
"""

from flas.ist.activations import (
    extract_layer_activations, activation_distance, masked_mean)
from flas.ist.inverse import (
    TransportMixture, generic_direction, solve_sparse, solve_greedy, steered_nll)
from flas.ist.mixture_generate import MixtureFlasGenerator, load_mixture_generator
