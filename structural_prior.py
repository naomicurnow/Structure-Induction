
import math
import numpy as np
from functools import lru_cache
from config import THETA, THETA_PROD
from graphs_and_forms import StructuralForm, ProductClusterGraph
from scipy.special import stirling2


def log_structural_prior(cluster_graph, form: StructuralForm, reg_theta: bool = False) -> float:
    n = cluster_graph.n_entities()
    form_name = form.name

    # find |S|
    if form_name == 'tree': # if tree, |S| is number of leaf nodes
        size_S = len(list(cluster_graph.leaf_ids))
    else: # else, |S| is number of latent nodes
        size_S = len(cluster_graph.latent_ids)

    if form_name == "grid":
        Z = Z_grid(n)
        if reg_theta: theta = THETA
        else: theta = THETA_PROD
    elif form_name == "cylinder":
        Z = Z_cylinder(n)
        if reg_theta: theta = THETA
        else: theta = THETA_PROD
    else:
        Z = Z_other(n, form_name)
        theta = THETA

    numerator = theta ** size_S
    return np.log(numerator) - np.log(Z)  # return log P(S|F)


@lru_cache(None)
def Z_other(n: int, form_name: str) -> float:
    total = 0.0
    for k in range(1, n + 1):
        total += stirling2(n, k) * C_F_k(form_name, k) * (THETA ** k)
    return total


def C_F_k(form_name: str, k: int) -> float:
    if form_name == "partition":
        return 1.0
    elif form_name == "directed_chain":
        return math.factorial(k)
    elif form_name == "chain":
        return 1.0 if k == 1 else math.factorial(k) / 2.0
    elif form_name == "order":
        return math.factorial(k)
    elif form_name == "connected":
        return 1.0
    elif form_name == "directed_ring":
        return math.factorial(k - 1)
    elif form_name == "ring":
        if k <= 2:
            return 1.0
        return math.factorial(k - 1) / 2.0
    elif form_name == "directed_hierarchy":
        return k ** (k - 1)
    elif form_name == "hierarchy":
        return k ** (k - 2)
    elif form_name == "tree":
        # (2k - 5)!! = product of odd numbers up to (2k-5); define for k>=2
        if k < 2:
            return 1.0
        val = 1
        for t in range(1, k - 1):
            val *= (2 * t - 1)
        return val
    else:
        raise ValueError(f"Unknown form {form_name}")
    

def L(n: int, i: int) -> float:
    if i == 1:
        return 1.0
    return (math.factorial(i) / 2.0) * stirling2(n, i)

def G(n: int, i: int, j: int) -> float:
    if i != j:
        return L(n, i) * L(n, j) * (THETA ** (i * j))
    else:
        return ((L(n, i) ** 2 + L(n, i)) / 2.0) * (THETA ** (i * i))

def R(n: int, i: int) -> float:
    return L(n, i) / i

def Y(n: int, i: int, j: int) -> float:
    return (R(n, i) * L(n, j)) * (THETA ** (i * j))


@lru_cache(None)
def Z_grid(n: int) -> float:
    total = 0.0
    for i in range(1, n + 1):
        for j in range(i, n + 1):  # i <= j
            total += G(n, i, j)
    return total

@lru_cache(None)
def Z_cylinder(n: int) -> float:
    total = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            total += Y(n, i, j)
    return total