
import numpy as np
from scipy.optimize import minimize, check_grad
from graphs_and_forms import EntityGraph
from typing import Tuple, Dict, Any, List
from config import *
import warnings
import logging
import math

logger = logging.getLogger(__name__)

MIN_L = 1e-12      # min edge length (prevents 1/0 in weights)
MIN_SIG = 1e-9     # min sigma (keeps prior finite/PD)

LOG_L_MIN  = float(np.log(MIN_L))
LOG_SIG_MIN = float(np.log(MIN_SIG))


def node_sort_key(x):
    """
    Total order for different node id types:
      - ints -> bucket 0, key (0, int(x))
      - tuples (e.g., (rgid, cgid)) -> bucket 1, key (1, len, *tuple)
    """
    if isinstance(x, int):
        return (0, int(x))
    if isinstance(x, tuple):
        return (1, len(x), *x)
    

def edge_sort_key(e):
    u, v = e
    ku, kv = node_sort_key(u), node_sort_key(v)
    return (ku, kv) if ku <= kv else (kv, ku)


def build_matrix_layout(entity_graph) -> Tuple[List[int], Dict[int,int]]:
    """
    Return (node_order_gids, gid_to_idx) where
      node_order_gids = entity_gids + cluster_cids
      gid_to_idx maps a global node id to its row/col index in matrices.
    """
    node_order_gids = list(entity_graph.entity_gids) + list(entity_graph.cluster_cids)
    gid_to_idx = {gid: i for i, gid in enumerate(node_order_gids)}
    return node_order_gids, gid_to_idx


def compute_weight_matrix(edge_lengths: np.ndarray,
                          ordered_edges: List[Tuple[int,int]],
                          gid_to_idx: Dict[int,int],
                          n_nodes: int) -> np.ndarray:
    """
    Compute weight matrix W where for each edge (i,j): w_ij = 1 / e_ij.
    """
    if len(edge_lengths) != len(ordered_edges):
        raise ValueError(f"edge_lengths length {len(edge_lengths)} "
                         f"does not match ordered_edges length {len(ordered_edges)}")
    
    W = np.zeros((n_nodes, n_nodes), dtype=float)
    for k, (u_gid, v_gid) in enumerate(ordered_edges):
        length = float(edge_lengths[k])
        w = 1.0 / length
        u = gid_to_idx[u_gid]
        v = gid_to_idx[v_gid]
        W[u, v] = w
        W[v, u] = w

    return W


def compute_graph_laplacian(W: np.ndarray) -> np.ndarray:
    """Compute Laplacian Δ = E − W where E is diagonal with row sums of W."""
    row_sums = np.sum(W, axis=1)
    E = np.diag(row_sums)
    return E - W


def add_feature_prior(Delta: np.ndarray, sigma: float, n_entities: int) -> np.ndarray:
    """
    Augment Laplacian with prior: add V where first n_entities diagonal entries get 1/sigma^2.
    """
    N = Delta.shape[0]
    V = np.zeros((N, N))
    V[:n_entities, :n_entities] = np.eye(n_entities) * (1.0 / (sigma ** 2))
    return Delta + V


def neg_log_posterior(params: np.ndarray,
                       D_or_S: np.ndarray,
                       entity_graph: EntityGraph,
                       tied: str,
                       is_similarity: bool,
                       return_grad: bool = False) -> float:
    """
    If return_grad == True, returns neg_logpost (−[ log P(D|params) + log P(params) ]), grad (∂/∂params (neg_logpost)). 
    Else, just returns neg_logpost.

    params: if tied = 'both' -> single log_l
            if tied = 'int' -> single log_l for internal edges, untied external
            if tied = 'ext' -> single log_l for external edges, untied internal
            if tied = 'none' -> log_l_1, ..., log_l_E different for each edge

    If is_similarity == False:
        D_or_S is a feature matrix D with shape (n_entities, n_features).
        We use S = (D D^T) / f and f_eff = n_features.

    If is_similarity == True:
        D_or_S is a precomputed similarity/scatter matrix S with shape (n_entities, n_entities),
        assumed to be (approximately) symmetric PSD and already averaged across features.
        We use S as-is (symmetrized) and set f_eff = EFF_M.
    """
    n_ent = entity_graph.n_entities

    # build S and set effective feature count f_eff
    if is_similarity:
        S = np.asarray(D_or_S, dtype=float)
        if S.shape != (n_ent, n_ent):
            raise ValueError(f"Similarity matrix must be {n_ent}x{n_ent}, got {S.shape}.")
        f_eff = EFF_M
    else:
        D = np.asarray(D_or_S, dtype=float)
        if D.shape[0] != n_ent:
            raise ValueError(f"Feature matrix has {D.shape[0]} rows but entity_graph has {n_ent} entities.")
        _, f = D.shape
        f_eff = float(f)
        S = (D @ D.T) / f_eff  # empirical scatter

    # ensure consistent matrix layout and edge ordering
    node_order_gids, gid_to_idx = build_matrix_layout(entity_graph)
    n_nodes = len(node_order_gids)
    idx_e = list(range(n_ent))
    idx_c = list(range(n_ent, n_nodes))

    ordered_edges = sorted(entity_graph.edges, key=edge_sort_key) # list of (gid,gid)
    ordered_ext_edges = sorted(entity_graph.edges_external, key=edge_sort_key)
    ordered_int_edges = sorted(entity_graph.edges_internal, key=edge_sort_key)

    # unpack parameters
    if tied == 'both':
        log_l, log_sigma = params
        l = np.exp(log_l)
        # same log-length for every edge
        log_l_dict = {edge: log_l for edge in ordered_edges}
    elif tied == 'int':
        *log_ls_ext, log_l_int, log_sigma = params.tolist()
        if len(log_ls_ext) != len(ordered_ext_edges):
            raise ValueError("Expected %d external lengths, got %d" % (len(ordered_ext_edges), len(log_ls_ext)))
        log_l_dict = {}
        for i, e in enumerate(ordered_ext_edges):
            log_l_dict[e] = log_ls_ext[i]
        for e in entity_graph.int_edges:
            log_l_dict[e] = log_l_int
        l_int = np.exp(log_l_int)
    elif tied == 'ext':
        *log_ls_int, log_l_ext, log_sigma = params.tolist()
        if len(log_ls_int) != len(ordered_int_edges):
            raise ValueError("Expected %d internal lengths, got %d" % (len(ordered_int_edges), len(log_ls_int)))
        log_l_dict = {}
        for i, e in enumerate(ordered_int_edges):
            log_l_dict[e] = log_ls_int[i]
        for e in entity_graph.ext_edges:
            log_l_dict[e] = log_l_ext
        l_ext = np.exp(log_l_ext)
    elif tied == 'none':
        *log_ls, log_sigma = params.tolist()
        if len(log_ls) != len(ordered_edges):
            raise ValueError("Expected %d lengths, got %d" % (len(ordered_edges), len(log_ls)))
        log_l_dict = {edge: log_ls[i] for i, edge in enumerate(ordered_edges)}
    else:
        raise ValueError(f"Unknown tying mode: {tied}")

    sigma = max(np.exp(log_sigma), 1e-9)

    # edge lengths in the exact `ordered_edges` order
    edge_lengths = np.array([np.exp(log_l_dict[e]) for e in ordered_edges], dtype=float)

    # precision matrix = Laplacian + prior
    W = compute_weight_matrix(edge_lengths, ordered_edges, gid_to_idx, n_nodes)
    L = compute_graph_laplacian(W)
    P_full = add_feature_prior(L, sigma, n_ent)

    P_ee = P_full[np.ix_(idx_e, idx_e)]
    P_ec = P_full[np.ix_(idx_e, idx_c)]
    P_ce = P_full[np.ix_(idx_c, idx_e)]
    P_cc = P_full[np.ix_(idx_c, idx_c)]
    
    try:
        P_cc_inv = np.linalg.inv(P_cc)
        prec_small = P_ee - P_ec @ P_cc_inv @ P_ce
        sign, logdet_small = np.linalg.slogdet(prec_small)
        if sign <= 0: return np.inf, np.inf * np.ones_like(params)
    except np.linalg.LinAlgError:
        return np.inf, np.inf * np.ones_like(params)

    neg_loglik = 0.5 * f_eff * (-logdet_small + np.trace(prec_small @ S))

    # priors
    prior_sigma = BETA * sigma
    if PRIOR == 'exp':
        prior_l = BETA * sum(np.exp(log_l) for log_l in log_l_dict.values())
    elif PRIOR == 'gamma':
        prior_l = 0.0
        for log_l in log_l_dict.values():
            l = np.exp(log_l)
            prior_l += BETA * l - (ALPHA - 1) * log_l
    else:
        raise ValueError(f"Unknown prior '{PRIOR}'")

    neg_logpost = neg_loglik + (prior_l + prior_sigma)

    if not return_grad:
        return neg_logpost

    # ***** gradient calculation *****

    grad = np.zeros_like(params)

    # common terms for trace gradient
    M = P_cc_inv @ P_ce
    N = S @ P_ec @ P_cc_inv

    try:
        if tied == 'both':
            # dP/dlogl = (-1/l) * L0  where L0 is Laplacian with all weights = 1
            W0 = compute_weight_matrix(np.ones(len(ordered_edges), dtype=float),
                                       ordered_edges, gid_to_idx, n_nodes)
            L0_full = compute_graph_laplacian(W0)
            dP_full_dlogl = (-1.0 / l) * L0_full
            dP_cc_dlogl = dP_full_dlogl[np.ix_(idx_c, idx_c)]

            # log-det grad term
            logdet_grad_l = -0.5 * f_eff * (
                np.trace(np.linalg.solve(P_full, dP_full_dlogl)) - 
                np.trace(np.linalg.solve(P_cc, dP_cc_dlogl))
                )

            # trace grad term
            dP_ee_dl = dP_full_dlogl[np.ix_(idx_e, idx_e)]
            dP_ec_dl = dP_full_dlogl[np.ix_(idx_e, idx_c)]
            dP_ce_dl = dP_full_dlogl[np.ix_(idx_c, idx_e)]
            dP_cc_dl = dP_full_dlogl[np.ix_(idx_c, idx_c)]

            T1 = dP_ee_dl
            T2 = dP_ec_dl @ P_cc_inv @ P_ce
            T3 = P_ec @ P_cc_inv @ dP_cc_dl @ P_cc_inv @ P_ce
            T4 = P_ec @ P_cc_inv @ dP_ce_dl
            dP_small_dlogl = T1 - T2 + T3 - T4

            trace_grad_l = 0.5 * f_eff * np.trace(S @ dP_small_dlogl)

            if PRIOR == 'exp':
                grad[0] = logdet_grad_l + trace_grad_l + (BETA * l * len(edge_lengths))
            elif PRIOR == 'gamma':
                grad[0] = logdet_grad_l + trace_grad_l + (BETA * l - (ALPHA-1)) * len(edge_lengths)
        
        elif tied == 'int':
            # external edges (each has its own length)
            for j, e in enumerate(ordered_ext_edges):
                u_gid, v_gid = e
                u, v = gid_to_idx[u_gid], gid_to_idx[v_gid]
                l_j = np.exp(log_ls_ext[j])

                E_j = np.zeros((n_nodes, n_nodes))
                E_j[u, u] = E_j[v, v] = 1
                E_j[u, v] = E_j[v, u] = -1

                dP_full_dj = (-1.0 / l_j) * E_j
                dP_cc_dj = dP_full_dj[np.ix_(idx_c, idx_c)]

                logdet_j = -0.5 * f_eff * (
                    np.trace(np.linalg.solve(P_full, dP_full_dj)) -
                    np.trace(P_cc_inv @ dP_cc_dj)
                )

                dP_ee = dP_full_dj[np.ix_(idx_e, idx_e)]
                dP_ec = dP_full_dj[np.ix_(idx_e, idx_c)]
                dP_ce = dP_full_dj[np.ix_(idx_c, idx_e)]
                
                trace_j = 0.5 * f_eff * (
                                    np.trace(S @ dP_ee)
                                    - np.trace(N @ dP_ce)
                                    - np.trace(S @ dP_ec @ M)
                                    + np.trace(N @ dP_cc_dj @ M)
                                )
                
                if PRIOR == 'exp':
                    grad[j] = logdet_j + trace_j + (BETA * l_j)
                elif PRIOR == 'gamma':
                    grad[j] = logdet_j + trace_j + (BETA * l_j - (ALPHA - 1))

            # shared internal length
            E_int = np.zeros((n_nodes, n_nodes))
            for u_gid, v_gid in ordered_int_edges:
                u, v = gid_to_idx[u_gid], gid_to_idx[v_gid]
                E_int[u, u] += 1.0
                E_int[v, v] += 1.0
                E_int[u, v] -= 1.0
                E_int[v, u] -= 1.0

            dP_full_int = (-1.0 / l_int) * E_int
            dP_cc_int = dP_full_int[np.ix_(idx_c, idx_c)]

            logdet_int = -0.5 * f_eff * (
                np.trace(np.linalg.solve(P_full, dP_full_int)) -
                np.trace(P_cc_inv @ dP_cc_int)
            )
            
            dP_ee_int = dP_full_int[np.ix_(idx_e, idx_e)]
            dP_ec_int = dP_full_int[np.ix_(idx_e, idx_c)]
            dP_ce_int = dP_full_int[np.ix_(idx_c, idx_e)]

            trace_int = 0.5 * f_eff * (
                np.trace(S @ dP_ee_int)
                - np.trace(N @ dP_ce_int)
                - np.trace(S @ dP_ec_int @ M)
                + np.trace(N @ dP_cc_int @ M)
            )
            
            idx_shared = len(ordered_ext_edges)
            if PRIOR == 'exp':
                grad[idx_shared] = logdet_int + trace_int + (BETA * l_int * len(ordered_int_edges))
            elif PRIOR == 'gamma':
                grad[idx_shared] = logdet_int + trace_int + (BETA * l_int - (ALPHA - 1)) * len(ordered_int_edges)
        
        elif tied == 'ext':
            # internal edges (each has its own length)
            for j, e in enumerate(ordered_int_edges):
                u_gid, v_gid = e
                u, v = e
                l_j = np.exp(log_ls_int[j])

                E_j = np.zeros((n_nodes, n_nodes))
                E_j[u, u] = E_j[v, v] = 1
                E_j[u, v] = E_j[v, u] = -1

                dP_full_dj = (-1.0 / l_j) * E_j
                dP_cc_dj = dP_full_dj[np.ix_(idx_c, idx_c)]

                logdet_j = -0.5 * f_eff * (
                    np.trace(np.linalg.solve(P_full, dP_full_dj)) -
                    np.trace(P_cc_inv @ dP_cc_dj)
                )

                dP_ee = dP_full_dj[np.ix_(idx_e, idx_e)]
                dP_ec = dP_full_dj[np.ix_(idx_e, idx_c)]
                dP_ce = dP_full_dj[np.ix_(idx_c, idx_e)]

                trace_j = 0.5 * f_eff * (
                    np.trace(S @ dP_ee)
                    - np.trace(N @ dP_ce)
                    - np.trace(S @ dP_ec @ M)
                    + np.trace(N @ dP_cc_dj @ M)
                )
                
                if PRIOR == 'exp':
                    grad[j] = logdet_j + trace_j + (BETA * l_j)
                elif PRIOR == 'gamma':
                    grad[j] = logdet_j + trace_j + (BETA * l_j - (ALPHA - 1))

            # shared external length
            E_ext = np.zeros((n_nodes, n_nodes))
            for u_gid, v_gid in ordered_ext_edges:
                u, v = gid_to_idx[u_gid], gid_to_idx[v_gid]
                E_ext[u, u] += 1.0
                E_ext[v, v] += 1.0
                E_ext[u, v] -= 1.0
                E_ext[v, u] -= 1.0

            dP_full_ext = (-1.0 / l_ext) * E_ext
            dP_cc_ext = dP_full_ext[np.ix_(idx_c, idx_c)]

            logdet_ext = -0.5 * f_eff * (
                np.trace(np.linalg.solve(P_full, dP_full_ext)) -
                np.trace(P_cc_inv @ dP_cc_ext)
            )

            dP_ee_ext = dP_full_ext[np.ix_(idx_e, idx_e)]
            dP_ec_ext = dP_full_ext[np.ix_(idx_e, idx_c)]
            dP_ce_ext = dP_full_ext[np.ix_(idx_c, idx_e)]

            trace_ext = 0.5 * f_eff * (
                np.trace(S @ dP_ee_ext)
                - np.trace(N @ dP_ce_ext)
                - np.trace(S @ dP_ec_ext @ M)
                + np.trace(N @ dP_cc_ext @ M)
            )

            idx_shared = len(ordered_int_edges)
            if PRIOR == 'exp':
                grad[idx_shared] = logdet_ext + trace_ext + (BETA * l_ext * len(ordered_ext_edges))
            elif PRIOR == 'gamma':
                grad[idx_shared] = logdet_ext + trace_ext + (BETA * l_ext - (ALPHA - 1)) * len(ordered_ext_edges)

        elif tied == 'none':
            # each edge has its own parameter
            for j, (u_gid, v_gid) in enumerate(ordered_edges):
                u, v = gid_to_idx[u_gid], gid_to_idx[v_gid]
                l_j = edge_lengths[j]

                E_j = np.zeros((n_nodes, n_nodes))
                E_j[u, u] = E_j[v, v] = 1.0
                E_j[u, v] = E_j[v, u] = -1.0

                dP_full_dlogl_j = (-1.0 / l_j) * E_j
                dP_cc_dlogl_j = dP_full_dlogl_j[np.ix_(idx_c, idx_c)]

                logdet_grad_j = -0.5 * f_eff * (
                    np.trace(np.linalg.solve(P_full, dP_full_dlogl_j)) -
                    np.trace(P_cc_inv @ dP_cc_dlogl_j)
                )

                dP_ee_dlogl_j = dP_full_dlogl_j[np.ix_(idx_e, idx_e)]
                dP_ec_dlogl_j = dP_full_dlogl_j[np.ix_(idx_e, idx_c)]
                dP_ce_dlogl_j = dP_full_dlogl_j[np.ix_(idx_c, idx_e)]

                trace_grad_j = 0.5 * f_eff * np.trace(
                    S @ dP_ee_dlogl_j
                    - (N @ dP_ce_dlogl_j)
                    - (S @ dP_ec_dlogl_j @ M)
                    + (N @ dP_cc_dlogl_j @ M)
                )

                if PRIOR == 'exp':
                    grad[j] = logdet_grad_j + trace_grad_j + (BETA * l_j)
                elif PRIOR == 'gamma':
                    grad[j] = logdet_grad_j + trace_grad_j + (BETA * l_j - (ALPHA - 1))

        # gradient with respect to sigma (same for tied and untied)
        dP_dlogsigma_full = np.zeros_like(P_full)
        np.fill_diagonal(dP_dlogsigma_full[:n_ent, :n_ent], -2.0 / (sigma**2))

        logdet_grad_s = -0.5 * f_eff * np.trace(np.linalg.solve(P_full, dP_dlogsigma_full))
        dP_ee_dlogsigma = dP_dlogsigma_full[np.ix_(idx_e, idx_e)]
        trace_grad_s = 0.5 * f_eff * np.trace(S @ dP_ee_dlogsigma)

        grad[-1] = logdet_grad_s + trace_grad_s + (BETA * sigma)

    except np.linalg.LinAlgError:
        return np.inf, np.zeros_like(params) * np.inf

    return neg_logpost, grad


def compute_hessian_from_grad(func, x, epsilon=1e-5, *args, **kwargs):
    """
    finite-difference Hessian via central differences on the gradient.
    
    H_ij = ∂/∂x_j [ ∂f/∂x_i ] ≈ (g_i(x+ε e_j) - g_i(x-ε e_j)) / (2ε)
    where g = grad f.
    """
    # evaluate once to get gradient size
    _, grad0 = func(x, *args, **kwargs)
    n = x.size
    H = np.zeros((n, n), float)
    for j in range(n):
        dx = np.zeros_like(x); dx[j] = epsilon
        _, grad_plus = func(x + dx, *args, **kwargs)
        _, grad_minus = func(x - dx, *args, **kwargs)
        H[:, j] = (grad_plus - grad_minus) / (2 * epsilon)
    return H


def laplace_correction(neg_log_posterior,
                       x_mode: np.ndarray,
                       epsilon: float = np.sqrt(np.finfo(float).eps)):
    """
    compute the Laplace-approximation correction term
      ½ d log(2π) + ½ log det Cov
    where Cov = [-H]^{-1}, H = ∇² log posterior at x_mode.
    
    """
    H = compute_hessian_from_grad(neg_log_posterior, x_mode, epsilon)
    d = H.shape[0]
    try:
        # use Cholesky decomposition to find the log-determinant.
        # this fails if H is not positive definite.
        L = np.linalg.cholesky(H)
        log_det_H = 2 * np.sum(np.log(np.diag(L)))
        correction = 0.5 * (d * np.log(2 * np.pi) - log_det_H)
        return correction
    except np.linalg.LinAlgError:
        warnings.warn(
            "Hessian is not positive definite at the found mode. "
            "The Laplace approximation is invalid. The optimiser may not have converged to a true maximum. "
            "Returning -inf for the log marginal likelihood."
        )
        return -np.inf


def log_marginal_likelihood(data: np.ndarray,
                            entity_graph,
                            form,
                            tied_only: bool,
                            is_similarity: bool) -> Tuple[float, Dict[str, Any]]:
    """
    If is_similarity=False: data is D (data matrix: n_entities x n_features).
    If is_similarity=True : data is S (similarity matrix: n_entities x n_entities).
    """
    def obj_tied(x):
        return neg_log_posterior(x, data, entity_graph, tied='both', is_similarity=is_similarity, return_grad=True)

    def obj_untied(x):
        # if form.name == 'tree':
        #     return neg_log_posterior(x, data, entity_graph, tied='ext', is_similarity=is_similarity, return_grad=True)
        # else:
        return neg_log_posterior(x, data, entity_graph, tied='none', is_similarity=is_similarity, return_grad=True)

    # tied phase: optimise single length + sigma
    x0_tied = np.log([1.0, 1.0])
    bounds_tied = [(LOG_L_MIN, None), (LOG_SIG_MIN, None)]
    res_tied = minimize(
        obj_tied,
        x0_tied,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_tied
    )

    if not res_tied.success:
        warnings.warn(f"Tied optimisation did not converge: {res_tied.message}")
    log_l_tied, log_sigma_tied = res_tied.x

    # test_full_gradient(data, entity_graph)
    # test_gradient_parts(data, entity_graph)

    if not tied_only:
        # untied phase: optimise one log_l per edge + log_sigma

        # if form.name == 'tree':
        #     init_untied = np.concatenate([np.full(len(entity_graph.int_edges)+1, log_l_tied), [log_sigma_tied]])
        # else:
        ordered_edges = sorted(entity_graph.edges, key=edge_sort_key)

        init_untied = np.concatenate([np.full(len(ordered_edges), log_l_tied), [log_sigma_tied]])
        bounds_untied = [(LOG_L_MIN, None)] * len(ordered_edges) + [(LOG_SIG_MIN, None)]
        res_untied = minimize(
            obj_untied,
            x0=init_untied,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds_untied
        )

        if not res_untied.success:
            warnings.warn(f"Untied optimisation did not converge: {res_untied.message}")

        x_star = res_untied.x
        neglogpost_at_mode, _ = obj_untied(x_star)
        log_posterior_mode = -neglogpost_at_mode
        laplace_corr = laplace_correction(obj_untied, x_star)
        log_marginal = log_posterior_mode + laplace_corr

        # if form.name == 'tree':
        #     log_l_dict = {edge: res_untied.x[:-2][i] for i, edge in enumerate(entity_graph.int_edges)}
        #     log_l_dict['ext'] = res_untied.x[-2]
        # else:
        log_l_dict = { ordered_edges[i]: res_untied.x[i] for i in range(len(ordered_edges)) }

        params = {
            "tied": {"log_l": log_l_tied, "log_sigma": log_sigma_tied},
            "untied": {"log_ls": log_l_dict, "log_sigma": res_untied.x[-1]},
        }

        if not math.isinf(log_marginal):
            return log_marginal, params
    
    # either tied_only OR untied optimisation returned -inf
    
    params = {
        "tied": {"log_l": log_l_tied, "log_sigma": log_sigma_tied},
    }
    
    x_star = res_tied.x
    neglogpost_at_mode, _ = obj_tied(x_star)
    log_posterior_mode = -neglogpost_at_mode
    laplace_corr = laplace_correction(obj_tied, x_star)
    log_marginal = log_posterior_mode + laplace_corr

    return log_marginal, params



## testing

def test_full_gradient(D: np.ndarray, entity_graph: EntityGraph):
    """
    Tests the gradient of neg_log_posterior for both tied and untied modes via finite differences.
    Prints out the check_grad errors.
    """
    # Data and graph
    eg = entity_graph
    ordered_edges = sorted(list(eg.edges), key=edge_sort_key)
    n_edges = len(ordered_edges)

    # Prepare initial points in log-space
    x0_tied = np.log([1.0, 1.0])  # [log_l, log_sigma]
    x0_untied = np.zeros(n_edges + 1)  # log_l_i = 0, log_sigma = 0 => all params = log(1)

    def obj_tied_val(x):
        val, _ = neg_log_posterior(x, D, eg, tied='both', is_similarity=False, return_grad=True)
        return val
    def obj_tied_grad(x):
        _, grad = neg_log_posterior(x, D, eg, tied='both', is_similarity=False, return_grad=True)
        return grad
    
    def obj_tied_int_val(x):
        val, _ = neg_log_posterior(x, D, eg, tied='int', is_similarity=False, return_grad=True)
        return val
    def obj_tied_int_grad(x):
        _, grad = neg_log_posterior(x, D, eg, tied='int', is_similarity=False, return_grad=True)
        return grad
    
    def obj_tied_ext_val(x):
        val, _ = neg_log_posterior(x, D, eg, tied='ext', is_similarity=False, return_grad=True)
        return val
    def obj_tied_ext_grad(x):
        _, grad = neg_log_posterior(x, D, eg, tied='ext', is_similarity=False, return_grad=True)
        return grad

    def obj_untied_val(x):
        val, _ = neg_log_posterior(x, D, eg, tied='none', is_similarity=False, return_grad=True)
        return val
    def obj_untied_grad(x):
        _, grad = neg_log_posterior(x, D, eg, tied='none', is_similarity=False, return_grad=True)
        return grad

    # Check gradients
    err_tied = check_grad(obj_tied_val, obj_tied_grad, x0_tied)
    err_untied = check_grad(obj_untied_val, obj_untied_grad, x0_untied)

    print(f"Full neg_log_posterior gradient error (TIED):   {err_tied:.3e}")
    print(f"Full neg_log_posterior gradient error (UNTIED): {err_untied:.3e}")

# --- Function to check ONLY the Log-Determinant term ---
def check_logdet_component(params, D, entity_graph):
    f = D.shape[1]
    n = D.shape[0]
    log_l, log_sigma = params
    l, sigma = np.exp(log_l), np.exp(log_sigma)
    
    # --- Setup Precision Matrix P ---
    W = compute_weight_matrix(entity_graph, np.ones(len(entity_graph.edges)) * l)
    L = compute_graph_laplacian(W)

    precision = add_feature_prior(L, sigma, entity_graph.n_entities)
    idx_e = range(entity_graph.n_entities) # entity nodes: 0…n-1
    idx_c = [i for i in range(precision.shape[0]) if i not in idx_e]
    P_ee = precision[np.ix_(idx_e, idx_e)]
    P_ec = precision[np.ix_(idx_e, idx_c)]
    P_cc = precision[np.ix_(idx_c, idx_c)]
    prec_small = P_ee - P_ec @ np.linalg.inv(P_cc) @ P_ec.T

    # negative log-likelihood
    S = (D @ D.T) / float(f) # empirical scatter
    sign, logdet = np.linalg.slogdet(prec_small)
    if sign <= 0:
        # these parameters are in an invalid region
        return np.inf, np.zeros_like(params) * np.inf
    
    value = -0.5 * f * logdet
    
    # --- Gradient ---
    grad = np.zeros_like(params)
    
    # Grad wrt log_l
    W0 = compute_weight_matrix(entity_graph, np.ones(len(entity_graph.edges)))
    L0_full = compute_graph_laplacian(W0)
    dP_full_dlogl = (-1.0 / l) * L0_full
    dP_cc_dlogl = dP_full_dlogl[np.ix_(idx_c, idx_c)]
    
    term_full_l = np.trace(np.linalg.solve(precision, dP_full_dlogl))
    term_cc_l = np.trace(np.linalg.solve(P_cc, dP_cc_dlogl))
    grad[0] = -0.5 * f * (term_full_l - term_cc_l)
    
    # Grad wrt log_sigma
    dP_full_dlogsigma = np.zeros_like(precision)
    np.fill_diagonal(dP_full_dlogsigma[:entity_graph.n_entities, :entity_graph.n_entities], -2.0 / (sigma**2))
    # Note: dP_cc_dlogsigma is a zero matrix since the sigma prior only affects P_ee
    
    term_full_s = np.trace(np.linalg.solve(precision, dP_full_dlogsigma))
    grad[1] = -0.5 * f * term_full_s # term_cc_s is zero

    return value, grad

# --- Function to check ONLY the Trace term ---
def check_trace_component(params, D, entity_graph):
    f = D.shape[1]
    n = D.shape[0]
    log_l, log_sigma = params
    l, sigma = np.exp(log_l), np.exp(log_sigma)
    S = (D @ D.T) / float(f)
    
    # --- Setup Precision Matrix P ---
    W = compute_weight_matrix(entity_graph, np.ones(len(entity_graph.edges)) * l)
    L = compute_graph_laplacian(W)

    precision = add_feature_prior(L, sigma, entity_graph.n_entities)
    idx_e = range(entity_graph.n_entities) # entity nodes: 0…n-1
    idx_c = [i for i in range(precision.shape[0]) if i not in idx_e]
    P_ee = precision[np.ix_(idx_e, idx_e)]
    P_ec = precision[np.ix_(idx_e, idx_c)]
    P_ce = precision[np.ix_(idx_c, idx_e)]
    P_cc = precision[np.ix_(idx_c, idx_c)]
    P_cc_inv = np.linalg.inv(P_cc)
    prec_small = P_ee - P_ec @ P_cc_inv @ P_ce
    
    # --- Value ---
    value = 0.5 * f * np.trace(prec_small @ S)
    
    # --- Gradient ---
    grad = np.zeros_like(params)
    
    # Grad wrt log_l
    W0 = compute_weight_matrix(entity_graph, np.ones(len(entity_graph.edges)))
    L0_full = compute_graph_laplacian(W0)
    dP_full_dlogl = (-1.0 / l) * L0_full
    
    # Get sub-blocks of the derivative matrix
    dP_ee_dl = dP_full_dlogl[np.ix_(idx_e, idx_e)]
    dP_ec_dl = dP_full_dlogl[np.ix_(idx_e, idx_c)]
    dP_ce_dl = dP_full_dlogl[np.ix_(idx_c, idx_e)]
    dP_cc_dl = dP_full_dlogl[np.ix_(idx_c, idx_c)]
    
    # Assemble the derivative of the Schur complement using the formula
    T1 = dP_ee_dl
    T2 = dP_ec_dl @ P_cc_inv @ P_ce
    T3 = P_ec @ P_cc_inv @ dP_cc_dl @ P_cc_inv @ P_ce
    T4 = P_ec @ P_cc_inv @ dP_ce_dl
    dP_small_dlogl = T1 - T2 + T3 - T4
    
    grad[0] = 0.5 * f * np.trace(S @ dP_small_dlogl)
    
    # Grad wrt log_sigma
    dP_full_dlogsigma = np.zeros_like(precision)
    np.fill_diagonal(dP_full_dlogsigma[:entity_graph.n_entities, :entity_graph.n_entities], -2.0 / (sigma**2))
    # For sigma, only the P_ee block has a non-zero derivative
    dP_small_dlogsigma = dP_full_dlogsigma[np.ix_(idx_e, idx_e)]
    grad[1] = 0.5 * f * np.trace(S @ dP_small_dlogsigma)

    return value, grad

# --- Function to check ONLY the Prior term ---
def check_prior_component(params):
    log_l, log_sigma = params
    l, sigma = np.exp(log_l), np.exp(log_sigma)
    
    # --- Value ---
    # Gamma prior on l + Exponential prior on sigma
    value = BETA * (l + sigma)
    
    # --- Gradient ---
    grad = np.zeros_like(params)
    grad[0] = BETA * l
    grad[1] = BETA * sigma
    
    return value, grad


def test_gradient_parts(D, entity_graph):
    x0_tied = np.log([1.0, 1.0]) 

    # Wrap the functions for check_grad
    func_logdet = lambda x: check_logdet_component(x, D, entity_graph)[0]
    grad_logdet = lambda x: check_logdet_component(x, D, entity_graph)[1]

    func_trace = lambda x: check_trace_component(x, D, entity_graph)[0]
    grad_trace = lambda x: check_trace_component(x, D, entity_graph)[1]

    func_prior = lambda x: check_prior_component(x)[0]
    grad_prior = lambda x: check_prior_component(x)[1]

    # Run the checks
    error_logdet = check_grad(func_logdet, grad_logdet, x0_tied)
    error_trace = check_grad(func_trace, grad_trace, x0_tied)
    error_prior = check_grad(func_prior, grad_prior, x0_tied)

    print(f"Log-determinant gradient error: {error_logdet}")
    print(f"Trace term gradient error:      {error_trace}")
    print(f"Prior term gradient error:      {error_prior}")