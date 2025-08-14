
from graphs_and_forms import ClusterGraph, StructuralForm, ProductClusterGraph, EntityGraph
from structural_prior import log_structural_prior
from likelihoods import log_marginal_likelihood, neg_log_posterior
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def cluster_graph_to_entity_graph(cluster_graph: ClusterGraph) -> EntityGraph:
    n_entities = cluster_graph.n_entities
    eg = EntityGraph(n_entities)
    latent_ids = sorted(cluster_graph.latent_nodes)
    
    # local (current) -> local-index (compact) for building matrices
    cluster_to_local = {lid: n_entities + idx for idx, lid in enumerate(latent_ids)}

    ent_glob = cluster_graph.latent_to_global_id["entities"]
    clu_glob = cluster_graph.latent_to_global_id["clusters"]

    gid_of_local = [ent_glob[i] for i in range(n_entities)]
    for lid in latent_ids:
        gid_of_local.append(clu_glob[lid])

    # connect entities to their latent
    for i in range(n_entities):
        cid = cluster_graph.entity_assignments[i]
        if cid != -1:
            eg.add_edge(i, cluster_to_local[cid], type='external')

    # latent-latent edges per topology (locals)
    for a, neighbours in cluster_graph.latent_adjacency.items():
        for b in neighbours:
            if a < b:
                eg.add_edge(cluster_to_local[a], cluster_to_local[b], type='internal')

    eg.finalise(n_latent=len(latent_ids), gid_of_local=gid_of_local)
    return eg


def get_params(cluster_graph, entity_graph, form, data, is_similarity):
    # grab tied init params from metadata if present, else compute

    # parent_meta = current.metadata
    # parent_tied = None
    # if parent_meta and "opt_params" in parent_meta:
    #     parent_tied = parent_meta["opt_params"].get("tied")  # expects dict with "log_l", "log_sigma"
    # else:
    #     _, params = find_score(current, form, data, is_similarity, speed='slow')
    #     parent_tied = params.get("tied")  # expects dict with "log_l", "log_sigma"
    # log_l_init = parent_tied["log_l"]
    # log_sigma_init = parent_tied["log_sigma"]
    # tied_params = np.array([log_l_init, log_sigma_init], dtype=float)

    if cluster_graph.metadata and "opt_params" in cluster_graph.metadata:
        return cluster_graph.metadata["opt_params"]
    else:
        _, params = log_marginal_likelihood(data, entity_graph, form, tied_only=False, is_similarity=is_similarity)
        cluster_graph.metadata.setdefault("opt_params", {}).update(params)
        return params


def build_fast_params_vector(entity_graph, opt_params: Dict[str, Any]) -> np.ndarray:
    """
    Returns a vector [log_l_1, ..., log_l_E, log_sigma] aligned with sorted(entity_graph.all_edges).
    For edges not present in the 'untied' map, uses the mean log-length of those that are.
    """
    edges = sorted(list(entity_graph.all_edges_global))

    if not opt_params or "untied" not in opt_params:
        log_sigma = opt_params["tied"].get("log_sigma")
        log_l = (opt_params.get("tied", {}) or {}).get("log_l", 0.0)
        logging.info(f"Untied params not present in graph for fast scoring. Using tied params: log_sigma = {log_sigma}, log_l = {log_l}")
        log_ls = [log_l] * len(edges)
    else:
        log_sigma = opt_params["untied"].get("log_sigma")
        src = { k:v for k, v in opt_params["untied"]["log_ls"].items() }
        present_logs = [src[e] for e in edges if e in src]
        if not present_logs:
            fill = (opt_params.get("tied", {}) or {}).get("log_l", 0.0)
        else:
            fill = float(np.mean(present_logs))  # mean in log-space
        log_ls = [src.get(e, fill) for e in edges]
    return np.array(log_ls + [log_sigma], dtype=float)


def find_score(cluster_graph, form: StructuralForm, data: np.ndarray, is_similarity: bool, speed: str) -> float:
    """
    Compute log posterior = log P(S,F|D) = log P(D|S) + log P(S|F).
    If speed = 'fast', just evaluate this with given params (weights + sigma).
    If speed = 'med', integrate out params using Laplace approximation, but only run with tied params (all weights the same).
    If speed = 'slow', integrate out params using Laplace approximation, and allow params to vary.
    """
    if isinstance(cluster_graph, ProductClusterGraph):
        cluster_graph = cluster_graph.form_cartesian_product()

    # restrict to assigned entities
    restricted_graph, mask = restrict_to_assigned(cluster_graph)
    if restricted_graph is None:
        return -np.inf, []  # no assigned entities, invalid
    
    logger.debug("Scoring cluster_graph under form %s", form.name)
    assigns = restricted_graph.entity_assignments.tolist()
    adj = {k: sorted(v) for k, v in restricted_graph.latent_adjacency.items()}
    logger.debug("Current entity→latent assignments: %s", assigns)
    logger.debug("Current latent adjacency: %s", adj)

    logger.debug("Data: %s", data)
    logger.debug("Data after excluding unassigned entities: %s", data[mask, :])

    entity_graph = cluster_graph_to_entity_graph(restricted_graph)

    if speed == 'fast':
        if is_similarity:
            opt_params = get_params(cluster_graph, entity_graph, form, data[np.ix_(mask, mask)], is_similarity=True)
            vec = build_fast_params_vector(entity_graph, opt_params)
            neg_ll = neg_log_posterior(vec, data[np.ix_(mask, mask)], entity_graph, tied='none', is_similarity=True)
        else:
            opt_params = get_params(cluster_graph, entity_graph, form, data[mask, :], is_similarity=False)
            vec = build_fast_params_vector(entity_graph, opt_params)
            neg_ll = neg_log_posterior(vec, data[mask, :], entity_graph, tied='none', is_similarity=False)
        ll = -neg_ll
    elif speed == 'med':
        if is_similarity:
            ll, opt_params = log_marginal_likelihood(data[np.ix_(mask, mask)], entity_graph, form, tied_only=True, is_similarity=True)
        else:
            ll, opt_params = log_marginal_likelihood(data[mask, :], entity_graph, form, tied_only=True, is_similarity=False)
    else:
        if is_similarity:
            ll, opt_params = log_marginal_likelihood(data[np.ix_(mask, mask)], entity_graph, form, tied_only=False, is_similarity=True)
        else:
            ll, opt_params = log_marginal_likelihood(data[mask, :], entity_graph, form, tied_only=False, is_similarity=False)
    prior = log_structural_prior(restricted_graph, form)
    total = ll + prior
    logger.debug("ll=%.4f, prior=%.4f, total=%.4f", ll, prior, total)
    return total, opt_params


def restrict_to_assigned(cluster_graph: ClusterGraph):
    """
    Returns (restricted_cluster_graph, mask) where mask selects entities with assignment != -1,
    and the cluster_graph is reduced to just those entities (keeping same latent topology).
    """
    mask = cluster_graph.entity_assignments != -1
    if not np.any(mask):
        return None, mask  # nothing assigned
    new_assign = cluster_graph.entity_assignments[mask].copy()
    # copy latent adjacency and metadata; leave latent IDs as-is
    restricted = cluster_graph.copy()
    restricted.update_entity_assignments(new_assign)
    # rebuild entity mapping: new local j -> old global id of original local i
    keep_idx = np.where(mask)[0]
    old_ent = cluster_graph.latent_to_global_id["entities"]
    new_ent = {j: old_ent[i] for j, i in enumerate(keep_idx)}
    restricted.latent_to_global_id["entities"] = new_ent
    # ensure all present latents have cluster gids (allocator preserved by copy)
    restricted.init_global_maps(entity_global_ids=new_ent)
    return restricted, mask