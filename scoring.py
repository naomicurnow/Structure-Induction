
from graphs_and_forms import ClusterGraph, StructuralForm, ProductClusterGraph, EntityGraph
from structural_prior import log_structural_prior
from likelihoods import log_marginal_likelihood, neg_log_posterior
from typing import Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


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


def cluster_graph_to_entity_graph(cluster_graph: ClusterGraph) -> EntityGraph:
    n_entities = cluster_graph.n_entities()
    latent_lids = sorted(cluster_graph.latent_ids)

    idx_to_eid = cluster_graph.entity_idx_to_eid
    lid_to_cid = cluster_graph.latent_lid_to_cid

    entity_gids = [int(idx_to_eid[i]) for i in range(n_entities)]
    cluster_cids = [lid_to_cid[lid] for lid in latent_lids]

    eg = EntityGraph(entity_gids=entity_gids, cluster_cids=cluster_cids)

    # entity→cluster edges (external)
    for idx in range(n_entities):
        lid = int(cluster_graph.entity_assignments[idx])
        if lid == -1:
            continue  # orphaned
        cid = lid_to_cid.get(lid)
        if cid is None:
            raise KeyError(f"ClusterGraph has assignment to latent lid {lid}, "
                           f"but no global cid mapping was found.")
        eg.add_external(eid=entity_gids[idx], cid=cid)

    # cluster↔cluster edges (internal)
    for a_lid, neighs in cluster_graph.latent_adjacency.items():
        a_cid = lid_to_cid.get(a_lid)
        for b_lid in neighs:
            if a_lid < b_lid:  # emit each undirected edge once
                b_cid = lid_to_cid.get(b_lid)
                eg.add_internal(a_cid, b_cid)

    return eg


def get_params(cluster_graph: ClusterGraph, entity_graph: EntityGraph, form: StructuralForm, data: np.ndarray, is_similarity: bool) -> Dict[str, Any]:

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
        return params


def build_fast_params_vector(entity_graph: EntityGraph, opt_params: Dict[str, Any]) -> np.ndarray:
    """
    Returns a vector [log_l_1, ..., log_l_E, log_sigma] aligned with sorted(entity_graph.edges).
    For edges not present in the 'untied' map, uses the mean log-length of those that are.
    """
    edges = sorted(list(entity_graph.edges), key=edge_sort_key)

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


def find_score(cluster_graph: ClusterGraph, form: StructuralForm, data: np.ndarray, is_similarity: bool, speed: str) -> float:
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

    # build new idx->eid map (new local indices 0..k-1 -> original eids)
    keep_idx = np.where(mask)[0]
    old_idx_to_eid = cluster_graph.entity_idx_to_eid
    new_idx_to_eid = {j: int(old_idx_to_eid[i]) for j, i in enumerate(keep_idx)}

    restricted = cluster_graph.copy()
    restricted.update_entity_assignments(new_assign)
    restricted.entity_idx_to_eid = new_idx_to_eid
    restricted.init_id_maps(entity_idx_to_eid=new_idx_to_eid)
    return restricted, mask