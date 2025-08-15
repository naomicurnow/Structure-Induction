
from heapq import nlargest
from collections import deque
import random
import numpy as np
from scoring import find_score
from graphs_and_forms import ClusterGraph, ProductClusterGraph
from typing import Set, Optional, Dict, List
from config import EPSILON
import copy


def simplify_graph(G) -> ClusterGraph:
    """
    Simplify a (cluster) graph in-place by repeatedly applying three rules:

      (a) remove any unoccupied cluster node with total cluster-degree ≤ 1
      (b) remove an unoccupied degree-2 node by wiring its two neighbours together
          (if tree, do not remove the root; also avoid removing parents in a hierarchy)
      (c) if tree: if a parent has ≥2 children that each have exactly 1 entity, merge two of them
          else (non-product graphs): if a node has one neighbour + 1 entity, move it to the neighbour and delete

    After each successful change the graph is normalised/updated.
    If one rule returns no change in a round, mark that rule "finished" and only try the other two next.
    Continue until all three rules make no change.
    """
    def sync(axis: str, assignments: np.ndarray):
        if axis == "row":
            G.row_assignments = assignments
            G.row_graph.entity_assignments = assignments
        else:
            G.col_assignments = assignments
            G.col_graph.entity_assignments = assignments
        G.check_assignments()

    def rule_a(G) -> bool:
        counts = G.entity_counts()
        for lid in G.latent_adjacency.keys():
            if counts.get(lid, 0) == 0 and len(G.latent_adjacency.get(lid, ())) <= 1:
                G.remove_latents([lid])
                G.normalise()
                return True
        return False
    
    def rule_b(G) -> bool:
        # if G.__class__.__name__ == "TreeClusterGraph":
        #     return False
        
        counts = G.entity_counts()
        for lid in G.latent_adjacency.keys():
            if counts.get(lid, 0) != 0:
                continue
            neighs = list(G.latent_adjacency.get(lid, ()))
            if len(neighs) != 2:
                continue
            u, v = neighs
            if u != v:
                G.add_edges([(u, v)])
            G.remove_latents([lid])
            G.normalise()
            return True
        return False
    
    def rule_c(G) -> bool:
        """
        Tree branch:
            If a parent has >=2 children that each have exactly 1 entity, merge two of those children.
        Else (non-product graphs only):
            If a node has exactly one neighbour and exactly one entity, move its entity to the neighbour and delete the node.
        """
        if G.__class__.__name__ == "TreeClusterGraph":
            counts = G.entity_counts()
            for p in list(G.internal_ids):
                # children are leaf neighbours
                kids = [c for c in G.latent_adjacency.get(p, set()) if c in G.leaf_ids]
                ones = [c for c in kids if counts.get(c, 0) == 1]
                if len(ones) >= 2:
                    c1, c2 = ones[0], ones[1]
                    # move entity c2 -> c1
                    idx = int(np.where(G.entity_assignments == c2)[0][0])
                    G.entity_assignments[idx] = c1
                    neighs = set(G.latent_adjacency.get(c2, set())) - {c1}
                    G.add_edges([(c1, n) for n in neighs])
                    G.remove_latents([c2])
                    G.leaf_ids.discard(c2)
                    G.normalise()
                    return True
            return False

        # If a node has exactly one neighbour and exactly one entity:
        counts = G.entity_counts()
        for lid in G.latent_adjacency.keys():
            if counts.get(lid, 0) == 1 and len(G.latent_adjacency.get(lid, ())) == 1:
                neigh = list(G.latent_adjacency[lid])[0]
                # move the single entity
                idx = int(np.where(G.entity_assignments == lid)[0][0])
                G.entity_assignments[idx] = neigh
                # delete the node
                G.remove_latents([lid])
                G.normalise()
                return True
        return False
    
    def rule_a_axis(axis: str) -> bool:
        graph = G.row_graph if axis == "row" else G.col_graph
        assignments = G.row_assignments if axis == "row" else G.col_assignments
        counts = graph.entity_counts()

        for lid in list(graph.latent_adjacency.keys()):
            deg = len(graph.latent_adjacency.get(lid, ()))
            if counts.get(lid, 0) == 0 and deg <= 1:
                graph.remove_latents([lid])
                graph.normalise()
                sync(axis, assignments)
                return True
        return False

    def rule_b_axis(axis: str) -> bool:
        graph = G.row_graph if axis == "row" else G.col_graph
        assignments = G.row_assignments if axis == "row" else G.col_assignments
        counts = graph.entity_counts()

        for lid in list(graph.latent_adjacency.keys()):
            if counts.get(lid, 0) != 0:
                continue
            neighs = list(graph.latent_adjacency.get(lid, ()))
            if len(neighs) != 2:
                continue
            u, v = neighs
            if u != v:
                graph.add_edges([(u, v)])
            if np.any(assignments == lid):
                assignments[assignments == lid] = -1
            graph.remove_latents([lid])
            graph.normalise()
            sync(axis, assignments)
            return True
        return False
    
    if G.__class__.__name__ == "ProductClusterGraph":
        RULES_AXIS = (lambda axis: rule_a_axis(axis), lambda axis: rule_b_axis(axis))
        axes = ["row", "col"]
        random.shuffle(axes)
        for axis in axes:
            not_changing: Set[int] = set()
            while True:
                for i, rule in enumerate(RULES_AXIS): # skip last rule -- not applied to product graphs
                    if i in not_changing:
                        continue
                    changed = rule(axis)
                    if not changed:
                        # this rule finished for now; try others next round
                        not_changing.add(i)
                # no rules changed this round; if all 3 are not changing, we are done
                if len(not_changing) == len(RULES_AXIS):
                    break
                # else: keep trying the remaining enabled rules in next round
        return G
    
    RULES = (rule_a, rule_b, rule_c)
    not_changing: Set[int] = set()
    while True:
        # try each rule once per round unless didn't change last round
        for i, rule in enumerate(RULES):
            if i in not_changing:
                continue
            changed = rule(G)
            if not changed:
                # this rule finished for now; try others next round
                not_changing.add(i)

        # no rules changed this round; if all 3 are not changing, we are done
        if len(not_changing) == len(RULES):
            break
        # else: keep trying the remaining enabled rules in next round
    
    return G


def refine_graph_topology(current, current_score, form, data, is_similarity, near_k=4, radius=None):
    """
    Try different ways to change to graph topology to see if it improves the score:
      (a) moving/swapping entities between nodes
      (b) tree/hierarchy reattachments
      (c) product-only: (i) attempt dimension collapse; (ii) moves/swaps on the product graph
    """
    best_graph = simplify_graph(current)
    best_score = current_score
    all_near_misses = []  # list of top k (graph, score) behind best_graph, best_score

    def consider(g, s):
        nonlocal best_graph, best_score, all_near_misses
        if s > best_score + EPSILON:
            best_graph, best_score = g, s
        else:
            all_near_misses.append((g, s))
            all_near_misses = nlargest(near_k, all_near_misses, key=lambda x: x[1])

    # movement/swaps of entities from one cluster to another
    # if product graph, only do this within a component
    if isinstance(best_graph, ProductClusterGraph):
        order = ['row', 'col']
        random.shuffle(order)
        for ax in order:
            graph_ax, score_ax, near_misses_ax = move_and_swap_entities_for_product_graph(
                best_graph, best_score, form, data, is_similarity,
                axis=ax, radius=radius
            )
            consider(graph_ax, score_ax)
            all_near_misses.extend(near_misses_ax)
    else:
        graph, score, near_misses = move_and_swap_entities(best_graph, best_score, form, data, is_similarity, radius=radius)
        consider(graph, score)
        all_near_misses.extend(near_misses)

    # tree / hierarchy reattachments
    if form.name in ("tree", "hierarchy"):
        graph, score, near_misses = reattach_tree_hierarchy(best_graph, best_score, form, data, is_similarity)
        consider(graph, score)
        all_near_misses.extend(near_misses)

    # product-only
    if isinstance(best_graph, ProductClusterGraph):
        # try dimension collapse on rows and cols
        order = ['row', 'col']
        random.shuffle(order)  # randomise whether row or col first
        for ax in order:
            graph_ax, score_ax, near_misses_ax = collapse_dimension(
                best_graph, best_score, form, data, is_similarity,
                axis=ax
            )
            consider(graph_ax, score_ax)
            all_near_misses.extend(near_misses_ax)

        # run the regular (non-product) moves/swaps on the product graph
        prod = best_graph.form_cartesian_product()
        prod_best, prod_best_score, prod_near = move_and_swap_entities(
            prod, best_score, form, data, is_similarity, radius=radius
        )
        lifted_best = product_to_ProductClusterGraph(prod_best, best_graph)
        s_lifted_best = fast_score(lifted_best, form, data, is_similarity)
        consider(lifted_best, s_lifted_best)

        for pcand, ps in prod_near[:4]:
            lifted = product_to_ProductClusterGraph(pcand, best_graph)
            s_lifted = fast_score(lifted, form, data, is_similarity)
            all_near_misses.append((lifted, s_lifted))

    # optimise lengths of the best graph
    best_score, opt_params = find_score(best_graph, form, data, is_similarity, speed='slow')
    best_graph.metadata.setdefault("opt_params", {}).update(opt_params)

    # finalise near misses (top-4 excluding the final best)
    all_near_misses = [ns for ns in all_near_misses if ns[1] <= best_score + EPSILON]
    all_near_misses = nlargest(near_k, all_near_misses, key=lambda x: x[1])
    return best_graph, best_score, all_near_misses


def product_to_ProductClusterGraph(prod_cand, template_grid):
    # axis maps: local axis lids <-> global axis ids
    row_lid_to_rgid = template_grid.row_graph.latent_lid_to_cid
    col_lid_to_cgid = template_grid.col_graph.latent_lid_to_cid
    rgid_to_row_lid = {rgid: lid for lid, rgid in row_lid_to_rgid.items()}
    cgid_to_col_lid = {cgid: lid for lid, cgid in col_lid_to_cgid.items()}

    # product map: prod local lid -> (rgid, cgid)
    prod_lid_to_pair = getattr(prod_cand, "latent_lid_to_cid", None)
    if prod_lid_to_pair is None:
        # legacy fallback (shouldn't happen with the new product code)
        raise TypeError("Product ClusterGraph lacks latent_lid_to_cid mapping.")

    # start from current assignments; overwrite where prod_cand moved things
    new_row = template_grid.row_assignments.copy()
    new_col = template_grid.col_assignments.copy()

    for i, plid in enumerate(prod_cand.entity_assignments):
        if plid == -1:
            continue
        rgid, cgid = prod_lid_to_pair[int(plid)]
        r_lid = rgid_to_row_lid[rgid]
        c_lid = cgid_to_col_lid[cgid]
        new_row[i] = r_lid
        new_col[i] = c_lid

    prod_cand.metadata['row_graph'] = template_grid.row_graph.copy()
    prod_cand.metadata['col_graph'] = template_grid.col_graph.copy()
    prod_cand.metadata['row_assignments'] = new_row
    prod_cand.metadata['col_assignments'] = new_col

    # ensure entity ids present for the undo
    prod_cand.entity_idx_to_eid = {int(i): int(template_grid.entity_idx_to_eid[i])
                                    for i in range(template_grid.n_entities())}

    lifted = prod_cand.undo_cartesian_product()
    lifted = simplify_graph(lifted)
    lifted.check_assignments()
    return lifted


def fast_score(graph, form, data, is_similarity):
    score, opt_params = find_score(graph, form, data, is_similarity, speed='fast')
    graph.metadata.setdefault("opt_params", {}).update(opt_params)
    return score


def move_and_swap_entities_for_product_graph(G: ProductClusterGraph, base_score, form, data, is_similarity,
                                axis: str, radius=None):
    """
    Within-component moves/swaps for ProductClusterGraph along one axis ('row' or 'col').
    Only reassign that axis; keep the other axis fixed.
    """
    assert axis in ('row', 'col')
    near_misses = []
    best_graph = G
    best_score = base_score
    improved = True

    while improved:

        improved = False
        current_graph = best_graph
        chain = current_graph.row_graph if axis == 'row' else current_graph.col_graph
        assignments = current_graph.row_assignments if axis == 'row' else current_graph.col_assignments
        other_assignments = current_graph.col_assignments if axis == 'row' else current_graph.row_assignments

        adj = chain.latent_adjacency
        nodes = list(chain.latent_ids)
        within_k = {u: nodes_within_k(adj, u, radius) for u in nodes}

        counts = {u: int(np.sum(assignments == u)) for u in nodes}
        occupied = [u for u in nodes if counts[u] > 0]
        random.shuffle(occupied)

        # moves
        for u in occupied:
            neighs = sorted(within_k[u])
            random.shuffle(neighs)
            for v in neighs:
                if v == u:
                    continue
                cand = current_graph.copy()
                if axis == 'row':
                    new_row = cand.row_assignments.copy()
                    new_row[assignments == u] = v
                    cand.update_entity_assignments(list(zip(new_row.tolist(), other_assignments.tolist())))
                else:  # axis == 'col'
                    new_col = cand.col_assignments.copy()
                    new_col[assignments == u] = v
                    cand.update_entity_assignments(list(zip(other_assignments.tolist(), new_col.tolist())))
                cand = simplify_graph(cand)
                s = fast_score(cand, form, data, is_similarity)
                if s > best_score + EPSILON:
                    improved = True
                    best_graph, best_score = cand, s
                else:
                    near_misses.append((cand, s))

        # swaps (between two occupied nodes)
        occ_copy = occupied[:]
        random.shuffle(occ_copy)
        for i, u in enumerate(occ_copy):
            for v in occ_copy[i+1:]:
                if v not in within_k[u]:
                    continue
                cand = current_graph.copy()
                if axis == 'row':
                    new_row = cand.row_assignments.copy()
                    mask_u = assignments == u
                    mask_v = assignments == v
                    new_row[mask_u] = v
                    new_row[mask_v] = u
                    cand.update_entity_assignments(list(zip(new_row.tolist(), other_assignments.tolist())))
                else:  # axis == 'col'
                    new_col = cand.col_assignments.copy()
                    mask_u = assignments == u
                    mask_v = assignments == v
                    new_col[mask_u] = v
                    new_col[mask_v] = u
                    cand.update_entity_assignments(list(zip(other_assignments.tolist(), new_col.tolist())))
                cand = simplify_graph(cand)
                s = fast_score(cand, form, data, is_similarity)
                if s > best_score + EPSILON:
                    improved = True
                    best_graph, best_score = cand, s
                else:
                    near_misses.append((cand, s))

    near_misses = nlargest(4, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses


def move_and_swap_entities(G: ClusterGraph, base_score, form, data, is_similarity,
                         radius=None):
    """Moves and swaps for non-product ClusterGraphs."""
    near_misses = []
    best_graph = G
    best_score = base_score
    improved = True

    while improved:

        improved = False
        current_graph = best_graph
        adj = current_graph.latent_adjacency
        nodes = list(current_graph.latent_ids)
        within_k = {u: nodes_within_k(adj, u, radius) for u in nodes}

        assigns = current_graph.entity_assignments
        counts = {u: int(np.sum(assigns == u)) for u in nodes}

        occupied = [u for u in nodes if counts[u] > 0]
        random.shuffle(occupied)

        # moves
        for u in occupied:
            # don't allow large structural change by moving all entities away from an internal node in a hierarchy
            # (also true for trees but here internal nodes are unoccupied by definition)
            if form.name == 'hierarchy' and u not in list(current_graph.leaf_ids()): continue 

            neighs = sorted(within_k[u])
            random.shuffle(neighs)
            for v in neighs:
                if v == u:
                    continue
                if form.name == 'tree' and v not in list(current_graph.leaf_ids): continue
                cand = current_graph.copy()
                na = cand.entity_assignments.copy()
                na[assigns == u] = v
                cand.update_entity_assignments(na)
                cand = simplify_graph(cand)
                s = fast_score(cand, form, data, is_similarity)
                if s > best_score + EPSILON:
                    improved = True
                    best_graph, best_score = cand, s
                else:
                    near_misses.append((cand, s))

        # swaps (between two occupied nodes)
        occ_copy = occupied[:]
        random.shuffle(occ_copy)
        for i, u in enumerate(occ_copy):
            for v in occ_copy[i+1:]:
                if v not in within_k[u]:
                    continue
                cand = current_graph.copy()
                na = cand.entity_assignments.copy()
                mask_u = assigns == u
                mask_v = assigns == v
                na[mask_u] = v
                na[mask_v] = u
                cand.update_entity_assignments(na)
                cand = simplify_graph(cand)
                s = fast_score(cand, form, data, is_similarity)
                if s > best_score + EPSILON:
                    improved = True
                    best_graph, best_score = cand, s
                else:
                    near_misses.append((cand, s))

    near_misses = nlargest(4, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses


def nodes_within_k(adj, start, k=None):
    """
    Return nodes within <= k hops of `start` (including neighbours, excluding start).

    If k is None, return all nodes reachable from `start` (unbounded BFS).
    """
    seen = {start}
    out = set()
    q = deque([start])
    dist = {start: 0}

    while q:
        u = q.popleft()
        if k is not None and dist[u] == k:
            continue
        for v in adj.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            out.add(v)
            q.append(v)
            if k is not None:
                dist[v] = dist[u] + 1
    return out


def reattach_tree_hierarchy(G, base_score, form, data, is_similarity):
    """
    For trees (nodes+entities): reparent entities under any leaf; and move subtree roots.
    For hierarchies (nodes only): try reattaching nodes (no entity moves).
    """
    near_misses = []
    best_graph = G
    best_score = base_score

    improved = True

    while improved:

        improved = False
        G = best_graph

        # (entities) reparent under any leaf — move all entities from leaf a to leaf b
        # if form.name == 'tree':
        #     # work over entities, not leaves
        #     entity_idxs = [int(i) for i in np.where(assigns != -1)[0].tolist()]
        #     random.shuffle(entity_idxs)

        #     for idx in entity_idxs:
        #         u = int(assigns[idx]) # current leaf of this entity

        #         # all possible destinations: any other leaf
        #         dests = list(leaves - {u})
        #         random.shuffle(dests)

        #         for v in dests:
        #             # (explicitly avoids moving to internals because v ∈ leaf_set)
        #             cand = G.copy()
        #             na = cand.entity_assignments.copy()
        #             na[idx] = v
        #             cand.update_entity_assignments(na)
        #             cand = simplify_graph(cand)
        #             s = fast_score(cand, form, data, is_similarity)
        #             if s > best_score + EPSILON:
        #                 improved = True
        #                 best_graph, best_score = cand, s
        #             else:
        #                 near_misses.append((cand, s))

        def subtree_nodes(G, root: int) -> Set[int]:
            """Collect root plus all descendants using children map."""
            out = {root}
            stack = [root]
            while stack:
                x = stack.pop()
                for c in G.children_of(x):
                    if c not in out:
                        out.add(c)
                        stack.append(c)
            return out

        # (nodes) move a subtree rooted at u by reconnecting u to a different t
        nodes = list(G.latent_ids)
        for u in nodes:
            old_par = G.parent.get(u, None)
            if old_par is None:
                # root: do not move
                continue

            u_sub = subtree_nodes(G, u)  # u plus all descendants

            # valid new parents: any node != u, not old_par, and not in u's subtree
            candidates = [t for t in nodes if t != u and t != old_par and t not in u_sub]
            random.shuffle(candidates)

            for t in candidates:
                if form.name == 'tree' and t not in list(G.internal_ids): continue
                cand = G.copy()

                # detach u from old parent in adjacency
                if old_par in cand.latent_adjacency:
                    cand.latent_adjacency[old_par].discard(u)
                cand.latent_adjacency.setdefault(u, set()).discard(old_par)

                # update hierarchy bookkeeping
                cand.set_parent_of(u, t)

                # attach u under new parent in adjacency
                cand.latent_adjacency.setdefault(t, set()).add(u)
                cand.latent_adjacency.setdefault(u, set()).add(t)

                cand.normalise()
                cand = simplify_graph(cand)
                s = fast_score(cand, form, data, is_similarity)
                if s > best_score + EPSILON:
                    improved = True
                    best_graph, best_score = cand, s
                else:
                    near_misses.append((cand, s))

    near_misses = nlargest(4, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses



def collapse_dimension(G: ProductClusterGraph, base_score, form, data,
                                   is_similarity, axis: str):
    """
    Try to collapse dimensions by effectively removing one node from a row/col chain:
      - For a candidate row/col r with >=1 entities, if the full graph has more empty cells
        than the number of occupied cells in r's row/col, reassign those entities into
        nearest empty cells (in product space), then drop the component node.
    """
    assert axis in ('row', 'col')
    near_misses = []
    best_graph = G
    best_score = base_score

    product = G.form_cartesian_product()
    counts = product.entity_counts() # product local lid -> count

    # axis maps: local lid <-> global axis id
    row_lid_to_rgid = G.row_graph.latent_lid_to_cid
    col_lid_to_cgid = G.col_graph.latent_lid_to_cid
    rgid_to_row_lid = {rgid: lid for lid, rgid in row_lid_to_rgid.items()}
    cgid_to_col_lid = {cgid: lid for lid, cgid in col_lid_to_cgid.items()}

    # product mapping: product local lid -> (rgid, cgid)
    prod_lid_to_pair = product.latent_lid_to_cid  # {plid: (rgid, cgid)}

    adj = product.latent_adjacency

    def shortest_target(srcL: int, allowed_targets: Set[int]) -> Optional[int]:
        """Return the closest target (by BFS hops) among allowed_targets; None if unreachable."""
        if srcL in allowed_targets:
            return srcL
        seen = {srcL}
        q = deque([srcL])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v in seen:
                    continue
                if v in allowed_targets:
                    return v
                seen.add(v)
                q.append(v)
        return None

    # candidates in this axis
    chain_nodes = (G.row_graph.order if axis == 'row' else G.col_graph.order)

    for node in chain_nodes:
        if axis == 'row':
            rgid = row_lid_to_rgid[node]
            # line cells: all product lids whose row id equals this rgid
            line_cells = [plid for plid, (rgi, cgi) in prod_lid_to_pair.items() if rgi == rgid]
            members = np.where(G.row_assignments == node)[0]
            # empty targets outside this row
            allowed_empty = {plid for plid, k in counts.items() if k == 0 and prod_lid_to_pair[plid][0] != rgid}
        else:
            cgid = col_lid_to_cgid[node]
            line_cells = [plid for plid, (rgi, cgi) in prod_lid_to_pair.items() if cgi == cgid]
            members = np.where(G.col_assignments == node)[0]
            allowed_empty = {plid for plid, k in counts.items() if k == 0 and prod_lid_to_pair[plid][1] != cgid}

        if members.size == 0:
            continue

        # source product cells on this line that actually have entities
        source_cells = [L for L in line_cells if counts.get(L, 0) > 0]

        # need at least one empty target per source cell
        if len(allowed_empty) < len(source_cells):
            continue

        # assign empty target per source cell (move all entities from source to same cell), greedy by nearest
        working_empty = set(allowed_empty)
        mapping = {}  # source_plid -> target_plid
        random.shuffle(source_cells)
        feasible = True
        for src in source_cells:
            tgt = shortest_target(src, working_empty)
            if tgt is None:
                feasible = False
                break
            mapping[src] = tgt
            working_empty.remove(tgt)

        if not feasible or not mapping:
            continue

        # apply the batch reassignment
        cand = G.copy()
        new_row = cand.row_assignments.copy()
        new_col = cand.col_assignments.copy()

        for src_plid, tgt_plid in mapping.items():
            rs_gid, cs_gid = prod_lid_to_pair[src_plid]
            rt_gid, ct_gid = prod_lid_to_pair[tgt_plid]
            rs_lid = rgid_to_row_lid[rs_gid]
            cs_lid = cgid_to_col_lid[cs_gid]
            rt_lid = rgid_to_row_lid[rt_gid]
            ct_lid = cgid_to_col_lid[ct_gid]
            # move all entities currently in (rs_lid, cs_lid) together to (rt_lid, ct_lid)
            idxs = np.where((G.row_assignments == rs_lid) & (G.col_assignments == cs_lid))[0]
            if idxs.size:
                new_row[idxs] = rt_lid
                new_col[idxs] = ct_lid

        cand.update_entity_assignments(list(zip(new_row.tolist(), new_col.tolist())))

        # simplify (this will remove the now-unoccupied row/col node)
        cand = simplify_graph(cand)

        s = fast_score(cand, form, data, is_similarity)
        if s > best_score + EPSILON:
            best_graph, best_score = cand, s
        else:
            near_misses.append((cand, s))

    near_misses = nlargest(4, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses


def move_individual_entities_for_product_graph(
    G, base_score, form, data, is_similarity, near_k=4, radius=3
):
    """
    Moving individual entities between cells in the cartesian product graph.
    """
    near_misses = []
    best_graph = G
    best_score = base_score
    improved = True

    while improved:
        improved = False
        current = best_graph

        prod = current.form_cartesian_product()
        assigns_prod = prod.entity_assignments

        row_lid_to_rgid = current.row_graph.latent_lid_to_cid
        col_lid_to_cgid = current.col_graph.latent_lid_to_cid
        rgid_to_row_lid = {rgid: lid for lid, rgid in row_lid_to_rgid.items()}
        cgid_to_col_lid = {cgid: lid for lid, cgid in col_lid_to_cgid.items()}

        # product lid -> (rgid, cgid)
        lid_to_pair = prod.latent_lid_to_cid

        # adjacency + within-k neighborhoods on the product graph
        adj = prod.latent_adjacency
        prod_nodes = list(adj.keys())
        within_k = {u: nodes_within_k(adj, u, radius) for u in prod_nodes}

        # occupancy: product lid -> list(entity indices)
        cell_to_entities: Dict[int, List[int]] = {}
        for i, L in enumerate(assigns_prod.tolist()):
            if L == -1:
                continue
            cell_to_entities.setdefault(int(L), []).append(i)

        # iterate occupied cells in random order
        occupied_cells = list(cell_to_entities.keys())
        random.shuffle(occupied_cells)

        accepted_this_round = False

        for src in occupied_cells:
            # neighbors within radius (exclude staying put)
            nbrs = sorted(within_k.get(src, set()))
            random.shuffle(nbrs)

            src_entities = cell_to_entities.get(src, [])
            if not src_entities:
                continue

            for i in src_entities:
                for dst in nbrs:
                    if dst == src:
                        continue

                    # map dst -> (rgid, cgid) -> (row_lid, col_lid)
                    rgid_t, cgid_t = lid_to_pair[dst]
                    rt = rgid_to_row_lid[rgid_t]
                    ct = cgid_to_col_lid[cgid_t]

                    cand = current.copy()
                    new_row = cand.row_assignments.copy()
                    new_col = cand.col_assignments.copy()
                    new_row[i] = rt
                    new_col[i] = ct
                    cand.update_entity_assignments(list(zip(new_row.tolist(), new_col.tolist())))

                    cand = simplify_graph(cand)
                    s = fast_score(cand, form, data, is_similarity)

                    if s > best_score + EPSILON:
                        best_graph, best_score = cand, s
                        improved = True
                        accepted_this_round = True
                        break  # accept first improvement; recompute neighborhoods next loop
                    else:
                        near_misses.append((cand, s))

                if accepted_this_round:
                    break
            if accepted_this_round:
                break

        # if no candidate improved, exit outer while; else loop again with updated best_graph

    near_misses = nlargest(near_k, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses


def move_individual_entities(
    G, base_score, form, data, is_similarity, near_k=4, radius=3
):
    """
    """
    if form.name in ('grid', 'cylinder'):
        return move_individual_entities_for_product_graph(G, base_score, form, data, is_similarity, near_k, radius)

    near_misses = []
    best_graph = G
    best_score = base_score
    improved = True

    while improved:
        improved = False
        current = best_graph

        adj = current.latent_adjacency
        nodes = list(current.latent_ids)
        within_k = {u: nodes_within_k(adj, u, radius) for u in nodes}

        assigns = current.entity_assignments
        counts = current.entity_counts()

        occupied_nodes = [u for u in nodes if counts.get(u, 0) > 0]
        random.shuffle(occupied_nodes)

        accepted_this_round = False

        for u in occupied_nodes:
            if form.name != "partition" and counts.get(u, 0) == 1:
                continue

            # candidate destinations within radius
            nbrs = sorted(within_k.get(u, set()))
            random.shuffle(nbrs)

            # all entity indices at u
            src_indices = np.where(assigns == u)[0].tolist()

            for i in src_indices:
                for v in nbrs:
                    if v == u:
                        continue

                    if form.name == 'tree' and v not in list(current.leaf_ids): continue

                    cand = current.copy()
                    na = cand.entity_assignments.copy()
                    na[i] = v
                    cand.update_entity_assignments(na)
                    cand = simplify_graph(cand)
                    s = fast_score(cand, form, data, is_similarity)

                    if s > best_score + EPSILON:
                        best_graph, best_score = cand, s
                        improved = True
                        accepted_this_round = True
                        break  # accept first improvement; restart outer loop
                    else:
                        near_misses.append((cand, s))
                if accepted_this_round:
                    break
            if accepted_this_round:
                break

    near_misses = nlargest(near_k, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses
