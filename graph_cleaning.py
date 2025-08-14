from heapq import nlargest
from collections import deque
import random
import numpy as np
from scoring import find_score
from graphs_and_forms import ClusterGraph, ProductClusterGraph
from typing import Set
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

    After *each* successful change the graph is normalised/updated.
    If one rule returns no change in a round, mark that rule "finished" and only try the other two next.
    Continue until all three rules make no change.
    """
    def refresh_product_index_maps(product):
        # rebuild _row_gid/_col_gid and reverse maps after any row/col change
        gid = 0
        product._row_gid = {}
        for r in product.row_graph.order:
            product._row_gid[r] = gid
            gid += 1
        product._col_gid = {}
        for c in product.col_graph.order:
            product._col_gid[c] = gid
            gid += 1
        product._gid_to_row = {g: r for r, g in product._row_gid.items()}
        product._gid_to_col = {g: c for c, g in product._col_gid.items()}

    def sync(axis: str, assignments: np.ndarray):
        # keep productgraph and subgraphs aligned
        if axis == "row":
            G.row_assignments = assignments
            G.row_graph.entity_assignments = assignments
        else:
            G.col_assignments = assignments
            G.col_graph.entity_assignments = assignments

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
            # TreeClusterGraph: use internal/leaf sets
            counts = G.entity_counts()
            for p in list(getattr(G, "internal_ids", set())):
                # children are leaf neighbours
                kids = [c for c in G.latent_adjacency.get(p, set())
                        if c in getattr(G, "leaf_ids", set())]
                ones = [c for c in kids if counts.get(c, 0) == 1]
                if len(ones) >= 2:
                    c1, c2 = ones[0], ones[1]
                    # move entity c2 -> c1
                    idx = int(np.where(G.entity_assignments == c2)[0][0])
                    G.entity_assignments[idx] = c1
                    neighs = set(G.latent_adjacency.get(c2, set())) - {c1}
                    G.add_edges([(c1, n) for n in neighs])
                    G.remove_latents([c2])
                    # keep metadata coherent: c2 was a leaf
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
                refresh_product_index_maps(G)
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
            refresh_product_index_maps(G)
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
    
    RULES = (
        lambda g: rule_a(g),
        lambda g: rule_b(g),
        lambda g: rule_c(g),
    )

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
    Post-selection local refinements:
      (a) within-component moving/swapping (all graph types; product uses component chains)
      (b) tree/hierarchy reattachments (best-effort: works if the form is a tree/hierarchy)
      (c) product-only: (i) attempt dimension collapse; (ii) full-graph moves/swaps
    Returns: (best_graph, best_score, near_misses[list of (graph,score)])
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
            # keep only top-k near misses
            all_near_misses = nlargest(near_k, all_near_misses, key=lambda x: x[1])

    # movement/swaps of entities from one cluster to another
    # if product graph, only do this within a component
    if isinstance(best_graph, ProductClusterGraph):
        order = ['row', 'col']
        random.shuffle(order)  # randomise whether row or col first
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

    # product-only passes
    if isinstance(best_graph, ProductClusterGraph):
        # (i) try dimension collapse on rows and cols
        order = ['row', 'col']
        random.shuffle(order)  # randomise whether row or col first
        for ax in order:
            graph_ax, score_ax, near_misses_ax = collapse_dimension(
                best_graph, best_score, form, data, is_similarity,
                axis=ax
            )
            consider(graph_ax, score_ax)
            all_near_misses.extend(near_misses_ax)

        # lift a product-graph candidate back to a ProductClusterGraph
        def _lift_product_to_grid(prod_cand, template_grid):
            rows = template_grid.row_graph.order
            cols = template_grid.col_graph.order
            # make a mapping from product lid -> (row_id, col_id)
            lid_to_cell = {}
            lid = 0
            for r in rows:
                for c in cols:
                    lid_to_cell[lid] = (r, c)
                    lid += 1

            new_row = template_grid.row_assignments.copy()
            new_col = template_grid.col_assignments.copy()
            for i, L in enumerate(prod_cand.entity_assignments):
                if L == -1:
                    continue
                r, c = lid_to_cell[int(L)]
                new_row[i] = r
                new_col[i] = c

            lifted = template_grid.copy()
            lifted.update_entity_assignments(list(zip(new_row.tolist(), new_col.tolist())))
            lifted = simplify_graph(lifted)
            return lifted

        # build current product graph
        prod = best_graph.form_cartesian_product()  # -> ClusterGraph over grid cells

        # run the regular (non-product) moves/swaps on the product graph
        prod_best, prod_best_score, prod_near = move_and_swap_entities(
            prod, best_score, form, data, is_similarity, radius=radius
        )

        # lift the best candidate back to ProductClusterGraph and score
        lifted_best = _lift_product_to_grid(prod_best, best_graph)
        s_lifted_best = fast_score(lifted_best, form, data, is_similarity)
        consider(lifted_best, s_lifted_best)

        # also lift a few near-misses
        for pcand, ps in prod_near[:4]:
            lifted = _lift_product_to_grid(pcand, best_graph)
            s_lifted = fast_score(lifted, form, data, is_similarity)
            all_near_misses.append((lifted, s_lifted))

    # optimise lengths of the best graph
    best_score, opt_params = find_score(best_graph, form, data, is_similarity, speed='slow')
    best_graph.metadata.setdefault("opt_params", {}).update(opt_params)

    # finalise near misses (top-4 excluding the final best)
    all_near_misses = [ns for ns in all_near_misses if ns[1] <= best_score + EPSILON]
    all_near_misses = nlargest(near_k, all_near_misses, key=lambda x: x[1])
    return best_graph, best_score, all_near_misses


def fast_score(graph, form, data, is_similarity):
    score, opt_params = find_score(graph, form, data, is_similarity, speed='fast')
    graph.metadata.setdefault("opt_params", {}).update(opt_params)
    return score


# =========================
# Helpers for (a)
# =========================

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
        nodes = list(chain.latent_nodes)
        within_k = {u: _within_k(adj, u, radius) for u in nodes}

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
    """
    moves and swaps.
    """
    near_misses = []
    best_graph = G
    best_score = base_score
    improved = True

    while improved:

        improved = False
        current_graph = best_graph
        adj = current_graph.latent_adjacency
        nodes = list(current_graph.latent_nodes)
        within_k = {u: _within_k(adj, u, radius) for u in nodes}

        assigns = current_graph.entity_assignments
        counts = {u: int(np.sum(assigns == u)) for u in nodes}

        occupied = [u for u in nodes if counts[u] > 0]
        random.shuffle(occupied)

        # moves
        for u in occupied:
            # don't allow large structural change by moving all entities away from an internal node in a hierarchy
            # (also true for trees but here internal nodes are unoccupied by definition)
            if form.name == 'hierarchy' and u not in list(current_graph.leaf_nodes): continue 

            neighs = sorted(within_k[u])
            random.shuffle(neighs)
            for v in neighs:
                if v == u:
                    continue
                if form.name == 'tree' and v not in list(current_graph.leaf_nodes): continue
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


def _within_k(adj, start, k=None):
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


# =========================
# (b) Tree/Hierarchy reattachments (best-effort)
# =========================

def reattach_tree_hierarchy(G, base_score, form, data, is_similarity):
    """
    For trees (nodes+entities): reparent entities under any leaf; and move subtree roots.
    For hierarchies (nodes only): try reattaching nodes (no entity moves here).
    """
    near_misses = []
    best_graph = G
    best_score = base_score

    improved = True

    while improved:

        improved = False
        G = best_graph
        leaves = G.leaf_nodes
        assigns = G.entity_assignments

        # (entities) reparent under any leaf — move all entities from leaf a to leaf b
        if form.name == 'tree':
            # work over entities, not leaves
            entity_idxs = [int(i) for i in np.where(assigns != -1)[0].tolist()]
            random.shuffle(entity_idxs)

            for idx in entity_idxs:
                u = int(assigns[idx]) # current leaf of this entity

                # all possible destinations: any other leaf
                dests = list(leaves - {u})
                random.shuffle(dests)

                for v in dests:
                    # (explicitly avoids moving to internals because v ∈ leaf_set)
                    cand = G.copy()
                    na = cand.entity_assignments.copy()
                    na[idx] = v
                    cand.update_entity_assignments(na)
                    cand = simplify_graph(cand)
                    s = fast_score(cand, form, data, is_similarity)
                    if s > best_score + EPSILON:
                        improved = True
                        best_graph, best_score = cand, s
                    else:
                        near_misses.append((cand, s))

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

        # (nodes) move a "subtree" rooted at u by reconnecting u to a different t
        nodes = list(G.latent_nodes)
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
                if form.name == 'tree' and t not in list(G.internal_nodes): continue
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


# =========================
# (c) Product-only passes
# =========================

def collapse_dimension(G: ProductClusterGraph, base_score, form, data,
                                   is_similarity, axis: str):
    """
    (c)(i) Try to collapse dimensions by effectively removing one node from a row/col chain:
      - For a candidate row/col r with >=1 entities, if the full graph has more empty cells
        than the number of occupied cells in r's row/col, reassign those entities into
        nearest empty cells (in product space), then drop the component node.
    Uses FAST (untied) scoring via params for speed.
    """
    assert axis in ('row', 'col')
    near_misses = []
    best_graph = G
    best_score = base_score

    # Build product graph + counts
    product = G.form_cartesian_product()
    counts = product.entity_counts()

    rows = G.row_graph.order
    cols = G.col_graph.order
    cell_to_lid = {}
    lid_to_cell = {}
    lid = 0
    for r in rows:
        for c in cols:
            cell_to_lid[(r, c)] = lid
            lid_to_cell[lid] = (r, c)
            lid += 1

    # precompute empty cells
    empty_cells = [L for L, k in counts.items() if k == 0]

    # Manhattan distance
    def manhattan(l1, l2):
        (r1, c1) = lid_to_cell[l1]; (r2, c2) = lid_to_cell[l2]
        return abs(r1 - r2) + abs(c1 - c2)

    # candidates in this axis
    chain_nodes = (G.row_graph.order if axis == 'row' else G.col_graph.order)

    for node in chain_nodes:
        # entities attached to this row / col
        if axis == 'row':
            members = np.where(G.row_assignments == node)[0]
            line_cells = [cell_to_lid[(node, c)] for c in cols]
            # exclude empty cells in the *same row* from the target pool
            allowed_empty = [L for L in empty_cells if lid_to_cell[L][0] != node]
        else:
            members = np.where(G.col_assignments == node)[0]
            line_cells = [cell_to_lid[(r, node)] for r in rows]
            # exclude empty cells in the *same col* from the target pool
            allowed_empty = [L for L in empty_cells if lid_to_cell[L][1] != node]

        if members.size == 0:
            continue

        # source product cells on this line that actually have entities
        source_cells = [L for L in line_cells if counts.get(L, 0) > 0]

        # need at least one empty target per *source cell*
        if len(allowed_empty) < len(source_cells):
            continue

        # assign empty target per source cell (move all entities from source to same cell), greedy by nearest
        working_empty = set(allowed_empty)
        mapping = {}  # source_lid -> target_lid
        random.shuffle(source_cells)
        for src in source_cells:
            # choose nearest currently-empty cell outside this component
            target = min(working_empty, key=lambda L: manhattan(src, L))
            working_empty.remove(target)
            mapping[src] = target

        # apply the batch reassignment
        cand = G.copy()
        new_row = cand.row_assignments.copy()
        new_col = cand.col_assignments.copy()

        for src_lid, tgt_lid in mapping.items():
            rs, cs = lid_to_cell[src_lid]
            rt, ct = lid_to_cell[tgt_lid]
            # all entities currently in (rs, cs) move together to (rt, ct)
            idxs = np.where((G.row_assignments == rs) & (G.col_assignments == cs))[0]
            if idxs.size:
                new_row[idxs] = rt
                new_col[idxs] = ct

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
    Product-level *individual* moves:
      - Build the product ClusterGraph (grid cells as latent nodes).
      - For each occupied product cell, and for each entity in that cell,
        try moving it to any product cell within `radius` hops (default 3).
      - If a move improves the score, accept immediately and restart (greedy hill-climb).
      - Destination filter: if the candidate graph exposes `leaf_nodes` (tree-like),
        do not move into non-leaf (internal) nodes.

    Returns: (best_graph: ProductClusterGraph, best_score: float, near_misses: list[(graph, score)])
    """
    near_misses = []
    best_graph = G
    best_score = base_score
    improved = True

    while improved:
        improved = False
        current = best_graph

        # --- product graph + mappings ---
        prod = current.form_cartesian_product()  # -> ClusterGraph over product cells
        assigns_prod = prod.entity_assignments  # length n_entities, each is a product lid or -1

        # map product lid <-> (row_id, col_id) using current row/col orders
        rows = current.row_graph.order
        cols = current.col_graph.order
        lid_to_cell = {}
        cell_to_lid = {}
        lid = 0
        for r in rows:
            for c in cols:
                lid_to_cell[lid] = (r, c)
                cell_to_lid[(r, c)] = lid
                lid += 1

        # adjacency + within-k neighborhoods on the *product* graph
        adj = prod.latent_adjacency
        prod_nodes = list(prod.latent_adjacency.keys())
        within_k = {u: _within_k(adj, u, radius) for u in prod_nodes}

        # occupancy: product lid -> list(entity indices)
        cell_to_entities = {}
        for i, L in enumerate(assigns_prod.tolist()):
            if L == -1:
                continue
            cell_to_entities.setdefault(int(L), []).append(i)

        # iterate occupied cells in random order to diversify search
        occupied_cells = list(cell_to_entities.keys())
        random.shuffle(occupied_cells)

        accepted_this_round = False

        for srcL in occupied_cells:
            # neighbors within radius (exclude staying put)
            nbrs = sorted(within_k.get(srcL, set()))
            random.shuffle(nbrs)

            src_entities = cell_to_entities.get(srcL, [])
            if not src_entities:
                continue

            for i in src_entities:
                for dstL in nbrs:
                    if dstL == srcL:
                        continue

                    # move single entity i to dstL = (rt, ct)
                    rt, ct = lid_to_cell[dstL]
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

        # if no candidate improved, we will exit outer while; else loop again with updated best_graph

    near_misses = nlargest(near_k, near_misses, key=lambda x: x[1])
    return best_graph, best_score, near_misses


def move_individual_entities(
    G, base_score, form, data, is_similarity, near_k=4, radius=3
):
    """
    Non-product *individual* moves on a ClusterGraph/Hierarchy/Tree/etc.

    Algo:
      - For each occupied latent node u, for each entity i at u,
        try moving i to any node within <= `radius` hops (via latent adjacency).
      - If a move improves the score, accept immediately and restart (greedy).
      - Destination filter: if the graph exposes `leaf_nodes` (tree),
        only allow destinations in leaves.

    Returns: (best_graph, best_score, near_misses[list of (graph,score)])
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

        # adjacency + within-k on current graph
        adj = current.latent_adjacency
        nodes = list(current.latent_nodes)
        within_k = {u: _within_k(adj, u, radius) for u in nodes}

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
            if not src_indices:
                continue

            for i in src_indices:
                for v in nbrs:
                    if v == u:
                        continue

                    if form.name == 'tree' and v not in list(current.leaf_nodes): continue

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
