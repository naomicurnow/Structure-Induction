
import numpy as np
from typing import Tuple, List, Dict
from graphs_and_forms import ClusterGraph, ProductClusterGraph, StructuralForm, SplitProposal
from graph_cleaning import refine_graph_topology, move_individual_entities, simplify_graph
from scoring import find_score
from config import *
import random
import logging

logger = logging.getLogger(__name__)


def search_over_forms(forms: List[StructuralForm], data: np.ndarray, n_entities: int, is_similarity: bool) -> Dict[str, Tuple[object, float]]:
    results = {}
    for form in forms:
        logger.important("***** Looking at form: %s *****", form.name)
        graphs = greedy_search(form, data, n_entities, is_similarity, restarts=N_RESTARTS)
        graphs.sort(key=lambda x: x[1], reverse=True)  # higher score first
        top_graph, top_score = graphs[0]
        for i, (cg, score) in enumerate(graphs, 1):
            logger.important(f"Graph #{i} (score={score:.2f})")
            logger.important(cg)      
            logger.important("")
        results[form.name] = (top_graph, top_score)
        logger.important("***** Completed form %s with best score %.4f *****", form.name, top_score)
    return results


def greedy_search(form: StructuralForm,
                  data: np.ndarray,
                  n_entities: int,
                  is_similarity: bool,
                  restarts: int) -> list[tuple[object, float]]:
    """
    Greedy top-down search for best cluster_graph under a given form.
    One best graph per restart is produced (local optimum). Returns list of (ClusterGraph, score).
    Higher score is better (maximizing log posterior: likelihood + prior).
    """
    best_graphs = []

    for ri in range(restarts):
        logging.info("*** RESTART %d ***", ri+1)
        current = form.initial_structure(n_entities)
        current_score, opt_params = find_score(current, form, data, is_similarity, speed='slow')
        current.metadata.setdefault("opt_params", {}).update(opt_params)
        best_this_restart = (current, current_score)

        improved = True
        while improved:
            improved = False
            candidate_graphs = []

            # get all split proposals in one pass
            proposals = form.propose_splits(current)

            for P in proposals:
                if len(P.parent_members) <= 1:
                    continue # can't split
                # logger.info("Splitting a node with %d entities, children IDs=%s", len(P.parent_members), [c for c in P.children])
                new_graph, new_score = split_cluster(P, current, current_score, form, data, is_similarity)
                # print(new_graph)
                # logger.debug("Split result for cluster %d: new_score=%.6f (current_score=%.6f)", cluster_id, new_score, current_score)
                candidate_graphs.append((new_graph, new_score))

            # if product graph (grid or cylinder), also test splits between clusters in the full grid
            # not just between clusters in the component graphs
            if isinstance(current, ProductClusterGraph):
                candidate_graphs = candidate_graphs + find_product_splits(current, current_score, form, data, is_similarity)

            if not candidate_graphs:
                break

            # pick best candidate among all possible single-cluster splits
            candidate_graphs.sort(key=lambda x: x[1], reverse=True)  # higher score first
            top_graph, top_score = candidate_graphs[0]

            refined_graph, refined_score, near_misses = refine_graph_topology(
                top_graph, top_score, form, data, is_similarity, near_k=4, radius=3
            )
            if refined_score > top_score:
                top_graph, top_score = refined_graph, refined_score

            # the new graph scores worse than the last round --
            # do one last refinement with more costly searching
            # (swaps / moves between all nodes, moving individual entities)
            # and optimise lengths
            # else, exit loop
            if top_score < current_score + EPSILON:
                refined_graph, refined_score, near_misses = refine_graph_topology(
                    top_graph, top_score, form, data, is_similarity, near_k=4, radius=None
                )
                if refined_score > top_score:
                    top_graph, top_score = refined_graph, refined_score
                refined_graph, refined_score, near_misses = move_individual_entities(
                    top_graph, top_score, form, data, is_similarity, near_k=4, radius=None
                )
                if refined_score > top_score:
                    top_graph, top_score = refined_graph, refined_score
                top_score, opt_params = find_score(top_graph, form, data, is_similarity, speed='slow')
                top_graph.metadata.setdefault("opt_params", {}).update(opt_params)

            logger.info("Top score: %.6f; Graph: %s", top_score, top_graph)
                
            if top_score > current_score + EPSILON:
                current = top_graph
                current_score = top_score
                improved = True
                # update best for this restart
                if current_score > best_this_restart[1]:
                    best_this_restart = (current, current_score)
            else:
                logger.debug("Restart %d: no split improved the score (top delta = %.6g)", ri, top_score - current_score)

        best_graphs.append(best_this_restart)

    return best_graphs


def find_product_splits(current, base_score, form, data, is_similarity):
    # Build product graph and a (lid <-> (row,col)) mapping that mirrors ProductForm.form_cartesian_product
    product_graph = current.form_cartesian_product()

    rows = current.row_graph.order
    cols = current.col_graph.order
    cell_to_lid = {}
    lid_to_cell = {}
    _lid = 0
    for r in rows:
        for c in cols:
            cell_to_lid[(r, c)] = _lid
            lid_to_cell[_lid] = (r, c)
            _lid += 1

    # count entities per product latent node
    prod_assign = product_graph.entity_assignments
    counts = {lid: 0 for lid in product_graph.latent_adjacency.keys()}
    for a in prod_assign:
        if a != -1:
            counts[a] += 1

    # for each occupied product node with >1 entities,
    # try splitting with each EMPTY neighbour, choose best.
    split_graphs = []
    for node in sorted(product_graph.latent_adjacency.keys()):
        if counts.get(node, 0) <= 1:
            continue  # nothing to split
        # entities currently in this product cell
        members = np.where(prod_assign == node)[0]

        for neigh in sorted(product_graph.latent_adjacency[node]):
            if counts.get(neigh, 0) != 0:
                continue  # only consider empty neighbours

            # build a split proposal at the product level: children are (node, neigh)
            prop = SplitProposal(
                split_graph=product_graph,
                parent_members=members.tolist(),
                children=(node, neigh),
                axis=None
            )

            # run the split heuristic but on the product graph
            moved_prod_graph, moved_score = split_cluster(
                prop, product_graph, base_score, form, data, is_similarity
            )

            # map the resulting product assignments back into row/col assignments
            # to get a candidate ProductClusterGraph (to keep structure consistent across search).
            cand_prod = current.copy()
            new_row_assign = cand_prod.row_assignments.copy()
            new_col_assign = cand_prod.col_assignments.copy()

            moved_assign = moved_prod_graph.entity_assignments
            for i, lid in enumerate(moved_assign):
                if lid == -1:
                    # keep whatever it had (orphan stays orphan in both dims)
                    continue
                r, c = lid_to_cell[lid]
                new_row_assign[i] = r
                new_col_assign[i] = c

            cand_prod.update_entity_assignments(list(zip(new_row_assign.tolist(),
                                                            new_col_assign.tolist())))
            
            split_graphs.append((cand_prod, moved_score))

    return split_graphs


def split_cluster(proposal, current, current_score, form: StructuralForm,
                  data: np.ndarray, is_similarity: bool, near_k: int = 4) -> Tuple[ClusterGraph, float]:
    """
    Try splitting one cluster via form.split_cluster, then evaluate all seed-pair refinements
    with greedy assignment. Returns best improved graph and its score, or (original, base_score).
    """
    child1, child2 = proposal.children[0], proposal.children[1]
    seeds = set()
    
    if len(proposal.parent_members) <= 5: # exhaustive all unordered pairs
        for i in range(len(proposal.parent_members)):
            for j in range(i + 1, len(proposal.parent_members)):
                seeds.add((proposal.parent_members[i], proposal.parent_members[j]))
    else:
        for m in proposal.parent_members:
            others = [o for o in proposal.parent_members if o != m]
            chosen = list(np.random.choice(others, size=PARTNERS_PER, replace=False))
            for o in chosen:
                seeds.add((m, o))

    seeds = list(seeds)
    logger.debug("Generated %d seed pairs to try", len(seeds))

    # two stage optimisation
    # when searching through graphs to decide where to place each entity into the child nodes,
    # just return the log posterior with previously set weights as faster.
    # then, optimise the weights of the top scoring graph to return.

    candidate_graphs = []

    for (a, b) in seeds:
        
        for (seed_a, seed_b) in [(a, b),(b, a)]:
            logger.debug("Trying seed pair (a=%d, b=%d)", seed_a, seed_b)
            # initialise candidate with topology and seed assignment
            candidate = proposal.split_graph.copy()

            # seed a and b
            if isinstance(candidate, ProductClusterGraph):
                assign = list(candidate.entity_assignments)
                if proposal.axis == 'row':
                    assign[seed_a] = (child1, assign[seed_a][1])
                    assign[seed_b] = (child2, assign[seed_b][1])
                else:
                    assign[seed_a] = (assign[seed_a][0], child1)
                    assign[seed_b] = (assign[seed_b][0], child2)
                candidate.update_entity_assignments(assign)
            else:
                assign = candidate.entity_assignments.copy()
                assign[seed_a] = child1
                assign[seed_b] = child2
                candidate.update_entity_assignments(assign)

            best_score, opt_params = find_score(candidate, form, data, is_similarity, speed='fast')
            candidate.metadata.setdefault("opt_params", {}).update(opt_params)

            others = [m for m in proposal.parent_members if m not in (seed_a, seed_b)]
            random.shuffle(others)
            for o in others:
                # try put o in child1
                cand1 = candidate.copy()
                a1 = cand1.entity_assignments.copy()
                if isinstance(candidate, ProductClusterGraph):
                    if proposal.axis == 'row': a1[o] = (child1, a1[o][1])
                    else: a1[o] = (a1[o][0], child1)
                else: a1[o] = child1
                cand1.update_entity_assignments(a1)
                score1, opt_params = find_score(cand1, form, data, is_similarity, speed='fast')
                cand1.metadata.setdefault("opt_params", {}).update(opt_params)

                # try put o in child2
                cand2 = candidate.copy()
                a2 = cand2.entity_assignments.copy()
                if isinstance(candidate, ProductClusterGraph):
                    if proposal.axis == 'row': a2[o] = (child2, a2[o][1])
                    else: a2[o] = (a2[o][0], child2)
                else: a2[o] = child2
                cand2.update_entity_assignments(a2)
                score2, opt_params = find_score(cand2, form, data, is_similarity, speed='fast')
                cand2.metadata.setdefault("opt_params", {}).update(opt_params)

                # choose better
                if score1 >= score2:
                    candidate = cand1
                    best_score = score1
                else:
                    candidate = cand2
                    best_score = score2

            candidate_graphs.append((candidate, best_score))

    candidate_graphs.sort(key=lambda x: (round(x[1], 8), np.random.random()), reverse=True)  # higher score first

    def split_survives(g) -> bool:
        """Return True iff, after simplification, the original parent's members
        occupy >1 node along the split axis."""
        # pull assignments for the original parent members
        if isinstance(g, ProductClusterGraph):
            # look only along the split axis
            rows_seen, cols_seen = set(), set()
            for i in proposal.parent_members:
                a = g.entity_assignments[i]
                r, c = a
                if proposal.axis == 'row':
                    rows_seen.add(r)
                elif proposal.axis == 'col':
                    cols_seen.add(c)
            if proposal.axis == 'row':
                return len(rows_seen) > 1
            elif proposal.axis == 'col':
                return len(cols_seen) > 1
        else:
            # non-product: >1 unique latent id among the parent's members
            vals = [g.entity_assignments[i] for i in proposal.parent_members]
            return len(set(vals)) > 1

    # iterate best-to-worst candidate; accept first whose split survives simplification
    # if simplification collapses the split (all parent members end up in the same node),
    # treat that candidate as invalid and keep going down the list.
    # keep only candidates whose split survives simplification
    valid = []
    for cand, _ in candidate_graphs:
        simple = simplify_graph(cand)
        if split_survives(simple):
            valid.append(cand)

    if not valid:
        logger.info("All candidate splits collapsed after simplify_graph; keeping current structure.")
        return current, current_score
    
    # optimise the top 1 + near_k graphs and return the best one
    finalists = valid[: max(1, min(1 + near_k, len(valid)))]
    best_graph = finalists[0]
    best_score, opt_params = find_score(finalists[0], form, data, is_similarity, speed='slow')
    for i, g in enumerate(finalists[1:]):
        s, opt_params = find_score(g, form, data, is_similarity, speed='slow')
        if s > best_score:
            logger.info("Graph # %d after splitting yielded a higher optimised score.", i+2)
            best_graph, best_score = g, s
    best_graph.metadata.setdefault("opt_params", {}).update(opt_params)

    return best_graph, best_score