

import numpy as np
from typing import NamedTuple, List, Tuple, Dict, Set, Optional, Any, Iterable
import copy
from itertools import combinations


class ClusterGraph:
    """
    Represents the topology of a graph, with entities associated with cluster nodes.

    entity_assignments map each entity to a latent cluster ID (or -1 if unassigned/orphaned).
    latent_adjacency represents the topology among latent clusters (undirected).
    """
    def __init__(self, entity_assignments: np.ndarray, 
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 metadata: Optional[Dict] = None):
        self.entity_assignments = entity_assignments.copy().astype(int)
        all_ids = set(np.unique(self.entity_assignments.tolist()))
        self.latent_ids = {lid for lid in all_ids if lid != -1}
        if latent_adjacency is None:
            # initialise empty adjacency for existing latent clusters
            self.latent_adjacency = {lid: set() for lid in self.latent_ids}
        else:
            # copy and ensure symmetry
            self.latent_adjacency = {k: set(v) for k, v in latent_adjacency.items()}
            for a, neighs in list(self.latent_adjacency.items()):
                for b in list(neighs):
                    self.latent_adjacency.setdefault(b, set()).add(a)
            # ensure every latent in assignment appears
            for lid in self.latent_ids:
                self.latent_adjacency.setdefault(lid, set())
            self.latent_ids.update(self.latent_adjacency.keys())

        self.metadata = {} if metadata is None else copy.deepcopy(metadata)
        self.normalise()

        # maps: local entity idx -> global entity id; local latent id -> global cluster id
        self.latent_to_global_id: Dict[str, Dict[int, int]] = {
            "entities": {},  # {local_entity_index -> global_entity_id}
            "clusters": {},  # {local_latent_id   -> global_cluster_id}
        }
        # allocator for NEW cluster gids (monotonic)
        self.next_cluster_gid: int = 0

        self.init_global_maps()

    def normalise(self):
        # enforce symmetry
        for a, neighs in list(self.latent_adjacency.items()):
            for b in list(neighs):
                self.latent_adjacency.setdefault(b, set()).add(a)
    
    def init_global_maps(self, entity_global_ids: Optional[Dict[int,int]] = None):
        # entities: set once; default identity
        if not self.latent_to_global_id["entities"]:
            if entity_global_ids is None:
                for i in range(self.n_entities):
                    self.latent_to_global_id["entities"][i] = i
            else:
                self.latent_to_global_id["entities"].update(
                    {int(k): int(v) for k, v in entity_global_ids.items()}
                )
        # init allocator after max entity gid
        if self.next_cluster_gid == 0:
            e_vals = list(self.latent_to_global_id["entities"].values())
            self.next_cluster_gid = (max(e_vals) + 1) if e_vals else 0
        # ensure all CURRENT latents have a stable cluster gid
        for lid in self.latent_ids:
            if lid not in self.latent_to_global_id["clusters"]:
                self.latent_to_global_id["clusters"][lid] = self.next_cluster_gid
                self.next_cluster_gid += 1

    @property
    def latent_nodes(self):
        return set(self.latent_ids)

    @property
    def n_entities(self):
        return len(self.entity_assignments)

    @property
    def n_latents(self):
        return len(self.latent_nodes)

    def copy(self):
        g = ClusterGraph(self.entity_assignments.copy(),
                         {k: v for k, v in self.latent_adjacency.items()},
                         metadata=copy.deepcopy(self.metadata))
        # carry self properties
        g.latent_to_global_id = {
            "entities": dict(self.latent_to_global_id["entities"]),
            "clusters": dict(self.latent_to_global_id["clusters"]),
        }
        g.next_cluster_gid = int(self.next_cluster_gid)
        return g
    
    def __str__(self):
        return (
            f"ClusterGraph(\n"
            f"  latent_nodes={sorted(self.latent_ids)},\n"
            f"  latent to global mapping={self.latent_to_global_id},\n"
            f"  entity assignments={self.entity_assignments.tolist()},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in self.latent_adjacency.items())}}},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )
    
    def entity_counts(self) -> Dict[int, int]:
        """Count entities per latent id (ignores -1)."""
        counts: Dict[int, int] = {}
        for lid in self.entity_assignments:
            if lid == -1:
                continue
            counts[lid] = counts.get(lid, 0) + 1
        return counts

    def add_latents(self, n: int) -> List[int]:
        """
        Add `n` new latent nodes, choosing integer IDs not already present.
        Returns the list of new latent IDs.
        """
        if n <= 0:
            return []
        existing = set(self.latent_nodes)
        # pick next available integers (simple strategy)
        next_id = max(existing) + 1
        new_ids = []
        for i in range(n):
            lid = next_id + i
            self.latent_adjacency.setdefault(lid, set())
            new_ids.append(lid)
        self.latent_ids.update(new_ids)

        for lid in new_ids:
            self.latent_to_global_id["clusters"][lid] = self.next_cluster_gid
            self.next_cluster_gid += 1

        self.normalise()
        return new_ids

    def remove_latents(self, latent_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Remove specified latent nodes from the topology.
        Entities assigned to removed latents become orphaned (assignment set to -1).
        Returns mapping: removed latent -> array of orphaned entity indices that had been assigned to it.
        """
        orphaned: Dict[int, np.ndarray] = {}
        latent_ids = list(latent_ids)
        for lid in latent_ids:
            # find entities assigned to this latent before removal
            mask = self.entity_assignments == lid
            orphaned_entities = np.where(mask)[0]
            if orphaned_entities.size > 0:
                orphaned[lid] = orphaned_entities.copy()
                # mark them as orphaned
                self.entity_assignments[mask] = -1
            # remove node
            self.latent_ids.remove(lid)
            for neigh in list(self.latent_adjacency[lid]):
                self.latent_adjacency[neigh].discard(lid)
            self.latent_adjacency.pop(lid, None)
        self.normalise()
        return orphaned

    def add_edges(self, edges: Iterable[Tuple[int, int]]):
        """
        Add undirected latent-latent edges. Adds endpoints to adjacency if missing.
        """
        for a, b in edges:
            if a == b:
                raise ValueError(f"Cannot add self-loop on latent {a}.")
            if a not in self.latent_ids or b not in self.latent_ids:
                raise ValueError(f"Edge ({a}, {b}) references unknown latent id(s). "
                                f"Known: {sorted(self.latent_ids)}")
            self.latent_adjacency.setdefault(a, set()).add(b)
            self.latent_adjacency.setdefault(b, set()).add(a)
        self.normalise()

    def remove_edges(self, edges: Iterable[Tuple[int, int]]):
        """
        Remove undirected latent-latent edges (if present).
        """
        for a, b in edges:
            if a in self.latent_adjacency:
                self.latent_adjacency[a].discard(b)
            if b in self.latent_adjacency:
                self.latent_adjacency[b].discard(a)
        self.normalise()

    def update_entity_assignments(self, new_assignments: np.ndarray):
        """
        Replace the entity assignments.
        """
        self.entity_assignments = new_assignments.copy().astype(int)


class ChainRingClusterGraph(ClusterGraph):
    # chain / ring forms keep an order of nodes, with latent edges between adjacent latent nodes
    def __init__(self, entity_assignments: np.ndarray, 
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 order: List[int] = None,
                 metadata: Optional[Dict] = None):
        super().__init__(entity_assignments, latent_adjacency, metadata)
        self._order = order if order else [0]

    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, new_order):
        self._order = new_order

    def remove_latents(self, latent_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Overrides the base method to also update the order list.
        """
        ids_to_remove = set(latent_ids)
        orphaned = super().remove_latents(ids_to_remove)
        self._order = [lid for lid in self._order if lid not in ids_to_remove] 
        return orphaned
    
    def copy(self):
        g = ChainRingClusterGraph(self.entity_assignments.copy(),
                            {k: v for k, v in self.latent_adjacency.items()},
                            self._order.copy(),
                            metadata=copy.deepcopy(self.metadata))
        # carry self properties
        g.latent_to_global_id = {
            "entities": dict(self.latent_to_global_id["entities"]),
            "clusters": dict(self.latent_to_global_id["clusters"]),
        }
        g.next_cluster_gid = int(self.next_cluster_gid)
        return g
    
    def __str__(self):
        return (
            f"ChainRingClusterGraph(\n"
            f"  latent_nodes={sorted(self.latent_ids)},\n"
            f"  order={self.order},\n"
            f"  latent to global mapping={self.latent_to_global_id},\n"
            f"  entity assignments={self.entity_assignments.tolist()},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in self.latent_adjacency.items())}}},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )
    

class HierarchyClusterGraph(ClusterGraph):
    """
    Represents a hierarchical cluster graph structure.
    The hierarchy is defined by parent-child relationships between clusters.
    """
    def __init__(self, entity_assignments: np.ndarray,
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 parent: Optional[Dict[int, Optional[int]]] = None,
                 children: Optional[Dict[int, Set[int]]] = None,
                 metadata: Optional[Dict] = None):
        """
        Initializes the HierarchyClusterGraph.
        Args:
            entity_assignments: Maps each entity to a cluster ID.
            latent_adjacency: Adjacency list for the cluster graph.
            parent: A dictionary mapping each child cluster ID to its parent's ID.
                    The root node's parent should be None.
            metadata: Additional metadata.
        """
        super().__init__(entity_assignments, latent_adjacency, metadata)
        if parent is None and children is None:
            self._parent: Dict[int, Optional[int]] = {0: None}
            self._children: Dict[int, Set[int]] = {0: set()}
        else:
            self._parent = {k: v for k, v in parent.items()}
            self._children = {k: set(v) for k, v in children.items()}

        # ensure every latent has entries
        for lid in self.latent_nodes:
            self._parent.setdefault(lid, None if lid not in self._parent else self._parent[lid])
            self._children.setdefault(lid, set())

    def set_parent_of(self, child: int, parent: Optional[int]):
        old_parent = self._parent.get(child, None)
        if old_parent is not None:
            self._children.setdefault(old_parent, set()).discard(child)
        self._parent[child] = parent
        if parent is not None:
            self._children.setdefault(parent, set()).add(child)

    def add_child(self, parent: int, child: int):
        self.set_parent_of(child, parent)

    @property
    def parent(self) -> Dict[int, Optional[int]]:
        """
        A dictionary mapping each child cluster ID to its parent cluster ID.
        The root node maps to None.
        """
        return self._parent

    @parent.setter
    def parent(self, new_parent_map: Dict[int, Optional[int]]):
        self._parent = new_parent_map

    @property
    def children(self) -> Dict[int, Set[int]]:
        return self._children

    def children_of(self, node: int) -> Set[int]:
        return set(self._children.get(node, set()))

    @property
    def leaf_nodes(self) -> List[int]:
        """
        Returns a list of cluster IDs that are leaves in the graph.
        A leaf node is defined as a cluster with a degree of 1 or less.
        """
        leaves = []
        for lid in self.latent_nodes:
            degree = len(self.latent_adjacency.get(lid, set()))
            if degree <= 1:
                leaves.append(lid)
        return sorted(leaves)
    
    def remove_latents(self, latent_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Overrides the base method to also update the parent & children maps.
        Children of removed nodes become new roots (parent=None).
        """
        ids_to_remove = set(latent_ids)

        # update parent/children before topology removal
        for rid in ids_to_remove:
            # detach from its parent
            par = self._parent.pop(rid, None)
            if par is not None:
                self._children.setdefault(par, set()).discard(rid)
            # re-root its children
            chs = self._children.pop(rid, set())
            for c in chs:
                self._parent[c] = None

        orphaned = super().remove_latents(ids_to_remove)

        # ensure lookups have entries for survivors
        for lid in self.latent_nodes:
            self._parent.setdefault(lid, self._parent.get(lid, None))
            self._children.setdefault(lid, set())

        return orphaned

    def copy(self):
        """Creates a deep copy of the HierarchyClusterGraph."""
        g = HierarchyClusterGraph(
            self.entity_assignments.copy(),
            {k: v.copy() for k, v in self.latent_adjacency.items()},
            parent=copy.deepcopy(self._parent),
            children={k: set(v) for k, v in self._children.items()},
            metadata=copy.deepcopy(self.metadata)
        )
        # carry self properties
        g.latent_to_global_id = {
            "entities": dict(self.latent_to_global_id["entities"]),
            "clusters": dict(self.latent_to_global_id["clusters"]),
        }
        g.next_cluster_gid = int(self.next_cluster_gid)
        return g

    def __str__(self):
        """Returns a string representation of the HierarchyClusterGraph."""
        parent_str = '{' + ', '.join(f'{k}:{v}' for k, v in sorted(self.parent.items())) + '}'
        children_str = '{' + ', '.join(f'{k}:{sorted(list(v))}' for k, v in sorted(self.children.items())) + '}'
        return (
            f"HierarchyClusterGraph(\n"
            f"  latent_nodes={sorted(self.latent_ids)},\n"
            f"  latent to global mapping={self.latent_to_global_id},\n"
            f"  parent={parent_str},\n"
            f"  children={children_str},\n"
            f"  leaf_nodes={self.leaf_nodes},\n"
            f"  entity assignments={self.entity_assignments.tolist()},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in sorted(self.latent_adjacency.items()))}}},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )


class TreeClusterGraph(ClusterGraph):
    # tree forms keep leaf vs internal nodes seperate (as internals cannot be connected to an entity)
    # update_entity_assignments is also changed to remove nodes from latent_ids which are no longer 
    # connected to an entity (they are still within this structure)
    def __init__(self, entity_assignments: np.ndarray, 
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 leaf_ids = None,
                 internal_ids = None,
                 parent: Optional[Dict[int, Optional[int]]] = None,
                 children: Optional[Dict[int, Set[int]]] = None,
                 metadata: Optional[Dict] = None):
        super().__init__(entity_assignments, latent_adjacency, metadata)
        self.leaf_ids = leaf_ids if leaf_ids else set([0])
        self.internal_ids = internal_ids if internal_ids else set()

        if parent is None and children is None:
            self._parent: Dict[int, Optional[int]] = {0: None}
            self._children: Dict[int, Set[int]] = {0: set()}
        else:
            self._parent = {k: v for k, v in parent.items()}
            self._children = {k: set(v) for k, v in children.items()}

        # ensure every latent has entries
        for lid in self.latent_nodes:
            self._parent.setdefault(lid, None if lid not in self._parent else self._parent[lid])
            self._children.setdefault(lid, set())

    @property
    def leaf_nodes(self) -> Set[int]:
        return set(self.leaf_ids)
        
    @property
    def internal_nodes(self) -> Set[int]:
        return set(self.internal_ids)
    
    @leaf_nodes.setter
    def leaf_nodes(self, new_leaf_nodes):
        self.leaf_ids = set(new_leaf_nodes)

    @internal_nodes.setter
    def internal_nodes(self, new_internal_nodes):
        self.internal_ids = set(new_internal_nodes)
    
    @property
    def parent(self) -> Dict[int, Optional[int]]:
        return self._parent

    @property
    def children(self) -> Dict[int, Set[int]]:
        return self._children

    def children_of(self, node: int) -> Set[int]:
        return set(self._children.get(node, set()))

    def _reclassify(self, node: int):
        """Make sure leaf/internal sets match the children map for `node`."""
        if len(self._children.get(node, set())) == 0:
            self.internal_ids.discard(node)
            self.leaf_ids.add(node)
        else:
            self.leaf_ids.discard(node)
            self.internal_ids.add(node)

    def set_parent_of(self, child: int, parent: Optional[int]):
        old_parent = self._parent.get(child, None)
        if old_parent is not None:
            self._children.setdefault(old_parent, set()).discard(child)
            self._reclassify(old_parent)  # may have become a leaf

        self._parent[child] = parent
        if parent is not None:
            self._children.setdefault(parent, set()).add(child)
            self._reclassify(parent)      # definitely an internal now

    def remove_latents(self, latent_ids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Also update leaf/internal sets + parent/children maps.
        Children of removed nodes become new roots (parent=None).
        """
        ids_to_remove = set(latent_ids)

        # update hierarchical maps
        for rid in ids_to_remove:
            par = self._parent.pop(rid, None)
            if par is not None:
                self._children.setdefault(par, set()).discard(rid)
                self._reclassify(par)
            chs = self._children.pop(rid, set())
            for c in chs:
                self._parent[c] = None  # re-root children

        orphaned = super().remove_latents(ids_to_remove)

        if self.leaf_ids:
            self.leaf_ids.difference_update(ids_to_remove)
        if self.internal_ids:
            self.internal_ids.difference_update(ids_to_remove)

        # ensure entries exist for survivors
        for lid in self.latent_nodes:
            self._parent.setdefault(lid, self._parent.get(lid, None))
            self._children.setdefault(lid, set())

        return orphaned
    
    def copy(self):
        g = TreeClusterGraph(
            self.entity_assignments.copy(),
            {k: v for k, v in self.latent_adjacency.items()},
            self.leaf_ids.copy(),
            self.internal_ids.copy(),
            parent=copy.deepcopy(self._parent),
            children={k: set(v) for k, v in self._children.items()},
            metadata=copy.deepcopy(self.metadata)
        )
        # carry self properties
        g.latent_to_global_id = {
            "entities": dict(self.latent_to_global_id["entities"]),
            "clusters": dict(self.latent_to_global_id["clusters"]),
        }
        g.next_cluster_gid = int(self.next_cluster_gid)
        return g
    
    def __str__(self):
        return (
            f"TreeClusterGraph(\n"
            f"  entity assignments={self.entity_assignments.tolist()},\n"
            f"  latent to global mapping={self.latent_to_global_id},\n"
            f"  latent_nodes={sorted(self.latent_ids)},\n"
            f"  internal_ids={self.internal_ids},\n"
            f"  leaf_ids={self.leaf_ids},\n"
            f"  parent={{{', '.join(f'{k}:{v}' for k,v in sorted(self._parent.items()))}}},\n"
            f"  children={{{', '.join(f'{k}:{sorted(list(v))}' for k,v in sorted(self._children.items()))}}},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in self.latent_adjacency.items())}}},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )

    def update_entity_assignments(self, new_assignments: np.ndarray):
        """
        keep all existing latents (internal + leaf) and only update assignments.
        """
        self.entity_assignments = new_assignments.copy().astype(int)


class ProductClusterGraph:
    """
    ProductClusterGraph supports products of two 1D component graphs (e.g., Chain×Chain for grids,
    Chain×Ring for cylinders). It stores separate assignments per axis.
    """
    def __init__(self,
                 row_graph: ChainRingClusterGraph,
                 col_graph: ChainRingClusterGraph,
                 row_assignments: np.ndarray,
                 col_assignments: np.ndarray,
                 metadata: Optional[Dict] = None,
                 entity_global_ids: Optional[Dict[int, int]] = None):
        # must have same number of entities
        if len(row_assignments) != len(col_assignments):
            raise ValueError("row and column assignments must align in length")
        self.row_graph = row_graph
        self.col_graph = col_graph
        self.row_assignments = row_assignments.copy().astype(int)
        self.col_assignments = col_assignments.copy().astype(int)

        # build mapping from row ID or col ID to global id
        self._row_gid: Dict[int,int] = {}
        self._col_gid: Dict[int,int] = {}
        gid = 0
        for r in self.row_graph.order:
            self._row_gid[r] = gid
            gid += 1
        for c in self.col_graph.order:
            self._col_gid[c] = gid
            gid += 1
        # reverse maps: global -> local
        self._gid_to_row: Dict[int,int] = {g: r for r, g in self._row_gid.items()}
        self._gid_to_col: Dict[int,int] = {g: c for c, g in self._col_gid.items()}

        self.metadata = {} if metadata is None else copy.deepcopy(metadata)
        self.entity_global_ids: Dict[int, int] = (
            {i: i for i in range(self.n_entities)}
            if entity_global_ids is None else {int(k): int(v) for k, v in entity_global_ids.items()}
        )
        self.cell_gid_map: Dict[tuple, int] = {}
        self.next_cell_gid: int = self.n_entities+1

    @property
    def n_entities(self) -> int:
        return len(self.row_assignments)

    @property
    def entity_assignments(self) -> List[Tuple[int,int]]:
        # return (row, col) tuple for each entity
        return list(zip(self.row_assignments.tolist(), self.col_assignments.tolist()))

    def update_entity_assignments(self, new_assignments: List[Tuple[int,int]]):
        # new_assignments is a list of (row_id, col_id)
        for i, (r, c) in enumerate(new_assignments):
            self.row_assignments[i] = r
            self.col_assignments[i] = c
        self.row_graph.entity_assignments = self.row_assignments
        self.col_graph.entity_assignments = self.col_assignments

    @property
    def latent_nodes(self) -> set:
        return set(self.row_global_nodes + self.col_global_nodes)

    @property
    def row_global_nodes(self) -> List[int]:
        return [self._row_gid[r] for r in self.row_graph.order]

    @property
    def col_global_nodes(self) -> List[int]:
        return [self._col_gid[c] for c in self.col_graph.order]

    def global_to_local(self, gid: int) -> Tuple[str,int]:
        """
        Map a global ID back to ('row' or 'col', local chain ID).
        """
        if gid in self._gid_to_row:
            return 'row', self._gid_to_row[gid]
        if gid in self._gid_to_col:
            return 'col', self._gid_to_col[gid]
        raise KeyError(f"Global id {gid} not in grid")
    
    def form_cartesian_product(self):
        rows = self.row_graph.order # in local coords
        cols = self.col_graph.order

        row_gid_map = self.row_graph.latent_to_global_id["clusters"]  # local row id -> stable gid
        col_gid_map = self.col_graph.latent_to_global_id["clusters"]  # local col id -> stable gid

        # assign a stable cell_gid for every current (row_gid, col_gid)
        current_pairs = []
        for r in rows:
            for c in cols:
                rg = row_gid_map[r]
                cg = col_gid_map[c]
                key = (rg, cg)
                # reuse existing id or allocate a new, never-reused one
                if key not in self.cell_gid_map:
                    self.cell_gid_map[key] = self.next_cell_gid
                    self.next_cell_gid += 1
                current_pairs.append((r, c, key, self.cell_gid_map[key]))

        # give compact local lids in a stable order (by cell_gid)
        current_pairs.sort(key=lambda t: t[3])  # sort by cell_gid
        cell_gid_to_local = {cell_gid: idx for idx, (_, _, _, cell_gid) in enumerate(current_pairs)}
        # local lid lookup from local (r,c)
        cell_local_from_rc = {(r, c): cell_gid_to_local[cell_gid] for (r, c, _, cell_gid) in current_pairs}

        # build entity assignments over local lids
        new_assign = np.full(self.n_entities, -1, dtype=int)
        for i in range(self.n_entities):
            r, c = self.row_assignments[i], self.col_assignments[i]
            if r != -1 and c != -1:
                new_assign[i] = cell_local_from_rc[(r, c)]

        # build full adjacency over the grid cells
        n_cells = len(current_pairs)
        adj: Dict[int, set[int]] = {lid: set() for lid in range(n_cells)}
        # along rows
        for r, neighs in self.row_graph.latent_adjacency.items():
            for r2 in neighs:
                for c in cols:
                    a = cell_local_from_rc[(r, c)]
                    b = cell_local_from_rc[(r2, c)]
                    adj[a].add(b); adj[b].add(a)
        # along cols
        for c, neighs in self.col_graph.latent_adjacency.items():
            for c2 in neighs:
                for r in rows:
                    a = cell_local_from_rc[(r, c)]
                    b = cell_local_from_rc[(r, c2)]
                    adj[a].add(b); adj[b].add(a)

        # create ClusterGraph and attach global IDs for entities and cells
        cg = ClusterGraph(new_assign, latent_adjacency=adj, metadata=copy.deepcopy(self.metadata))
        # entities keep global ids
        cg.latent_to_global_id["entities"] = {i: int(self.entity_global_ids[i]) for i in range(self.n_entities)}
        # product cells: global id is a unique token per cell_gid (never reused)
        for (r, c, key, cell_gid) in current_pairs:
            local_lid = cell_gid_to_local[cell_gid]
            cg.latent_to_global_id["clusters"][local_lid] = int(cell_gid)  # stable unique id

        # sync internal invariants (fills any missing bits; allocator not used here)
        cg.init_global_maps()
        return cg
    
    def copy(self) -> 'ProductClusterGraph':
        g = ProductClusterGraph(
            self.row_graph.copy(),
            self.col_graph.copy(),
            self.row_assignments.copy(),
            self.col_assignments.copy(),
            copy.deepcopy(self.metadata),
            entity_global_ids=dict(self.entity_global_ids),
        )
        g.cell_gid_map = dict(self.cell_gid_map)
        g.next_cell_gid = int(self.next_cell_gid)
        return g

    def __str__(self):
        full_product_graph = self.form_cartesian_product()
        full_adj_str = '{' + ', '.join(f'{k}:{sorted(list(v))}' for k, v in sorted(full_product_graph.latent_adjacency.items())) + '}'

        return (
            f"ProductClusterGraph(\n"
            # Component graph info
            f"  -- Component Graphs --\n"
            f"  row_local_ids={self.row_graph.order},\n"
            f"  col_local_ids={self.col_graph.order},\n"
            f"  row_assignments={self.row_assignments.tolist()},\n"
            f"  col_assignments={self.col_assignments.tolist()},\n"
            f"  row_graph_adj={{{', '.join(f'{k}:{sorted(v)}' for k,v in sorted(self.row_graph.latent_adjacency.items()))}}},\n"
            f"  col_graph_adj={{{', '.join(f'{k}:{sorted(v)}' for k,v in sorted(self.col_graph.latent_adjacency.items()))}}},\n"
            # Full Cartesian product info
            f"  -- Full Cartesian Product --\n"
            f"  product_latent_nodes={sorted(full_product_graph.latent_ids)},\n"
            f"  product_entity_assignments={full_product_graph.entity_assignments.tolist()},\n"
            f"  product_latent_adjacency={full_adj_str},\n"
            f"  latent to global mapping={full_product_graph.latent_to_global_id},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )
        

def edge_order(u, v):  # smallest first
    return (u, v) if u <= v else (v, u)

class EntityGraph:
    def __init__(self, n_entities: int):
        self.n_entities = n_entities
        self.int_edges = []
        self.ext_edges = []
        self.all_edges = [] # local (i, j)
        self.all_edges_global = [] # aligned canonical (gid_i, gid_j)
        self.n_latent = 0
        self.n_nodes = n_entities # will be updated
        self.cluster_to_latent = {}  # mapping from cluster id to global node index
        self.gid_of_local = None         # length n_nodes; local->global
        self.local_of_gid = None         # dict global->local

    def add_edge(self, i: int, j: int, type: str):
        if type == 'external':
            self.ext_edges.append((i, j))
        elif type == 'internal':
            self.int_edges.append((i, j))
        self.all_edges.append((i, j))

    def finalise(self, n_latent: int, gid_of_local: List[int]):
        self.n_latent = n_latent
        self.n_nodes = self.n_entities + self.n_latent
        # map local<->global for this *current* graph
        self.gid_of_local = list(gid_of_local)
        self.local_of_gid = {g: i for i, g in enumerate(self.gid_of_local)}
        # build aligned global edge list
        self.all_edges_global = [
            edge_order(self.gid_of_local[i], self.gid_of_local[j])
            for (i, j) in self.all_edges
        ]


class SplitProposal(NamedTuple):
    split_graph: ClusterGraph        # graph with topology split but no reassignments
    parent_members: np.ndarray     # the indices of entities in that parent
    children: Tuple[int, int]   # the two IDs of the new children (in local co ords for the joint Forms)
    axis: str # the axis 'row' or 'col' for a split in a GridForm


class StructuralForm:
    name = "base"

    def initial_structure(self, n_entities: int) -> ClusterGraph:
        # single latent cluster containing all entities, no latent-latent edges
        return ClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: ClusterGraph, cluster_id: int) -> Tuple[ClusterGraph, np.ndarray, List[int]]:
        """
        Partition-like split. Remove parent cluster_id, add two children inheriting its neighbors.
        Does not assign any entities to the new children; returns:
          - new topology ClusterGraph (with parent removed, children present, entities orphaned)
          - array of entity indices that were in the parent (to be distributed)
          - list of new child latent IDs (length 2)
        """
        cg = cluster_graph.copy()
        new_children = cg.add_latents(2)
        orphaned = cg.remove_latents([cluster_id])
        parent_members = orphaned.get(cluster_id, np.array([], dtype=int))
        return cg, parent_members, new_children
    
    def propose_splits(self, G: ClusterGraph):
        proposals = []
        for lid in G.latent_nodes:
            new_G, members, [c1,c2] = self.split_cluster(G, lid)
            proposals.append(SplitProposal(new_G, members, (c1,c2), None))
        return proposals
    

class PartitionForm(StructuralForm):
    name = "partition"


class OrderForm(StructuralForm):
    """
    all latent nodes connect to all others.
    """
    name = "connected"

    def split_cluster(self, cluster_graph: ClusterGraph, cluster_id: int) -> Tuple[ClusterGraph, np.ndarray, List[int]]:
        cg, parent_members, new_children = super().split_cluster(cluster_graph, cluster_id)
        # fully connect all latent nodes
        nodes = list(cg.latent_nodes)
        for u, v in combinations(nodes, 2):
            cg.add_edges([(u, v)])
        return cg, parent_members, new_children
    

class ChainForm(StructuralForm):
    name = "chain"

    def initial_structure(self, n_entities: int) -> ChainRingClusterGraph:
        return ChainRingClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: ChainRingClusterGraph, cluster_id: int) -> Tuple[ChainRingClusterGraph, np.ndarray, List[int]]:
        """
        Replace cluster_id with two children, update chain order so they are adjacent.
        """
        cg = cluster_graph.copy()
        order = cg.order
        idx = order.index(cluster_id)

        left_neighbour = order[idx - 1] if idx > 0 else None
        right_neighbour = order[idx + 1] if idx < len(order) - 1 else None

        # add two new child latent nodes
        new_children = cg.add_latents(2)
        child_a, child_b = new_children

        # remove the parent cluster, orphaning its entities
        orphaned = cg.remove_latents([cluster_id])
        parent_members = orphaned.get(cluster_id, np.array([], dtype=int))

        # add edges, firstly child_a--child_b
        edges: List[Tuple[int, int]] = [(child_a, child_b)]
        # connect child_a to left neighbour
        if left_neighbour is not None:
            edges.append((child_a, left_neighbour))
        # connect child_b to right neighbour
        if right_neighbour is not None:
            edges.append((child_b, right_neighbour))

        cg.add_edges(edges)

        # update the chain order: replace parent with child_a, child_b
        new_order = order[:idx] + [child_a, child_b] + order[idx + 1 :]
        cg.order = new_order

        return cg, parent_members, new_children
    

class RingForm(StructuralForm):
    name = "ring"

    def initial_structure(self, n_entities: int) -> ChainRingClusterGraph:
        return ChainRingClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: ChainRingClusterGraph, cluster_id: int) -> Tuple[ChainRingClusterGraph, np.ndarray, List[int]]:
        """
        Replace cluster_id with two children, update chain order so they are adjacent.
        """
        cg = cluster_graph.copy()
        order = cg.order
        idx = order.index(cluster_id)

        left_neighbour = order[idx - 1] if idx > 0 else order[-1] if len(order) > 1 else None
        right_neighbour = order[idx + 1] if idx < len(order) - 1 else order[0] if len(order) > 1 else None

        # add two new child latent nodes
        new_children = cg.add_latents(2)
        child_a, child_b = new_children

        # remove the parent cluster, orphaning its entities
        orphaned = cg.remove_latents([cluster_id])
        parent_members = orphaned.get(cluster_id, np.array([], dtype=int))

        # add edges, firstly child_a--child_b
        edges: List[Tuple[int, int]] = [(child_a, child_b)]
        # connect child_a to left neighbour
        if left_neighbour is not None:
            edges.append((child_a, left_neighbour))
        # connect child_b to right neighbour
        if right_neighbour is not None:
            edges.append((child_b, right_neighbour))

        cg.add_edges(edges)

        # update the chain order: replace parent with child_a, child_b
        new_order = order[:idx] + [child_a, child_b] + order[idx + 1 :]
        cg.order = new_order

        # ensure wrap-around edge between first and last
        if len(new_order) > 1:
            cg.add_edges([(new_order[0], new_order[-1])])

        return cg, parent_members, new_children
    

class HierarchyForm(StructuralForm):
    """
    Hierarchy structural form: maintains a tree hierarchy of latent clusters.
    On split, one child replaces the parent (inheriting its connections),
    while the other becomes a new leaf attached only to that child.
    The tree is tracked via metadata['parent'], mapping each latent ID to
    its parent ID (or None for the root).
    """
    name = "hierarchy"

    def initial_structure(self, n_entities: int) -> HierarchyClusterGraph:
        return HierarchyClusterGraph(np.zeros(n_entities, dtype=int))
    
    def split_cluster(self, cluster_graph: HierarchyClusterGraph, cluster_id: int) -> Tuple[HierarchyClusterGraph, np.ndarray, List[int]]:
        cg = cluster_graph.copy()

        # parent of the cluster to split
        parent_of_cluster = cg.parent.get(cluster_id)
        # neighbours in latent adjacency
        parent_neighbours = set(cg.latent_adjacency.get(cluster_id, set()))
        children_of_cluster = set(cg.children_of(cluster_id))

        # add two new child latent nodes
        new_children = cg.add_latents(2)
        child_a, child_b = new_children

        # remove the parent cluster, orphaning its entities
        orphaned = cg.remove_latents([cluster_id])
        parent_members = orphaned.get(cluster_id, np.array([], dtype=int))

        # edges: child_a <-> child_b, and child_b inherits former neighbours
        edges: List[Tuple[int, int]] = [(child_a, child_b)]
        for neigh in parent_neighbours:
            edges.append((child_b, neigh))
        cg.add_edges(edges)

        # child_b replaces cluster_id in the hierarchy
        cg.set_parent_of(child_b, parent_of_cluster)
        # child_a becomes a new child of child_b
        cg.set_parent_of(child_a, child_b)
        # original children now attach under child_b
        for ch in children_of_cluster:
            cg.set_parent_of(ch, child_b)

        return cg, parent_members, new_children
    

class TreeForm(StructuralForm):
    """
    Tree structural form: maintains a binary tree where only leaves
    carry entity assignments. Internal nodes have no entities.
    Metadata tracks:
      - 'internal_ids': list of internal (split) node IDs
      - 'leaf_ids': list of current leaf node IDs

    Splitting is only applied to leaves; non-leaf splits return no-op.
    """
    name = "tree"

    def initial_structure(self, n_entities: int) -> TreeClusterGraph:
        return TreeClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: TreeClusterGraph, cluster_id: int) -> Tuple[TreeClusterGraph, np.ndarray, List[int]]:
        cg = cluster_graph.copy()
        internal_ids = list(cg.internal_nodes)
        leaf_ids = list(cg.leaf_nodes)

        if cluster_id not in leaf_ids:
            # can't split internal nodes
            return cg, np.array([], dtype=int), []

        # orphan entities from the leaf
        parent_members = np.where(cg.entity_assignments == cluster_id)[0]
        cg.entity_assignments[parent_members] = -1

        # add two new leaf children and attach to the internal node
        new_children = cg.add_latents(2)
        edges = [(cluster_id, new_children[0]), (cluster_id, new_children[1])]
        cg.add_edges(edges)

        cg.set_parent_of(new_children[0], cluster_id)
        cg.set_parent_of(new_children[1], cluster_id)

        # update metadata
        internal_ids.append(cluster_id)
        leaf_ids.remove(cluster_id)
        leaf_ids.extend(new_children)
        cg.internal_nodes = internal_ids
        cg.leaf_nodes = leaf_ids

        return cg, parent_members, new_children
    
    def propose_splits(self, G: TreeClusterGraph):
        proposals = []
        for lid in G.leaf_nodes:
            new_G, members, [c1,c2] = self.split_cluster(G, lid)
            proposals.append(SplitProposal(new_G, members, (c1,c2), None))
        return proposals
    

class ProductForm():
    """
    Grid form: latent clusters come from two independent chains (rows and columns).
    Search iterates over each global latent ID (row or col) as a split candidate.
    """
    def make_product(self,
                   original: ProductClusterGraph,
                   axis: str,
                   split_res: Tuple[ClusterGraph, np.ndarray, List[int]]
                   ) -> Tuple[ProductClusterGraph, np.ndarray, Tuple[int, int]]:
        """
        Lift a 1D split (chain split) back into a full ProductClusterGraph.

        Parameters:
        - split_res: (new_chain_graph, orphaned_entities, children_local_ids)
        - original: the original GridClusterGraph before split
        - axis: 'row' or 'col', indicating which chain was split

        Returns:
        - new_grid: the updated GridClusterGraph with one chain split
        - orphaned: array of entity indices that were orphaned
        - children_global: tuple of the two new global IDs for the children
        """
        new_chain, orphaned, children_local = split_res
        if axis == 'row':
            # update row assignments
            new_row_assign = original.row_assignments.copy()
            new_row_assign[orphaned] = -1
            new_grid = ProductClusterGraph(
                new_chain,                    # updated row graph
                original.col_graph,           # unchanged col graph
                new_row_assign,               # orphaned rows handled
                original.col_assignments,
                metadata=copy.deepcopy(original.metadata),
                entity_global_ids=dict(original.entity_global_ids)
            )
            # map local child IDs to global
            g1, g2 = (new_grid._row_gid[children_local[0]],
                      new_grid._row_gid[children_local[1]])
        else:
            new_col_assign = original.col_assignments.copy()
            new_col_assign[orphaned] = -1
            new_grid = ProductClusterGraph(
                original.row_graph,
                new_chain,
                original.row_assignments,
                new_col_assign,
                metadata=copy.deepcopy(original.metadata),
                entity_global_ids=dict(original.entity_global_ids)
            )
            g1, g2 = (new_grid._col_gid[children_local[0]],
                      new_grid._col_gid[children_local[1]])
        return new_grid, orphaned, (g1, g2)
    

class GridForm(ProductForm):
    """
    Grid form: product of a chain (rows) and a chain (columns).
    Uses ProductClusterGraph(Chain × Chain).
    """
    name = "grid"

    def initial_structure(self, n_entities: int) -> ProductClusterGraph:
        row_graph = ChainForm().initial_structure(n_entities)  # chain axis
        col_graph = ChainForm().initial_structure(n_entities)   # chain axis
        row_assign = np.zeros(n_entities, dtype=int)
        col_assign = np.zeros(n_entities, dtype=int)
        return ProductClusterGraph(row_graph, col_graph, row_assign, col_assign)

    def propose_splits(self, G: ProductClusterGraph):
        proposals = []
        # split along the chain axis (rows)
        for r in G.row_graph.latent_nodes:
            new_row_graph, members, [c1, c2] = ChainForm().split_cluster(G.row_graph, r)
            full, orphaned, (g1, g2) = self.make_product(G, 'row', split_res=(new_row_graph, members, (c1, c2)))
            proposals.append(SplitProposal(full, members, (c1, c2), 'row'))
        # split along the chain axis (columns)
        for c in G.col_graph.latent_nodes:
            new_col_graph, members, [c1, c2] = ChainForm().split_cluster(G.col_graph, c)
            full, orphaned, (g1, g2) = self.make_product(G, 'col', split_res=(new_col_graph, members, (c1, c2)))
            proposals.append(SplitProposal(full, members, (c1, c2), 'col'))
        return proposals


class CylinderForm(ProductForm):
    """
    Cylinder form: product of a chain (rows) and a ring (columns).
    Uses ProductClusterGraph(Chain × Ring).
    """
    name = "cylinder"

    def initial_structure(self, n_entities: int) -> ProductClusterGraph:
        row_graph = ChainForm().initial_structure(n_entities)  # chain axis
        col_graph = RingForm().initial_structure(n_entities)   # ring axis
        row_assign = np.zeros(n_entities, dtype=int)
        col_assign = np.zeros(n_entities, dtype=int)
        return ProductClusterGraph(row_graph, col_graph, row_assign, col_assign)

    def propose_splits(self, G: ProductClusterGraph):
        proposals = []
        # split along the chain axis (rows)
        for r in G.row_graph.latent_nodes:
            new_row_graph, members, [c1, c2] = ChainForm().split_cluster(G.row_graph, r)
            full, orphaned, (g1, g2) = self.make_product(G, 'row', split_res=(new_row_graph, members, (c1, c2)))
            proposals.append(SplitProposal(full, members, (c1, c2), 'row'))
        # split along the ring axis (columns)
        for c in G.col_graph.latent_nodes:
            new_col_graph, members, [c1, c2] = RingForm().split_cluster(G.col_graph, c)
            full, orphaned, (g1, g2) = self.make_product(G, 'col', split_res=(new_col_graph, members, (c1, c2)))
            proposals.append(SplitProposal(full, members, (c1, c2), 'col'))
        return proposals
