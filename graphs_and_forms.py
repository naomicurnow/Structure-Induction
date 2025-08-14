

import numpy as np
from typing import NamedTuple, List, Tuple, Dict, Set, Optional, Any, Iterable
import copy
from itertools import combinations


"""
ID & INDEXING GLOSSARY
-----------------------------------------------------
- idx  (EntityIdx)   : Local **array index** for an entity within the object (0..n_entities-1).
- lid  (LatentLID)   : Local **latent node label** within a graph instance's topology.
                       LIDs are ephemeral (may change after edits/copies).
- gid  (GlobalID)    : Stable **global identifier** that persists across copies/derivations.
                       There are several types of GID:
    * eid         : global entity id          (EntityGID)
    * cid         : global cluster id         (ClusterGID) for latent nodes in 1D graphs
    * rgid/cgid   : row/col cluster ids       (RowGID / ColGID) for ProductClusterGraph axes
    * cell_gid    : product-cell id           (CellGID) for cells (rgid, cgid) in a Cartesian product

Summary
-------------
- Anything that indexes a numpy vector of entities uses *idx*.
- Anything that names a node in the current latent topology uses *lid*.
- Anything that must be stable across copies/derivations uses a *gid*.

Global ID allocators
--------------------
- Cluster CIDs (and row/col CIDs inside each Chain/Ring) are allocated monotonically and never reused.
- Product cell GIDs are allocated per (rgid, cgid) pair and never reused even if the grid changes.
"""


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


def edge_order(u, v):
    """Return a consistent undirected edge (u,v) with u<=v."""
    return (u, v) if node_sort_key(u) <= node_sort_key(v) else (v, u)


class ClusterGraph:
    """
    Represents the topology of a graph, with entities associated to latent cluster nodes.

    entity_assignments[idx] -> lid (or -1 if unassigned/orphaned).
    latent_adjacency         : {lid: set[lid]} undirected adjacency among latent nodes.

    ID maps
    -------
    - entity_idx_to_eid : {idx -> eid}
    - latent_lid_to_cid : {lid -> cid}
    """
    def __init__(self, entity_assignments: np.ndarray, 
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 metadata: Optional[Dict] = None,
                 from_product: Optional[bool] = False):
        self.entity_assignments = entity_assignments.copy().astype(int)
        all_ids = set(np.unique(self.entity_assignments.tolist()))
        self.latent_ids = {lid for lid in all_ids if lid != -1}

        if latent_adjacency is None:
            self.latent_adjacency = {lid: set() for lid in self.latent_ids}
        else:
            # ensure symmetry
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

        # Global ID maps & allocators
        self.entity_idx_to_eid: Dict[int, int] = {}
        self.latent_lid_to_cid: Dict[int, int] = {}
        self.next_cluster_cid: int = 0  # allocator for new cluster CIDs
        self.init_id_maps()

        self.from_product = from_product # is this ClusterGraph fromed by taking the cartesian product of a ProductClusterGraph

    def normalise(self):
        # enforce symmetry
        for a, neighs in list(self.latent_adjacency.items()):
            for b in list(neighs):
                self.latent_adjacency.setdefault(b, set()).add(a)
    
    def init_id_maps(self, entity_idx_to_eid: Optional[Dict[int, int]] = None):

        # entities -- set once
        if not self.entity_idx_to_eid:
            if entity_idx_to_eid is None:
                for i in range(self.n_entities()):
                    self.entity_idx_to_eid[i] = i
            else:
                self.entity_idx_to_eid.update({int(k): int(v) for k, v in entity_idx_to_eid.items()})

        # initialise allocator after max entity gid (so CIDs start after EIDs -- important for later when 
        # forming edge tuples in entity graph; each node needs a unique index)
        if self.next_cluster_cid == 0:
            e_vals = list(self.entity_idx_to_eid.values())
            self.next_cluster_cid = (max(e_vals) + 1) if e_vals else 0

        # ensure all current latents have a stable cluster cid
        for lid in self.latent_ids:
            if lid not in self.latent_lid_to_cid:
                self.latent_lid_to_cid[lid] = self.next_cluster_cid
                self.next_cluster_cid += 1

    def n_entities(self):
        return len(self.entity_assignments)

    def n_latents(self):
        return len(self.latent_ids)

    def copy(self):
        g = ClusterGraph(self.entity_assignments.copy(),
                         {k: set(v) for k, v in self.latent_adjacency.items()},
                         metadata=copy.deepcopy(self.metadata),
                         from_product=self.from_product)
        g.entity_idx_to_eid = dict(self.entity_idx_to_eid)
        g.latent_lid_to_cid = dict(self.latent_lid_to_cid)
        g.next_cluster_cid = int(self.next_cluster_cid)
        return g
    
    def __str__(self):
        return (
            "ClusterGraph(\n"
            f"  latent_lids={sorted(self.latent_ids)},\n"
            f"  id_maps={{'entity_idx_to_eid': {self.entity_idx_to_eid}, 'latent_lid_to_cid': {self.latent_lid_to_cid}}},\n"
            f"  entity_assignments[idx->lid]={self.entity_assignments.tolist()},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in self.latent_adjacency.items())}}},\n"
            f"  metadata={self.metadata}\n"
            ")"
        )
    
    def entity_counts(self) -> Dict[int, int]:
        """Count entities per latent lid (ignores -1)."""
        counts: Dict[int, int] = {}
        for lid in self.entity_assignments:
            if lid == -1:
                continue
            counts[lid] = counts.get(lid, 0) + 1
        return counts

    def add_latents(self, n: int) -> List[int]:
        """
        Add `n` new latent nodes, choosing integer lids not already present.
        Returns the list of new latent lids.
        """
        if n <= 0:
            return []
        existing = set(self.latent_ids)
        next_lid = (max(existing) + 1) if existing else 0
        new_lids: List[int] = []
        for i in range(n):
            lid = next_lid + i
            self.latent_adjacency.setdefault(lid, set())
            new_lids.append(lid)
        self.latent_ids.update(new_lids)

        for lid in new_lids:
            self.latent_lid_to_cid[lid] = self.next_cluster_cid
            self.next_cluster_cid += 1

        self.normalise()
        return new_lids

    def remove_latents(self, latent_lids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Remove specified latent nodes.
        Entities assigned to removed latents become orphaned (assignment set to -1).
        Returns a list of orphaned entity idxs.
        """
        orphaned: List[int] = []
        latent_lids = list(latent_lids)
        for lid in latent_lids:
            # find entities assigned to this latent before removal
            mask = self.entity_assignments == lid
            orphaned_entities = np.where(mask)[0]
            if orphaned_entities.size > 0:
                orphaned.extend(orphaned_entities.tolist())
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
        Add undirected latent-latent edges.
        """
        for a, b in edges:
            if a == b:
                raise ValueError(f"Cannot add self-loop on latent {a}.")
            if a not in self.latent_ids or b not in self.latent_ids:
                raise ValueError(f"Edge ({a}, {b}) references unknown latent id(s). Known: {sorted(self.latent_ids)}")
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
        """Replace the entity->lid assignments (values must be lids or -1)."""
        self.entity_assignments = new_assignments.copy().astype(int)

    def undo_cartesian_product(self):
        if not self.from_product:
            raise TypeError("Cannot undo cartesian product when the graph was not made by forming one.")
        
        return ProductClusterGraph(
            self.metadata['row_graph'],
            self.metadata['col_graph'],
            self.metadata['row_assignments'],
            self.metadata['col_assignments'],
            copy.deepcopy(self.metadata),
            self.entity_idx_to_eid)



class ChainRingClusterGraph(ClusterGraph):
    # chain / ring forms keep an order of latent lids
    def __init__(self, entity_assignments: np.ndarray, 
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 order: List[int] = None,
                 metadata: Optional[Dict] = None):
        super().__init__(entity_assignments, latent_adjacency, metadata)
        self.order = order if order else [0]

    def remove_latents(self, latent_lids: Iterable[int]) -> Dict[int, np.ndarray]:
        """Overrides base to also update the order list."""
        lids_to_remove = set(latent_lids)
        orphaned = super().remove_latents(lids_to_remove)
        self.order = [lid for lid in self.order if lid not in lids_to_remove] 
        return orphaned
    
    def copy(self):
        g = ChainRingClusterGraph(self.entity_assignments.copy(),
                                  {k: set(v) for k, v in self.latent_adjacency.items()},
                                  self.order.copy(),
                                  metadata=copy.deepcopy(self.metadata))
        g.entity_idx_to_eid = dict(self.entity_idx_to_eid)
        g.latent_lid_to_cid = dict(self.latent_lid_to_cid)
        g.next_cluster_cid = int(self.next_cluster_cid)
        return g
    
    def __str__(self):
        return (
            "ChainRingClusterGraph(\n"
            f"  latent_lids={sorted(self.latent_ids)},\n"
            f"  order(lids)={self.order},\n"
            f"  id_maps={{'entity_idx_to_eid': {self.entity_idx_to_eid}, 'latent_lid_to_cid': {self.latent_lid_to_cid}}},\n"
            f"  entity_assignments[idx->lid]={self.entity_assignments.tolist()},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in self.latent_adjacency.items())}}},\n"
            f"  metadata={self.metadata}\n"
            ")"
        )
    

class HierarchyClusterGraph(ClusterGraph):
    # hierarchy form keeps a parent and child dictionary mapping {lid: set[lid]}
    def __init__(self, entity_assignments: np.ndarray,
                 latent_adjacency: Optional[Dict[int, Set[int]]] = None,
                 parent: Optional[Dict[int, Optional[int]]] = None,
                 children: Optional[Dict[int, Set[int]]] = None,
                 metadata: Optional[Dict] = None):
        super().__init__(entity_assignments, latent_adjacency, metadata)
        if parent is None and children is None:
            self.parent: Dict[int, Optional[int]] = {0: None}
            self.children: Dict[int, Set[int]] = {0: set()}
        else:
            self.parent = {k: v for k, v in parent.items()}
            self.children = {k: set(v) for k, v in children.items()}

        # ensure every latent has entries
        for lid in self.latent_ids:
            self.parent.setdefault(lid, None if lid not in self.parent else self.parent[lid])
            self.children.setdefault(lid, set())

    def set_parent_of(self, child: int, parent: Optional[int]):
        old_parent = self.parent.get(child, None)
        if old_parent is not None:
            self.children.setdefault(old_parent, set()).discard(child)
        self.parent[child] = parent
        if parent is not None:
            self.children.setdefault(parent, set()).add(child)

    def add_child(self, parent: int, child: int):
        self.set_parent_of(child, parent)

    def children_of(self, node: int) -> Set[int]:
        return set(self.children.get(node, set()))

    def leaf_ids(self) -> List[int]:
        """Return lids whose undirected degree <= 1."""
        leaves = []
        for lid in self.latent_ids:
            degree = len(self.latent_adjacency.get(lid, set()))
            if degree <= 1:
                leaves.append(lid)
        return sorted(leaves)
    
    def remove_latents(self, latent_lids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Overrides base to also update the parent & children maps.
        Children of removed nodes become new roots (parent=None).
        """
        lids_to_remove = set(latent_lids)

        # update parent/children before topology change
        for lid in lids_to_remove:
            parent = self.parent.pop(lid, None)
            if parent is not None:
                self.children.setdefault(parent, set()).discard(lid)
            children = self.children.pop(lid, set())
            for c in children:
                self.parent[c] = None

        orphaned = super().remove_latents(lids_to_remove)

        for lid in self.latent_ids:
            self.parent.setdefault(lid, self.parent.get(lid, None))
            self.children.setdefault(lid, set())

        return orphaned

    def copy(self):
        g = HierarchyClusterGraph(
            self.entity_assignments.copy(),
            {k: set(v) for k, v in self.latent_adjacency.items()},
            parent=copy.deepcopy(self.parent),
            children={k: set(v) for k, v in self.children.items()},
            metadata=copy.deepcopy(self.metadata)
        )
        g.entity_idx_to_eid = dict(self.entity_idx_to_eid)
        g.latent_lid_to_cid = dict(self.latent_lid_to_cid)
        g.next_cluster_cid = int(self.next_cluster_cid)
        return g

    def __str__(self):
        parent_str = '{' + ', '.join(f'{k}:{v}' for k, v in sorted(self.parent.items())) + '}'
        children_str = '{' + ', '.join(f'{k}:{sorted(list(v))}' for k, v in sorted(self.children.items())) + '}'
        return (
            "HierarchyClusterGraph(\n"
            f"  latent_lids={sorted(self.latent_ids)},\n"
            f"  id_maps={{'entity_idx_to_eid': {self.entity_idx_to_eid}, 'latent_lid_to_cid': {self.latent_lid_to_cid}}},\n"
            f"  parent={parent_str},\n"
            f"  children={children_str},\n"
            f"  leaf_lids={self.leaf_ids},\n"
            f"  entity_assignments[idx->lid]={self.entity_assignments.tolist()},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in sorted(self.latent_adjacency.items()))}}},\n"
            f"  metadata={self.metadata}\n"
            ")"
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
            self.parent: Dict[int, Optional[int]] = {0: None}
            self.children: Dict[int, Set[int]] = {0: set()}
        else:
            self.parent = {k: v for k, v in parent.items()}
            self.children = {k: set(v) for k, v in children.items()}

        # ensure every latent has entries
        for lid in self.latent_ids:
            self.parent.setdefault(lid, None if lid not in self.parent else self.parent[lid])
            self.children.setdefault(lid, set())

    def children_of(self, node: int) -> Set[int]:
        return set(self.children.get(node, set()))

    def reclassify(self, lid: int):
        """Make sure leaf/internal sets match the children map for lid provided."""
        if len(self.children.get(lid, set())) == 0:
            self.internal_ids.discard(lid)
            self.leaf_ids.add(lid)
        else:
            self.leaf_ids.discard(lid)
            self.internal_ids.add(lid)

    def set_parent_of(self, child: int, parent: Optional[int]):
        old_parent = self.parent.get(child, None)
        if old_parent is not None:
            self.children.setdefault(old_parent, set()).discard(child)
            self.reclassify(old_parent)  # may have become a leaf

        self.parent[child] = parent
        if parent is not None:
            self.children.setdefault(parent, set()).add(child)
            self.reclassify(parent) # definitely an internal now

    def remove_latents(self, latent_lids: Iterable[int]) -> Dict[int, np.ndarray]:
        """
        Also update leaf/internal sets + parent/children maps.
        Children of removed nodes become new roots (parent=None).
        """
        lids_to_remove = set(latent_lids)

        # update hierarchical maps
        for lid in lids_to_remove:
            parent = self.parent.pop(lid, None)
            if parent is not None:
                self.children.setdefault(parent, set()).discard(lid)
                self.reclassify(parent)
            children = self.children.pop(lid, set())
            for c in children:
                self.parent[c] = None  # re root children
            self.leaf_ids.discard(lid)
            self.internal_ids.discard(lid)

        orphaned = super().remove_latents(lids_to_remove)

        for lid in self.latent_ids:
            self.parent.setdefault(lid, self.parent.get(lid, None))
            self.children.setdefault(lid, set())

        return orphaned
    
    def copy(self):
        g = TreeClusterGraph(
            self.entity_assignments.copy(),
            {k: set(v) for k, v in self.latent_adjacency.items()},
            self.leaf_ids.copy(),
            self.internal_ids.copy(),
            parent=copy.deepcopy(self.parent),
            children={k: set(v) for k, v in self.children.items()},
            metadata=copy.deepcopy(self.metadata)
        )
        g.entity_idx_to_eid = dict(self.entity_idx_to_eid)
        g.latent_lid_to_cid = dict(self.latent_lid_to_cid)
        g.next_cluster_cid = int(self.next_cluster_cid)
        return g
    
    def __str__(self):
        return (
            "TreeClusterGraph(\n"
            f"  entity_assignments[idx->lid]={self.entity_assignments.tolist()},\n"
            f"  id_maps={{'entity_idx_to_eid': {self.entity_idx_to_eid}, 'latent_lid_to_cid': {self.latent_lid_to_cid}}},\n"
            f"  latent_lids={sorted(self.latent_ids)},\n"
            f"  internal_lids={self.internal_ids},\n"
            f"  leaf_lids={self.leaf_ids},\n"
            f"  parent={{{', '.join(f'{k}:{v}' for k,v in sorted(self.parent.items()))}}},\n"
            f"  children={{{', '.join(f'{k}:{sorted(list(v))}' for k,v in sorted(self.children.items()))}}},\n"
            f"  latent_adjacency={{{', '.join(f'{k}:{sorted(v)}' for k,v in self.latent_adjacency.items())}}},\n"
            f"  metadata={self.metadata}\n"
            ")"
        )


class ProductClusterGraph:
    """
    ProductClusterGraph supports products of two 1D component graphs (e.g., Chain×Chain for grids,
    Chain×Ring for cylinders). It stores separate assignments per axis.

    Row/Col axis IDs
    -----------------
    - row_lid_to_rgid: {row_lid -> rgid} using the underlying row_graph's latent_lid_to_cid
    - col_lid_to_cgid: {col_lid -> cgid} using the underlying col_graph's latent_lid_to_cid

    - entity_idx_to_eid: {idx -> eid} stable entity mapping
    - cell_gid_map: {(rgid, cgid) -> cell_gid} unique, never-reused product cell IDs
    """
    def __init__(self,
                 row_graph: ChainRingClusterGraph,
                 col_graph: ChainRingClusterGraph,
                 row_assignments: np.ndarray,
                 col_assignments: np.ndarray,
                 metadata: Optional[Dict] = None,
                 entity_idx_to_eid: Optional[Dict[int, int]] = None):
        if len(row_assignments) != len(col_assignments):
            raise ValueError("row and column assignments must align in length")
        self.row_graph = row_graph
        self.col_graph = col_graph
        self.row_assignments = row_assignments.copy().astype(int) # values are row lids (or -1)
        self.col_assignments = col_assignments.copy().astype(int) # values are col lids (or -1)

        # derive axis-specific stable ids directly from the component graphs' cids
        self.row_lid_to_rgid: Dict[int, int] = {rl: self.row_graph.latent_lid_to_cid[rl] for rl in self.row_graph.order}
        self.col_lid_to_cgid: Dict[int, int] = {cl: self.col_graph.latent_lid_to_cid[cl] for cl in self.col_graph.order}
        self.rgid_to_row_lid: Dict[int, int] = {rgid: rl for rl, rgid in self.row_lid_to_rgid.items()}
        self.cgid_to_col_lid: Dict[int, int] = {cgid: cl for cl, cgid in self.col_lid_to_cgid.items()}

        self.metadata = {} if metadata is None else copy.deepcopy(metadata)
        self.entity_idx_to_eid: Dict[int, int] = (
            {i: i for i in range(self.n_entities())}
            if entity_idx_to_eid is None else {int(k): int(v) for k, v in entity_idx_to_eid.items()}
        )

    def n_entities(self) -> int:
        return len(self.row_assignments)

    def entity_assignments(self) -> List[Tuple[int,int]]:
        """Return list of (row_lid, col_lid) for each entity idx."""
        return list(zip(self.row_assignments.tolist(), self.col_assignments.tolist()))

    def update_entity_assignments(self, new_assignments: List[Tuple[int,int]]):
        # new_assignments is a list of (row_lid, col_lid)
        for i, (r, c) in enumerate(new_assignments):
            self.row_assignments[i] = r
            self.col_assignments[i] = c
        self.row_graph.entity_assignments = self.row_assignments
        self.col_graph.entity_assignments = self.col_assignments
    
    def form_cartesian_product(self) -> ClusterGraph:
        rows = self.row_graph.order  # in local row lids
        cols = self.col_graph.order  # in local col lids

        row_gid_map = self.row_graph.latent_lid_to_cid  # row_lid -> rgid
        col_gid_map = self.col_graph.latent_lid_to_cid  # col_lid -> cgid

        # assign a stable cell_gid for every current (rgid, cgid)
        current_cells = []  # (row_lid, col_lid, (rgid, cgid))
        for r in rows:
            for c in cols:
                rgid = row_gid_map[r]
                cgid = col_gid_map[c]
                cell_gid = (rgid, cgid)
                current_cells.append((r, c, cell_gid))

        # create local cell-lids
        current_cells.sort(key=lambda t: t[2])
        cell_gid_to_local_lid = {cell_gid: idx for idx, (_, _, cell_gid) in enumerate(current_cells)}
        local_lid_from_rc = {(r, c): cell_gid_to_local_lid[cell_gid] for (r, c, cell_gid) in current_cells}

        # entity assignments over local cell-lids
        new_assign = np.full(self.n_entities(), -1, dtype=int)
        for i in range(self.n_entities()):
            r, c = self.row_assignments[i], self.col_assignments[i]
            if r != -1 and c != -1:
                new_assign[i] = local_lid_from_rc[(r, c)]

        # build full adjacency over the grid cells
        n_cells = len(current_cells)
        adj: Dict[int, Set[int]] = {lid: set() for lid in range(n_cells)}
        # along rows
        for r, neighs in self.row_graph.latent_adjacency.items():
            for r2 in neighs:
                for c in cols:
                    a = local_lid_from_rc[(r, c)]
                    b = local_lid_from_rc[(r2, c)]
                    adj[a].add(b); adj[b].add(a)
        # along cols
        for c, neighs in self.col_graph.latent_adjacency.items():
            for c2 in neighs:
                for r in rows:
                    a = local_lid_from_rc[(r, c)]
                    b = local_lid_from_rc[(r, c2)]
                    adj[a].add(b); adj[b].add(a)

        metadata = copy.deepcopy(self.metadata)
        metadata['row_graph'] = self.row_graph
        metadata['col_graph'] = self.col_graph
        metadata['row_assignments'] = self.row_assignments
        metadata['col_assignments'] = self.col_assignments
        cg = ClusterGraph(new_assign, latent_adjacency=adj, metadata=metadata, from_product=True)
        cg.entity_idx_to_eid = {i: int(self.entity_idx_to_eid[i]) for i in range(self.n_entities())}
        for (_, _, cell_gid) in current_cells:
            local_lid = cell_gid_to_local_lid[cell_gid]
            cg.latent_lid_to_cid[local_lid] = cell_gid  # stable unique product cell id
        cg.init_id_maps()
        return cg
    
    def copy(self) -> 'ProductClusterGraph':
        g = ProductClusterGraph(
            self.row_graph.copy(),
            self.col_graph.copy(),
            self.row_assignments.copy(),
            self.col_assignments.copy(),
            copy.deepcopy(self.metadata),
            entity_idx_to_eid=dict(self.entity_idx_to_eid),
        )
        return g

    def __str__(self):
        full_product_graph = self.form_cartesian_product()
        full_adj_str = '{' + ', '.join(f'{k}:{sorted(list(v))}' for k, v in sorted(full_product_graph.latent_adjacency.items())) + '}'

        return (
            "ProductClusterGraph(\n"
            # Component graph info
            f"  -- Component Graphs --\n"
            f"  row_order(lids)={self.row_graph.order},\n"
            f"  col_order(lids)={self.col_graph.order},\n"
            f"  row_lid->rgid={self.row_lid_to_rgid},\n"
            f"  col_lid->cgid={self.col_lid_to_cgid},\n"
            f"  row_assignments[idx->row_lid]={self.row_assignments.tolist()},\n"
            f"  col_assignments[idx->col_lid]={self.col_assignments.tolist()},\n"
            f"  row_graph_adj={{{', '.join(f'{k}:{sorted(v)}' for k,v in sorted(self.row_graph.latent_adjacency.items()))}}},\n"
            f"  col_graph_adj={{{', '.join(f'{k}:{sorted(v)}' for k,v in sorted(self.col_graph.latent_adjacency.items()))}}},\n"
            # Full Cartesian product info
            f"  -- Full Cartesian Product --\n"
            f"  product_latent_lids={sorted(full_product_graph.latent_ids)},\n"
            f"  product_entity_assignments[idx->cell_lid]={full_product_graph.entity_assignments.tolist()},\n"
            f"  product_latent_adjacency={full_adj_str},\n"
            f"  id_maps(product)={{'entity_idx_to_eid': {full_product_graph.entity_idx_to_eid}, 'latent_lid_to_cid(cell_gid)': {full_product_graph.latent_lid_to_cid}}},\n"
            f"  metadata={self.metadata}\n"
            ")"
        )
        

class EntityGraph:
    def __init__(self, entity_gids: list[int], cluster_cids: list[int]):

        self.entity_gids: list[int] = list(entity_gids)
        self.cluster_cids: list[int] = list(cluster_cids)

        self.n_entities: int = len(self.entity_gids)
        self.n_latent: int = len(self.cluster_cids)
        self.n_nodes: int = self.n_entities + self.n_latent

        self.edges_external: list[tuple[int, int]] = []
        self.edges_internal: list[tuple[int, int]] = []
        self.edges: list[tuple[int, int]] = []

    def add_edge(self, u_gid: int, v_gid: int) -> tuple[int, int] | None:
        e = edge_order(u_gid, v_gid)
        self.edges.append(e)
        return e

    def add_external(self, eid: int, cid: int) -> None:
        """Add an entity→cluster edge in global ids."""
        e = self.add_edge(eid, cid)
        self.edges_external.append(e)

    def add_internal(self, cid_u: int, cid_v: int) -> None:
        """Add a cluster↔cluster edge in global ids."""
        e = self.add_edge(cid_u, cid_v)
        self.edges_internal.append(e)


class SplitProposal(NamedTuple):
    split_graph: ClusterGraph      # graph with topology split but no reassignments
    parent_members: np.ndarray     # the entity idx that were in the split parent
    children: Tuple[int, int]      # child lids (for the split axis)
    axis: Optional[str]            # 'row' or 'col' for a split in a Product form; None otherwise


class StructuralForm:
    name = "base"

    def initial_structure(self, n_entities: int) -> ClusterGraph:
        # single latent cluster containing all entities, no latent-latent edges
        return ClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: ClusterGraph, cluster_lid: int) -> Tuple[ClusterGraph, np.ndarray, List[int]]:
        """
        Partition-like split--remove parent lid, add two child lids inheriting the parent's neighbours.
        Does not assign any entities to the new children; returns:
          - new topology ClusterGraph (with parent removed, children present, entities orphaned)
          - array of entity idxs that were in the parent (to be distributed)
          - list of new child lids (length 2)
        """
        cg = cluster_graph.copy()
        new_children = cg.add_latents(2)
        orphaned = cg.remove_latents([cluster_lid])
        return cg, orphaned, new_children
    
    def propose_splits(self, G: ClusterGraph):
        proposals = []
        for lid in G.latent_ids:
            new_G, members, [c1,c2] = self.split_cluster(G, lid)
            proposals.append(SplitProposal(new_G, members, (c1,c2), None))
        return proposals
    

class PartitionForm(StructuralForm):
    name = "partition"


class OrderForm(StructuralForm):
    """All latent nodes fully connect to all others."""
    name = "connected"

    def split_cluster(self, cluster_graph: ClusterGraph, cluster_lid: int) -> Tuple[ClusterGraph, np.ndarray, List[int]]:
        cg, orphoned, new_children = super().split_cluster(cluster_graph, cluster_lid)
        # fully connect all latent nodes
        nodes = list(cg.latent_ids)
        for u, v in combinations(nodes, 2):
            cg.add_edges([(u, v)])
        return cg, orphoned, new_children
    

class ChainForm(StructuralForm):
    name = "chain"

    def initial_structure(self, n_entities: int) -> ChainRingClusterGraph:
        return ChainRingClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: ChainRingClusterGraph, cluster_lid: int) -> Tuple[ChainRingClusterGraph, np.ndarray, List[int]]:
        """Replace cluster lid with two child lids; update chain order so they are adjacent."""
        cg = cluster_graph.copy()
        order = cg.order
        idx = order.index(cluster_lid)

        left_neighbour = order[idx - 1] if idx > 0 else None
        right_neighbour = order[idx + 1] if idx < len(order) - 1 else None

        # add two new child latent nodes
        new_children = cg.add_latents(2)
        child_a, child_b = new_children

        # remove the parent cluster, orphaning its entities
        orphaned = cg.remove_latents([cluster_lid])

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

        return cg, orphaned, new_children
    

class RingForm(StructuralForm):
    name = "ring"

    def initial_structure(self, n_entities: int) -> ChainRingClusterGraph:
        return ChainRingClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: ChainRingClusterGraph, cluster_lid: int) -> Tuple[ChainRingClusterGraph, np.ndarray, List[int]]:
        """
        Replace cluster_id with two children, update chain order so they are adjacent.
        """
        cg = cluster_graph.copy()
        order = cg.order
        idx = order.index(cluster_lid)

        left_neighbour = order[idx - 1] if idx > 0 else order[-1] if len(order) > 1 else None
        right_neighbour = order[idx + 1] if idx < len(order) - 1 else order[0] if len(order) > 1 else None

        # add two new child latent nodes
        new_children = cg.add_latents(2)
        child_a, child_b = new_children

        # remove the parent cluster, orphaning its entities
        orphaned = cg.remove_latents([cluster_lid])

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

        return cg, orphaned, new_children
    

class HierarchyForm(StructuralForm):
    """
    Hierarchy structural form: maintains a tree hierarchy of latent clusters.
    On split, one child replaces the parent (inheriting its connections),
    while the other becomes a new leaf attached only to that child.
    """
    name = "hierarchy"

    def initial_structure(self, n_entities: int) -> HierarchyClusterGraph:
        return HierarchyClusterGraph(np.zeros(n_entities, dtype=int))
    
    def split_cluster(self, cluster_graph: HierarchyClusterGraph, cluster_lid: int) -> Tuple[HierarchyClusterGraph, np.ndarray, List[int]]:
        cg = cluster_graph.copy()

        # parent of the cluster to split
        parent_of_cluster = cg.parent.get(cluster_lid)

        parent_neighbours = set(cg.latent_adjacency.get(cluster_lid, set()))
        children_of_cluster = set(cg.children_of(cluster_lid))

        # add two new child latent nodes
        new_children = cg.add_latents(2)
        child_a, child_b = new_children

        # remove the parent cluster, orphaning its entities
        orphaned = cg.remove_latents([cluster_lid])

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

        return cg, orphaned, new_children
    

class TreeForm(StructuralForm):
    """
    Tree structural form: maintains a binary tree where only leaves
    carry entity assignments. Internal nodes have no entities.
    Splitting is only applied to leaves; non-leaf splits return.
    """
    name = "tree"

    def initial_structure(self, n_entities: int) -> TreeClusterGraph:
        return TreeClusterGraph(np.zeros(n_entities, dtype=int))

    def split_cluster(self, cluster_graph: TreeClusterGraph, cluster_lid: int) -> Tuple[TreeClusterGraph, np.ndarray, List[int]]:
        cg = cluster_graph.copy()
        internal_ids = list(cg.internal_ids)
        leaf_ids = list(cg.leaf_ids)

        if cluster_lid not in leaf_ids:
            # can't split internal nodes
            return cg, np.array([], dtype=int), []

        # orphan entities from the leaf
        orphaned = np.where(cg.entity_assignments == cluster_lid)[0]
        cg.entity_assignments[orphaned] = -1

        # add two new leaf children and attach to the internal node
        new_children = cg.add_latents(2)
        edges = [(cluster_lid, new_children[0]), (cluster_lid, new_children[1])]
        cg.add_edges(edges)

        cg.set_parent_of(new_children[0], cluster_lid)
        cg.set_parent_of(new_children[1], cluster_lid)
        internal_ids.append(cluster_lid)
        leaf_ids.remove(cluster_lid)
        leaf_ids.extend(new_children)
        cg.internal_ids = set(internal_ids)
        cg.leaf_ids = set(leaf_ids)

        return cg, orphaned, new_children
    
    def propose_splits(self, G: TreeClusterGraph):
        proposals = []
        for lid in G.leaf_ids:
            new_G, members, [c1,c2] = self.split_cluster(G, lid)
            proposals.append(SplitProposal(new_G, members, (c1, c2), None))
        return proposals
    

class GridForm():
    """Grid form: product of Chain × Chain."""
    name = "grid"

    def initial_structure(self, n_entities: int) -> ProductClusterGraph:
        row_graph = ChainForm().initial_structure(n_entities)  # chain axis
        col_graph = ChainForm().initial_structure(n_entities)   # chain axis
        row_assign = np.zeros(n_entities, dtype=int)
        col_assign = np.zeros(n_entities, dtype=int)
        return ProductClusterGraph(row_graph, col_graph, row_assign, col_assign)

    def propose_splits(self, G: ProductClusterGraph):
        proposals = []
        # split along rows
        for r in G.row_graph.latent_ids:
            new_row_graph, orphaned, (c1, c2) = ChainForm().split_cluster(G.row_graph, r)
            row_assign = G.row_assignments.copy()
            row_assign[orphaned] = -1
            updated_G = ProductClusterGraph(
                new_row_graph,     
                G.col_graph,  
                row_assign,           
                G.col_assignments,
                metadata=copy.deepcopy(G.metadata),
                entity_idx_to_eid=dict(G.entity_idx_to_eid)
            )
            proposals.append(SplitProposal(updated_G, orphaned, (c1, c2), 'row'))
        # split along cols
        for c in G.col_graph.latent_ids:
            new_col_graph, orphaned, (c1, c2) = ChainForm().split_cluster(G.col_graph, c)
            col_assign = G.col_assignments.copy()
            col_assign[orphaned] = -1
            updated_G = ProductClusterGraph(
                G.row_graph,  
                new_col_graph,  
                G.row_assignments,           
                col_assign,
                metadata=copy.deepcopy(G.metadata),
                entity_idx_to_eid=dict(G.entity_idx_to_eid)
            )
            proposals.append(SplitProposal(updated_G, orphaned, (c1, c2), 'col'))
        return proposals


class CylinderForm():
    """Cylinder form: product of Chain × Ring."""
    name = "cylinder"

    def initial_structure(self, n_entities: int) -> ProductClusterGraph:
        row_graph = ChainForm().initial_structure(n_entities)  # chain axis
        col_graph = RingForm().initial_structure(n_entities)   # ring axis
        row_assign = np.zeros(n_entities, dtype=int)
        col_assign = np.zeros(n_entities, dtype=int)
        return ProductClusterGraph(row_graph, col_graph, row_assign, col_assign)

    def propose_splits(self, G: ProductClusterGraph):
        proposals = []
        # split along rows
        for r in G.row_graph.latent_ids:
            new_row_graph, orphaned, (c1, c2) = ChainForm().split_cluster(G.row_graph, r)
            row_assign = G.row_assignments.copy()
            row_assign[orphaned] = -1
            updated_G = ProductClusterGraph(
                new_row_graph,     
                G.col_graph,  
                row_assign,           
                G.col_assignments,
                metadata=copy.deepcopy(G.metadata),
                entity_idx_to_eid=dict(G.entity_idx_to_eid)
            )
            proposals.append(SplitProposal(updated_G, orphaned, (c1, c2), 'row'))
        # split along cols
        for c in G.col_graph.latent_ids:
            new_col_graph, orphaned, (c1, c2) = RingForm().split_cluster(G.col_graph, c)
            col_assign = G.col_assignments.copy()
            col_assign[orphaned] = -1
            updated_G = ProductClusterGraph(
                G.row_graph,  
                new_col_graph,  
                G.row_assignments,           
                col_assign,
                metadata=copy.deepcopy(G.metadata),
                entity_idx_to_eid=dict(G.entity_idx_to_eid)
            )
            proposals.append(SplitProposal(updated_G, orphaned, (c1, c2), 'col'))
        return proposals