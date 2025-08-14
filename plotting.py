
from typing import Dict, Tuple, Optional, List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import plotly.graph_objects as go
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.patheffects as pe


def build_graph(cg, len_scale: float = 1.0):
    """
    Build a NetworkX graph whose node ids are the global ids:
      - entity nodes use eid (from cg.entity_idx_to_eid)
      - cluster nodes use cid (from cg.latent_lid_to_cid)
        (in product graphs these are tuples (rgid, cgid))

    Edge lengths come from cg.metadata['opt_params']['untied']['log_ls'].
    """
    # local -> global maps
    local_ent_to_eid: Dict[int, object] = dict(getattr(cg, "entity_idx_to_eid", {}))  # idx -> eid
    local_lat_to_cid: Dict[int, object] = dict(getattr(cg, "latent_lid_to_cid", {}))  # lid -> cid

    # global -> local inverses (for labeling)
    gid2local_entity: Dict[object, int] = {eid: idx for idx, eid in local_ent_to_eid.items()}
    gid2local_cluster: Dict[object, int] = {cid: lid for lid, cid in local_lat_to_cid.items()}

    log_map: Dict[Tuple[object, object], float] = (
        cg.metadata.get("opt_params", {}).get("untied", {}).get("log_ls", {})
    )

    G = nx.Graph()

    for (u_gid, v_gid), logL in log_map.items():
        length = np.exp(float(logL)) * float(len_scale)

        if u_gid in gid2local_cluster:
            G.add_node(u_gid, kind="cluster")
        elif u_gid in gid2local_entity:
            G.add_node(u_gid, kind="entity")

        if v_gid in gid2local_cluster:
            G.add_node(v_gid, kind="cluster")
        elif v_gid in gid2local_entity:
            G.add_node(v_gid, kind="entity")

        G.add_edge(
            u_gid,
            v_gid,
            length=length,
            spring_weight=1.0 / max(length, 1e-12),
            kind="edge",
        )

    return G, gid2local_entity, gid2local_cluster


def compute_layout_fewer_crossings(G, scale, engine="neato"):
    """
    Prefer fewer crossings, while still using edge 'length' as ideal spring length.
    """
    lens = [d.get("length") for _, _, d in G.edges(data=True)]
    med = float(np.median(lens))

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        L = d.get("length") / med
        H.add_edge(u, v, len=max(L, 0.05))

    args = "-Goverlap=false -Gstart=rand -Gmaxiter=5000"
    if engine == "sfdp":
        args = "-Goverlap=false -GK=0.6 -Giterations=1000"
    pos = graphviz_layout(H, prog=engine, args=args)

    xs, ys = zip(*pos.values())
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    s = scale / span
    return {n: np.array([x * s, y * s], dtype=float) for n, (x, y) in pos.items()}


def compute_fixed_entity_positions(cg_top,
                                   len_scale: float,
                                   scale: float):
    """
    Use the best scoring graph to place entities and clusters,
    then freeze entity positions for the other graphs.
    """
    G_top, _, _ = build_graph(cg_top, len_scale=len_scale)
    pos_top = nx.kamada_kawai_layout(G_top, weight="length", scale=scale)

    entity_pos = {n: p for n, p in pos_top.items() if G_top.nodes[n].get("kind") == "entity"}
    return entity_pos


def plot_one_graph(cg,
                    entity_pos: Dict[int, np.ndarray],
                    plot_type: str,
                    entity_names: Optional[List[str]],
                    len_scale: float,
                    scale: float,
                    seed: int,
                    title: Optional[str] = None,
                    ax: Optional[plt.Axes] = None):
    """
    """
    G, e_gid2loc, _ = build_graph(cg, len_scale=len_scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal"); ax.axis("off")

    if plot_type == 'fixed':
        # initial positions: entities fixed, clusters start at their entities' centroid (if any) else origin
        pos_init = dict(entity_pos)
        for n, d in G.nodes(data=True):
            if d.get("kind") == "cluster" and n not in pos_init:
                # centroid of this cluster's attached entities (from edges)
                ent_neigh = [m for m in G.neighbors(n) if G.nodes[m].get("kind") == "entity" and m in entity_pos]
                if ent_neigh:
                    pos_init[n] = np.mean([entity_pos[m] for m in ent_neigh], axis=0)
                else:
                    pos_init[n] = np.array([0.0, 0.0])
        # spring layout with entities fixed
        pos = nx.spring_layout(
            G, pos=pos_init, fixed=list(entity_pos.keys()),
            weight="spring_weight", seed=seed, iterations=300
        )
    elif plot_type == 'non_overlapping':
        pos = compute_layout_fewer_crossings(G, scale=scale)
    elif plot_type == "unfixed":
        pos = nx.kamada_kawai_layout(G, weight="length", scale=scale)

    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), width=0.2*cg.n_entities(), edge_color="grey", ax=ax)

    clusters = [n for n, d in G.nodes(data=True) if d.get("kind") == "cluster"]
    nx.draw_networkx_nodes(
        G, pos, nodelist=clusters, node_size=int(7*cg.n_entities()), node_shape="o",
        edgecolors="black", node_color="none", linewidths=1.6, ax=ax
    )

    entities = [n for n, d in G.nodes(data=True) if d.get("kind") == "entity"]
    if entity_names is not None:
        e_labels = {}
        for n in entities:
            loc = e_gid2loc.get(n, None)
            e_labels[n] = entity_names[loc] if (loc is not None and 0 <= loc < len(entity_names)) else f"e{loc}"
        text_objs = nx.draw_networkx_labels(G, {k: pos[k] for k in entities}, labels=e_labels, font_size=cg.n_entities()/2, font_weight="bold", font_color="black", ax=ax)
        # add a clean white halo around each label for visibility
        # for t in text_objs.values():
        #     t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=entities, node_size=50*cg.n_entities(), node_shape="o", node_color="black", ax=ax)

    # if limits:
    #     ax.set_xlim(*limits["x"]); ax.set_ylim(*limits["y"])
    # ax.margins(x=0.05, y=0.05)  # tiny extra breathing room inside the axes
    if title:
        ax.set_title(title)

    return ax


def plot_score_bar_top(ax,
                       names,                 # all graph names, in order
                       scores,                # {name: score}
                       best_name: str):
    # keep provided order; drop any missing
    names = [n for n in names if n in scores]
    vals  = np.array([scores[n] for n in names], dtype=float)

    global_min = float(np.min(vals)) if len(vals) else 0.0
    global_max = float(np.max(vals)) if len(vals) else 1.0
    rng = global_max - global_min

    heights = (vals - global_min) + rng/50 # baseline shift so min -> rng/50
    pad = 0.06 * rng
    x = np.arange(len(names))
    colors = ["red" if n == best_name else "black" for n in names]

    ax.bar(x, heights, color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")

    ax.set_ylim(0, rng + pad)
    ax.yaxis.set_major_locator(NullLocator())
    ax.set_ylabel("")

    ax.margins(x=0.06)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # annotate actual values above bars
    top = ax.get_ylim()[1]
    for xi, h, v in zip(x, heights, vals):
        ax.text(xi, h + 0.02 * top, f"{v:.2f}", ha="center", va="bottom", fontsize=8)


def plot_score_bar(ax, names, scores, global_min, global_max, best_name, title="scores"):
    vals  = np.array([scores[n] for n in names], dtype=float)
    names = [n for n in names if n in scores]

    heights = [v - global_min for v in vals] 
    rng = global_max - global_min
    heights = [h+rng/50 for h in heights]
    pad = 0.06 * rng
    x = np.arange(len(names))
    colours = ["red" if n == best_name else "black" for n in names]

    ax.bar(x, heights, color=colours, width=0.7)
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_title(title, pad=6, fontsize=10)
    ax.set_ylim(0, rng + pad)
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(NullLocator())

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for xi, h, v in zip(x, heights, vals):
        ax.text(xi, h + (0.02 * (ax.get_ylim()[1] or 1.0)), f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
        

def position_bar_plot(ax, height_scale: float = 0.85, right_inset: float = 0.03):
    """
    Shrink the bar axis vertically and move its right edge inward by `right_inset`
    in figure-relative coordinates. Left edge stays fixed, so it doesn't move
    closer to the neighbour panel.
    """
    fig = ax.figure
    pos = ax.get_position()
    # shrink height and centre vertically
    new_h = pos.height * height_scale
    y0 = pos.y0 + 0.5 * (pos.height - new_h)
    # move slightly left from the right edge
    x0 = max(pos.x0 - right_inset, 0.0)
    ax.set_position([x0, y0, pos.width * 0.9, new_h]) 


def position_top_bar(ax, *, left_inset=0.04, right_inset=0.04, top_inset=0.03):
    """
    After tight_layout, shrink the bar axis horizontally (extra L/R padding)
    and pull it slightly down from the top edge.
    Insets are in *figure* coordinates.
    """
    pos = ax.get_position()  # [x0, y0, w, h] in figure coords
    # horizontal padding
    new_x0 = pos.x0 + left_inset
    new_w  = max(pos.width - (left_inset + right_inset), 0.1)
    # vertical: add extra gap to the *top* by shaving off height at the top
    new_h  = max(pos.height - top_inset, 0.05)
    new_y0 = pos.y0  # keep the bottom where it is
    ax.set_position([new_x0, new_y0, new_w, new_h])


def plot_all(graphs: List[Tuple[str, object]],
            save_path: str,
            scores,
            plot_type: str,
            entity_names: Optional[List[str]],
            len_scale: float = 1.0,
            scale: float = 8.0,
            seed: int = 7,
            panels_per_row: int = 3,
             bar_left_inset: float = 0.05,     # <- extra L padding for bar axis
             bar_right_inset: float = 0.05,    # <- extra R padding for bar axis
             bar_top_inset: float = 0.08):     # <- extra gap from the fig top
    """
    Renders rows of up to 3 graph panels + a per-row bar plot at right.
    All bar plots share the same baseline (global min score) and y-scale.
    The best form is drawn in red and its panel boxed in red.
    """
    best_name = max(scores, key=scores.get)
    name_to_cg = {nm: cg for nm, cg in graphs}
 
    for nm, cg in name_to_cg.items():
        if cg.__class__.__name__ == "ProductClusterGraph":
            name_to_cg[nm] = cg.form_cartesian_product()

    top_cg = name_to_cg.get(best_name, graphs[0][1])
    fixed_entity_pos = compute_fixed_entity_positions(cg_top=top_cg, len_scale=len_scale, scale=scale)

    n_graphs = len(graphs) # number of graph panels
    rows = int(np.ceil(n_graphs / panels_per_row))+1
    cols = panels_per_row

    per_panel_w, per_panel_h = 0.5*top_cg.n_entities(), 0.4*top_cg.n_entities()
    bar_row_height = per_panel_h
    bar_graph_gap = per_panel_h*0.2
    fig_w = per_panel_w * panels_per_row
    fig_h = bar_row_height + bar_graph_gap + per_panel_h * (rows-1)
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        1+rows, cols,
        height_ratios=[bar_row_height, bar_graph_gap] + [per_panel_h] * (rows-1),
        wspace=0.25, hspace=0.35
    )

    # top bar spanning the full width
    bar_ax = fig.add_subplot(gs[0, :])
    all_names = [nm for nm, _ in graphs]  # keep order
    plot_score_bar_top(bar_ax, names=all_names, scores=scores,
                       best_name=best_name)
    
    # spacer row (no axis content)
    fig.add_subplot(gs[1, :]).axis("off")

    idx = 0
    for r in range(rows-1):
        for c in range(cols):
            ax = fig.add_subplot(gs[r + 2, c])
            if idx < n_graphs:
                name, _ = graphs[idx]
                cg = name_to_cg[name]
                plot_one_graph(
                    cg, fixed_entity_pos, plot_type=plot_type, entity_names=entity_names,
                    len_scale=len_scale, scale=scale, seed=seed,
                    title=name, ax=ax
                )
                if name == best_name:
                    rect = plt.Rectangle((-0.1, -0.1), 1.2, 1.3, transform=ax.transAxes,
                                         fill=False, clip_on=False,
                                         edgecolor="red", linewidth=0.3*top_cg.n_entities())
                    ax.add_patch(rect)
                idx += 1
            else:
                ax.axis("off")

    fig.tight_layout()
    position_top_bar(bar_ax,
                     left_inset=bar_left_inset,
                     right_inset=bar_right_inset,
                     top_inset=bar_top_inset)
    
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig


def plot_best(
    cg,
    entity_names: Optional[List[str]] = None,
    len_scale: float = 1.0,
    scale: float = 3.0,
):
    """
    Solo plot for the best-scoring graph. Plotly interactive.
    """
    if cg.__class__.__name__ == "ProductClusterGraph":
        cg = cg.form_cartesian_product()

    G, e_gid2loc, c_gid2loc = build_graph(cg, len_scale=len_scale)
    pos = nx.kamada_kawai_layout(G, weight="length", scale=scale)

    # edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edges_tr = go.Scatter(x=edge_x, y=edge_y,
                          mode="lines",
                          line=dict(color="black", width=2),
                          hoverinfo="skip",
                          showlegend=False,
                          name="edges")

    # clusters (hollow black circles)
    clusters = [n for n, d in G.nodes(data=True) if d.get("kind") == "cluster"]
    cx = [pos[n][0] for n in clusters]
    cy = [pos[n][1] for n in clusters]
    clabels = [f"C{c_gid2loc.get(n,'?')}" for n in clusters]
    clusters_tr = go.Scatter(
        x=cx, y=cy, mode="markers+text",
        marker=dict(symbol="circle-open", size=14, line=dict(color="black", width=2)),
        name="clusters", hovertemplate="cluster %{text}<extra></extra>"
    )

    traces = [edges_tr, clusters_tr]

    # entities (filled black dots; hover shows name; if names provided, also labels)
    entities = [n for n, d in G.nodes(data=True) if d.get("kind") == "entity"]
    ex = [pos[n][0] for n in entities]
    ey = [pos[n][1] for n in entities]
    if entity_names is not None:
        names = [entity_names[e_gid2loc.get(n, -1)] if e_gid2loc.get(n, -1) >= 0 and e_gid2loc.get(n, -1) < len(entity_names)
                 else f"entity e{e_gid2loc.get(n, '?')}" for n in entities]
        entities_text_tr = go.Scatter(
            x=ex, y=ey, mode="text",
            text=names, textposition="top center", textfont=dict(size=9),
            showlegend=False, hoverinfo="skip"
        )
        traces.append(entities_text_tr)
    else:
        names = [f"entity e{e_gid2loc.get(n,'?')}" for n in entities]

    entities_tr = go.Scatter(
        x=ex, y=ey, mode="markers",
        marker=dict(symbol="circle", size=10, color="black"),
        name="entities", hovertext=names, hovertemplate="%{hovertext}<extra></extra>"
    )
    traces.append(entities_tr)

    fig = go.Figure(traces)
    fig.update_layout(
        template="none",
        dragmode="pan",
        hovermode="closest",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig