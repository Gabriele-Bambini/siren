"""C-GRAPH — context-graph builder for SIREN (E1..E5 edges, degree-preserving corruption).

Pure standard library; optional PyG export behind a guarded import.
JSON-serializable output format::

    {"nodes": [{"id": ..., "ntype": "element"|"gene", ...attrs}],
     "edges": [{"src": ..., "dst": ..., "etype": "E1".."E5",
                "distance": int|None, "score": float|None,
                "provenance": str}]}

Edge semantics (spec §5 / context_graph C-GRAPH):
- E1 distance-constrained candidate assignment (element -> gene).
- E2 physical chromatin contacts (bedpe anchor overlap), score = contact score.
- E3 same-TAD co-membership (undirected; requires ``tad_boundaries``, else skipped).
- E4 ABC-style weight = element activity x contact score (requires ``activity``).
- E5 eQTL-colocalization evidence (source DISJOINT from CRISPR labels by construction).
"""
from __future__ import annotations

import bisect
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "GraphNode",
    "GraphEdge",
    "SirenGraph",
    "build_context_graph",
    "permute_edges_preserving_degree",
    "to_pyg",
]


@dataclass
class GraphNode:
    """A graph node: either a candidate regulatory element or a promoter-gene."""

    id: str
    ntype: str  # "element" | "gene"
    chrom: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    tss: Optional[int] = None  # genes only
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    """A typed, attributed, directed edge."""

    src: str
    dst: str
    etype: str  # "E1".."E5"
    distance: Optional[int] = None
    score: Optional[float] = None
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SirenGraph:
    """Minimal container: nodes + edges, both JSON-serializable."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def edge_types(self) -> List[str]:
        return sorted({e.etype for e in self.edges})

    def without_edge_type(self, etype: str) -> "SirenGraph":
        return SirenGraph(nodes=self.nodes, edges=[e for e in self.edges if e.etype != etype])


def _norm_chrom(c: str) -> str:
    c = str(c).strip()
    if not c:
        return c
    if not c.startswith("chr"):
        c = "chr" + c
    alias = {"chrM": "chrMT", "chr23": "chrX", "chr24": "chrY"}
    return alias.get(c, c)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _parse_bedpe_row(row: Dict[str, Any]) -> Tuple[str, int, int, str, int, int, float]:
    """Accept common bedpe-ish key spellings and coerce types."""

    def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    ca = _norm_chrom(str(pick(row, "chrom_a", "chrom1", "c1", default="")))
    sa, ea = int(pick(row, "start_a", "start1", default=0)), int(pick(row, "end_a", "end1", default=0))
    cb = _norm_chrom(str(pick(row, "chrom_b", "chrom2", "c2", default="")))
    sb, eb = int(pick(row, "start_b", "start2", default=0)), int(pick(row, "end_b", "end2", default=0))
    sc = float(pick(row, "score", default=0.0))
    return ca, sa, ea, cb, sb, eb, sc


def build_context_graph(
    elements: Iterable[Dict[str, Any]],
    genes: Iterable[Dict[str, Any]],
    contacts_bedpe: Optional[Iterable[Dict[str, Any]]] = None,
    eqtl_pairs: Optional[Iterable[Dict[str, Any]]] = None,
    candidate_window_kb: int = 250,
    tad_boundaries: Optional[Dict[str, List[int]]] = None,
    activity: Optional[Dict[str, float]] = None,
) -> SirenGraph:
    """Build the enhancer-promoter-gene graph used by SIREN.

    Parameters mirror spec §5. ``elements``/``genes`` are dicts with at least
    ``id, chrom, start, end`` (genes also ``tss``). ``contacts_bedpe`` rows are
    loose bedpe-style mappings. ``eqtl_pairs`` rows need
    ``element_id, gene_id, coloc_posterior``. ``tad_boundaries`` maps
    chrom -> sorted TAD boundary positions (E3 emitted only when provided).
    ``activity`` maps element_id -> regulatory activity (E4 emitted only when
    provided together with contacts).
    """
    g = SirenGraph()
    elem_by_chrom: Dict[str, List[Tuple[int, int, str]]] = {}
    for el in elements:
        node = GraphNode(
            id=str(el["id"]),
            ntype="element",
            chrom=_norm_chrom(str(el["chrom"])),
            start=int(el["start"]),
            end=int(el["end"]),
            attrs={k: v for k, v in el.items() if k not in {"id", "chrom", "start", "end"}},
        )
        g.nodes.append(node)
        elem_by_chrom.setdefault(node.chrom, []).append((node.start, node.end, node.id))
    gene_by_id: Dict[str, GraphNode] = {}
    for ge in genes:
        node = GraphNode(
            id=str(ge["id"]),
            ntype="gene",
            chrom=_norm_chrom(str(ge["chrom"])),
            tss=int(ge.get("tss", ge.get("start", 0))),
            attrs={k: v for k, v in ge.items() if k not in {"id", "chrom", "start", "end", "tss"}},
        )
        if node.start is None and node.tss is not None:
            node.start = node.tss
            node.end = node.tss
        g.nodes.append(node)
        gene_by_id[node.id] = node

    # ---- E1: distance-constrained candidates (element -> gene TSS) ----
    win = int(candidate_window_kb) * 1000
    genes_by_chrom: Dict[str, List[GraphNode]] = {}
    for gn in gene_by_id.values():
        genes_by_chrom.setdefault(gn.chrom, []).append(gn)
    for chrom, glist in genes_by_chrom.items():
        glist_sorted = sorted(glist, key=lambda x: x.tss or 0)
        tss_list = [x.tss or 0 for x in glist_sorted]
        for el in elem_by_chrom.get(chrom, []):
            estart, eend, eid = el
            emid = (estart + eend) // 2
            lo = bisect.bisect_left(tss_list, emid - win)
            hi = bisect.bisect_right(tss_list, emid + win)
            for gi in range(lo, hi):
                gn = glist_sorted[gi]
                dist = abs((gn.tss or 0) - emid)
                g.edges.append(GraphEdge(src=eid, dst=gn.id, etype="E1", distance=dist,
                                         score=None, provenance=f"E1|window{candidate_window_kb}kb"))

    # ---- E2/E4: physical contacts (+ABC-style weighted variant) ----
    act = activity or {}
    for row in contacts_bedpe or []:
        ca, sa, ea, cb, sb, eb, sc = _parse_bedpe_row(row)
        for anchor_chrom, a_s, a_e in ((ca, sa, ea),):
            other = (cb, sb, eb)
            _link_contact_side(g, elem_by_chrom, gene_by_id, anchor_chrom, a_s, a_e, other, sc, "E2")
            # reverse orientation too (bedpe is unordered)
        _link_contact_side(g, elem_by_chrom, gene_by_id, cb, sb, eb, (ca, sa, ea), sc, "E2")
        if act:
            _link_abc_edges(g, elem_by_chrom, gene_by_id, ca, sa, ea, cb, sb, eb, sc, act)

    # ---- E3: same-TAD co-membership (only when boundaries provided) ----
    if tad_boundaries:
        bounds_by_chrom = { _norm_chrom(k): sorted(int(x) for x in v) for k, v in tad_boundaries.items() }
        def tad_index(chrom: str, pos: int) -> Optional[int]:
            bs = bounds_by_chrom.get(chrom)
            if not bs:
                return None
            return bisect.bisect_right(bs, pos)
        elems_all = [n for n in g.nodes if n.ntype == "element"]
        by_tad: Dict[Tuple[str, int], List[str]] = {}
        for n in elems_all:
            ti = tad_index(n.chrom, (n.start + n.end) // 2) if n.start is not None else None
            if ti is not None:
                by_tad.setdefault((n.chrom, ti), []).append(n.id)
        for (chrom, ti), ids in by_tad.items():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    g.edges.append(GraphEdge(src=a, dst=b, etype="E3", distance=None, score=1.0,
                                             provenance=f"E3|tad:{chrom}:{ti}", ))
                    # stored once; consumers treat E3 as undirected

    # ---- E5: eQTL-colocalization evidence (disjoint-from-labels flag) ----
    for q in eqtl_pairs or []:
        g.edges.append(GraphEdge(
            src=str(q["element_id"]), dst=str(q["gene_id"]), etype="E5",
            distance=None, score=float(q.get("coloc_posterior", 0.0)),
            provenance="E5|eqtl_catalogue|source_disjoint_from_labels=true",
        ))
    return g


def _link_contact_side(
    g: SirenGraph,
    elem_by_chrom: Dict[str, List[Tuple[int, int, str]]],
    gene_by_id: Dict[str, GraphNode],
    chrom: str,
    s: int,
    e: int,
    other: Tuple[str, int, int],
    score: float,
    etype: str,
) -> None:
    o_c, o_s, o_e = other
    # element anchored on this side, gene on the other side
    for es, ee, eid in elem_by_chrom.get(chrom, []):
        if not _overlaps(s, e, es, ee):
            continue
        for gn in gene_by_id.values():
            if gn.chrom != o_c:
                continue
            gs = gn.start if gn.start is not None else gn.tss or 0
            ge_ = gn.end if gn.end is not None else gn.tss or 0
            if _overlaps(o_s, o_e, gs, ge_) or (gs <= (o_s + o_e) // 2 <= ge_):
                dist = abs((gn.tss or 0) - (es + ee) // 2)
                g.edges.append(GraphEdge(src=eid, dst=gn.id, etype=etype, distance=dist,
                                         score=float(score), provenance=f"{etype}|contacts"))


def _link_abc_edges(
    g: SirenGraph,
    elem_by_chrom: Dict[str, List[Tuple[int, int, str]]],
    gene_by_id: Dict[str, GraphNode],
    ca: str, sa: int, ea: int,
    cb: str, sb: int, eb: int,
    contact_score: float,
    activity: Dict[str, float],
) -> None:
    """E4: activity(element) x contact score, element-side -> gene-side."""
    for el_chrom, el_s, el_e, gn_chrom, gn_s, gn_e in (
        (ca, sa, ea, cb, sb, eb), (cb, sb, eb, ca, sa, ea),
    ):
        for es, ee, eid in elem_by_chrom.get(el_chrom, []):
            if not _overlaps(el_s, el_e, es, ee) or eid not in activity:
                continue
            a = float(activity[eid])
            for gn in gene_by_id.values():
                if gn.chrom != gn_chrom:
                    continue
                gs = gn.start if gn.start is not None else gn.tss or 0
                ge_ = gn.end if gn.end is not None else gn.tss or 0
                if _overlaps(gn_s, gn_e, gs, ge_):
                    g.edges.append(GraphEdge(src=eid, dst=gn.id, etype="E4",
                                             distance=abs((gn.tss or 0) - (es + ee) // 2),
                                             score=a * contact_score,
                                             provenance="E4|activity_x_contact"))


def permute_edges_preserving_degree(graph: SirenGraph, seed: int = 0) -> SirenGraph:
    """Degree-preserving edge corruption control (double-edge swaps within each etype).

    Repeatedly selects two random edges of the same type and swaps their
    destinations, rejecting swaps that create self-loops or duplicates.
    Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    out = SirenGraph(nodes=list(graph.nodes), edges=[GraphEdge(**e.to_dict()) for e in graph.edges])
    by_type: Dict[str, List[int]] = {}
    for i, e in enumerate(out.edges):
        by_type.setdefault(e.etype, []).append(i)
    node_ids = {n.id for n in out.nodes}
    for etype, idxs in by_type.items():
        if len(idxs) < 2:
            continue
        attempts = 10 * len(idxs)
        while attempts > 0:
            attempts -= 1
            ia, ib = rng.choice(idxs), rng.choice(idxs)
            if ia == ib:
                continue
            ea, eb = out.edges[ia], out.edges[ib]
            new_da, new_db = eb.dst, ea.dst
            if new_da == ea.src or new_db == eb.src:
                continue
            if new_da == ea.dst or new_db == eb.dst:
                continue
            if new_da not in node_ids or new_db not in node_ids:
                continue
            existing = {(e.src, e.dst) for e in out.edges}
            if (ea.src, new_da) in existing or (eb.src, new_db) in existing:
                continue
            ea.dst, eb.dst = new_da, new_db
            ea.provenance += "|permuted"
            eb.provenance += "|permuted"
    return out


def to_pyg(graph: SirenGraph) -> Any:
    """Convert to ``torch_geometric.data.HeteroData`` (guarded import)."""
    try:
        from torch_geometric.data import HeteroData  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "to_pyg richiede torch_geometric installato (pip install torch_geometric); "
            f"errore originale: {exc}"
        ) from exc
    hd = HeteroData()
    for ntype in ("element", "gene"):
        ids = [n.id for n in graph.nodes if n.ntype == ntype]
        hd[ntype].num_nodes = len(ids)
        hd[ntype].id = ids  # kept as plain list attribute for traceability
    id_to_ntype = {n.id: n.ntype for n in graph.nodes}
    for etype in graph.edge_types():
        srcs: List[int] = []
        dsts: List[int] = []
        counters: Dict[str, Dict[str, int]] = {}
        for nt in ("element", "gene"):
            counters[nt] = {nid: i for i, nid in enumerate([n.id for n in graph.nodes if n.ntype == nt])}
        for e in graph.edges:
            if e.etype != etype:
                continue
            st, dt = id_to_ntype.get(e.src), id_to_ntype.get(e.dst)
            if st is None or dt is None:
                continue
            srcs.append(counters[st][e.src])
            dsts.append(counters[dt][e.dst])
        rel = f"{etype}"
        hd["element", rel, "gene"].edge_index = [srcs, dsts]
    return hd
