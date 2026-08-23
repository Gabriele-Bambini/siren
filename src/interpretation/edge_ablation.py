"""C-INTERP — edge-type ablation and HTML evidence dossier for SIREN.

``run_edge_ablation`` requires a callable ``metric_fn(graph)->float`` plus the
trained model object only for signature compatibility; when PyTorch-backed
evaluation is requested without torch installed it fails loudly and
informatively. ``generate_dossier_html`` is pure standard library.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

__all__ = ["AblationRow", "run_edge_ablation", "generate_dossier_html"]


@dataclass
class AblationRow:
    """One row of the ablation table."""

    removed_edge_type: str
    baseline_metric: float
    ablated_metric: float
    delta_metric: float  # baseline - ablated (positive => edge type was useful)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "removed_edge_type": self.removed_edge_type,
            "baseline_metric": self.baseline_metric,
            "ablated_metric": self.ablated_metric,
            "delta_metric": self.delta_metric,
        }


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "run_edge_ablation con modello torch richiede PyTorch installato "
            "(pip install torch). Per analisi offline usa metric_fn su grafi "
            f"serializzati senza passare dal modello. Errore originale: {exc}"
        ) from exc


def run_edge_ablation(
    model: Any,
    metric_fn: Callable[[Any], float],
    graph: Any,
    cfg: Optional[Dict[str, Any]] = None,
    require_torch_check: bool = True,
) -> List[AblationRow]:
    """Remove one edge-type at a time and measure the metric drop.

    Parameters
    ----------
    model:
        Trained model object (used only if ``metric_fn`` needs it); kept in the
        signature so callers can close over it inside ``metric_fn``.
    metric_fn:
        Callable mapping a graph-like object to a scalar metric (e.g. PCC on
        the held-out split). Called once on the intact graph and once per
        edge-type removal.
    graph:
        Graph exposing ``edge_types()`` and ``without_edge_type(etype)``
        (e.g. :class:`src.graph.build.SirenGraph`).
    cfg:
        Optional config dict; currently supports ``min_edges_to_ablate`` (int,
        default 1): edge types with fewer edges are skipped.
    """
    cfg = cfg or {}
    min_edges = int(cfg.get("min_edges_to_ablate", 1))
    if require_torch_check and getattr(model, "__module__", "").startswith("torch"):
        _require_torch()

    types = list(graph.edge_types())
    counts: Dict[str, int] = {t: 0 for t in types}
    for e in getattr(graph, "edges", []) or []:
        if e.etype in counts:
            counts[e.etype] += 1

    baseline = float(metric_fn(graph))
    rows: List[AblationRow] = []
    for etype in sorted(types):
        if counts.get(etype, 0) < min_edges:
            continue
        ablated_metric = float(metric_fn(graph.without_edge_type(etype)))
        rows.append(AblationRow(
            removed_edge_type=etype,
            baseline_metric=baseline,
            ablated_metric=ablated_metric,
            delta_metric=baseline - ablated_metric,
        ))
    rows.sort(key=lambda r: r.delta_metric, reverse=True)
    return rows


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def generate_dossier_html(
    locus_id: str,
    evidence: Dict[str, Any],
    out_path: str,
) -> str:
    """Render a minimal self-contained HTML dossier for one locus.

    Expected ``evidence`` keys (all optional but recommended):
    - ``delta_tracks``: {track_name -> value or list of values} (ISD deltas)
    - ``dominant_edges``: iterable of (etype, src, dst, score|str)
    - ``label_support``: str or iterable of str (published/observed support)
    - ``prediction``: {"y_hat": float, "lo": float, "hi": float}
    - ``limitations``: str or iterable of str (defaults provided)
    Returns the absolute path written.
    """
    import os

    parts: List[str] = []
    parts.append("<!DOCTYPE html><html lang='it'><head><meta charset='utf-8'>")
    parts.append(f"<title>SIREN dossier — {_esc(locus_id)}</title>")
    parts.append("<style>body{font-family:system-ui,sans-serif;margin:2rem}"
                 "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
                 "h2{margin-top:1.5rem}.warn{color:#a00}</style></head><body>")
    parts.append(f"<h1>Dossier interpretativo SIREN — locus {_esc(locus_id)}</h1>")
    parts.append("<p class='warn'>Generato automaticamente: i numeri vanno letti insieme alle limitazioni.</p>")

    # --- delta tracks ---
    parts.append("<h2>Delta tracks (silenziamento in-silico)</h2>")
    tracks = evidence.get("delta_tracks") or {}
    if isinstance(tracks, dict) and tracks:
        parts.append("<table><tr><th>track</th><th>&Delta;</th></tr>")
        for name, val in tracks.items():
            if isinstance(val, (list, tuple)):
                shown = ", ".join(f"{float(x):+.3g}" for x in val)
            else:
                shown = f"{float(val):+.3g}"
            parts.append(f"<tr><td>{_esc(name)}</td><td>{_esc(shown)}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p><i>Nessun delta track fornito.</i></p>")

    # --- dominant edges ---
    parts.append("<h2>Archi dominanti</h2>")
    edges = evidence.get("dominant_edges") or []
    if edges:
        parts.append("<table><tr><th>tipo</th><th>src</th><th>dst</th><th>score</th></tr>")
        for e in edges:
            try:
                etype, src, dst, score = e  # type: ignore[misc]
            except Exception:
                parts.append(f"<tr><td colspan='4'>{_esc(e)}</td></tr>")
                continue
            score_s = f"{float(score):.3g}" if isinstance(score, (int, float)) else _esc(score)
            parts.append(f"<tr><td>{_esc(etype)}</td><td>{_esc(src)}</td>"
                         f"<td>{_esc(dst)}</td><td>{_esc(score_s)}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p><i>Nessun arco dominante registrato.</i></p>")

    # --- label support ---
    parts.append("<h2>Supporto da label / letteratura</h2>")
    sup = evidence.get("label_support")
    if isinstance(sup, (list, tuple)):
        parts.append("<ul>" + "".join(f"<li>{_esc(s)}</li>" for s in sup) + "</ul>")
    elif sup:
        parts.append(f"<p>{_esc(sup)}</p>")
    else:
        parts.append("<p><i>Nessun supporto esterno registrato.</i></p>")

    # --- calibrated interval ---
    pred = evidence.get("prediction") or {}
    if {"y_hat", "lo", "hi"} <= set(pred.keys()):
        parts.append("<h2>Predizione calibrata</h2>")
        parts.append(f"<p>&Ycirc; = {float(pred['y_hat']):+.3f} · "
                     f"intervallo conformale [{float(pred['lo']):+.3f}, {float(pred['hi']):+.3f}] "
                     f"(larghezza {float(pred['hi']) - float(pred['lo']):.3f})</p>")

    # --- limitations ---
    limits = evidence.get("limitations") or [
        "Delta di encoder frozen: nessuna causalita' diretta, solo coerenza con perturbazioni simulate.",
        "Gli archi dominanti dipendono dalla topologia del grafo di contesto (contatti parziali).",
        "L'intervallo calibrato vale solo nel contesto/contesto-cellula calibrato.",
    ]
    parts.append("<h2>Limiti</h2><ul>")
    if isinstance(limits, str):
        limits = [limits]
    for lim in limits:
        parts.append(f"<li>{_esc(lim)}</li>")
    parts.append("</ul>")

    parts.append("</body></html>")
    html_text = "\n".join(parts)
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    return out_abs


def summarize_ablation_json(rows: Sequence[AblationRow], out_path: str) -> str:
    """Persist an ablation table as JSON (utility for reports)."""
    import os

    payload = [r.to_dict() for r in rows]
    out_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out_abs
