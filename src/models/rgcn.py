"""R-GCN over the E-P-G graph with an MLP head predicting a scalar LFC per pair.

Componente C-RGCN del progetto SIREN: due layer ``RGCNConv`` sulle relazioni
['E1', 'E2', 'E3', 'E4', 'E5'] prodotte da C-GRAPH, seguito da una testa MLP
che mappa le embedding dei nodi (element, gene) in un valore LFC scalare.

Gli import pesanti (torch / torch_geometric) sono protetti: l'import del modulo
resta possibile su CPU senza GPU/torch, mentre la costruzione o l'esecuzione
del modello richiede che entrambe le librerie siano installate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = [
    "SirenRGCN",
    "build_model",
    "DEFAULT_RELATIONS",
    "MAX_PARAMS",
    "TORCH_AVAILABLE",
]

# --- Import pesanti protetti -------------------------------------------------
try:
    import torch
    from torch import Tensor
    from torch_geometric.nn import RGCNConv

    TORCH_AVAILABLE: bool = True
    IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - ramo attivo solo senza torch/PyG.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    RGCNConv = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    IMPORT_ERROR = _exc

DEFAULT_RELATIONS: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
MAX_PARAMS: int = 2_000_000


if TORCH_AVAILABLE:

    class SirenRGCN(torch.nn.Module):
        """Two-layer R-GCN encoder + MLP head for signed LFC prediction.

        Args:
            in_dim: input feature dimension of E/P/G nodes.
            hidden_dim: hidden width of both RGCNConv layers (default 128).
            num_relations: number of typed relations (default 5: E1..E5).
            dropout: dropout probability used between layers and in the head.
        """

        def __init__(
            self,
            in_dim: int = 32,
            hidden_dim: int = 128,
            num_relations: int = len(DEFAULT_RELATIONS),
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if in_dim <= 0 or hidden_dim <= 0 or num_relations <= 0:
                raise ValueError("in_dim, hidden_dim e num_relations devono essere > 0.")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout deve essere in [0, 1).")

            self.in_dim = int(in_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_relations = int(num_relations)

            self.conv1 = RGCNConv(self.in_dim, self.hidden_dim, self.num_relations)
            self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim, self.num_relations)
            self.dropout = torch.nn.Dropout(float(dropout))
            # Pair head su [h_e, h_g, h_e * h_g] -> scalare LFC.
            self.head = torch.nn.Sequential(
                torch.nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(self.hidden_dim, 1),
            )

            n_params = sum(p.numel() for p in self.parameters())
            assert n_params < MAX_PARAMS, (
                f"SirenRGCN ha {n_params} parametri, oltre il budget di {MAX_PARAMS}."
            )
            self.param_count = n_params

        def encode(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
            """Return node embeddings [num_nodes, hidden_dim] after two R-GCN layers.

            ``edge_type`` deve essere un LongTensor con id di relazione in
            ``[0, num_relations)`` (ordine atteso: DEFAULT_RELATIONS).
            """
            h = self.dropout(torch.relu(self.conv1(x, edge_index, edge_type)))
            h = torch.relu(self.conv2(h, edge_index, edge_type))
            return h

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_type: Tensor,
            element_idx: Tensor,
            gene_idx: Tensor,
        ) -> Tensor:
            """Predict one scalar LFC per (element, gene) pair.

            Args:
                x: [num_nodes, in_dim] node features (enhancers/promoters/genes).
                edge_index: [2, num_edges] graph connectivity.
                edge_type: [num_edges] relation ids (long), coerenti con le relazioni usate in fase di costruzione.
                element_idx: [num_pairs] indici dei nodi enhancer/promoter.
                gene_idx: [num_pairs] indici dei nodi gene, allineati a element_idx.

            Returns:
                Tensor [num_pairs] con gli LFC previsti (non calibrati; la
                calibrazione e' responsabilita' di C-CONF).
            """
            h = self.encode(x, edge_index, edge_type)
            e = h[element_idx]
            g = h[gene_idx]
            z = torch.cat([e, g, e * g], dim=-1)
            return self.head(z).squeeze(-1)


def build_model(cfg: Optional[Dict[str, Any]] = None) -> "SirenRGCN":
    """Build :class:`SirenRGCN` from a config dict.

    Chiavi riconosciute (tutte opzionali): ``in_dim`` (32), ``hidden_dim``
    (128), ``dropout`` (0.1), ``relations`` (['E1','E2','E3','E4','E5']);
    chiavi sconosciute vengono ignorate.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "src.models.rgcn: costruire SirenRGCN richiede 'torch' e 'torch_geometric'. "
            "Installali (es. `pip install torch torch_geometric`) e riprova."
            + (f" Errore originale import: {IMPORT_ERROR}" if IMPORT_ERROR else "")
        )
    conf: Dict[str, Any] = dict(cfg or {})

    rel_cfg = conf.get("relations", DEFAULT_RELATIONS)
    relations: Tuple[str, ...] = (rel_cfg,) if isinstance(rel_cfg, str) else tuple(rel_cfg)
    if len(set(relations)) != len(relations):
        raise ValueError(f"relations contiene duplicati: {relations!r}")

    return SirenRGCN(
        in_dim=int(conf.get("in_dim", 32)),
        hidden_dim=int(conf.get("hidden_dim", 128)),
        num_relations=len(relations),
        dropout=float(conf.get("dropout", 0.1)),
    )
