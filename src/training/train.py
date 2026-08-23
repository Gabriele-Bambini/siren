"""Trainer SIREN (C-TRAIN): Huber(delta) + ranking intra-locus con PU-mask.

Loop di training con early stopping sulla Pearson del validation (``val_pcc_S1``;
patience 10, max 200 epoche di default), logging JSONL per epoca in
``results/logs/train_<seed>.jsonl`` e checkpoint best+last in
``results/checkpoints/`` (percorsi relativi alla root repo, sovrascrivibili via
cfg). L'import pesante di torch e' protetto: il modulo si importa anche senza
torch/GPU; le funzioni di training sollevano un errore chiaro in quel caso.

Contratto atteso (flessibile verso i moduli fratelli):
- ``loaders``: ``{'train': dl, 'val': dl}``, oppure ``(train_dl, val_dl)``, oppure
  un solo loader (nessuna validazione => early stopping disattivato);
- ogni batch e' idealmente un ``Mapping`` di tensori con chiavi riconosciute tra
  ``y/delta/target/label`` (signed log2 LFC), ``tested/is_tested/test_mask``
  (PU-mask: le coppie non testate NON entrano nella loss) e ``locus_id/locus/group``
  (gruppi per il ranking intra-locus); tollerati anche oggetti con attributi analoghi;
- ``model(batch)`` ritorna un tensore ``(N,)`` oppure un ``Mapping`` contenente la
  predizione tra le chiavi ``y_pred/pred/prediction/...``; con
  ``cfg['model_forward'] = 'kwargs'`` si usa invece ``model(**batch)``.

API pubblica: :func:`train_model` (ritorna ``best_state``, lo state_dict migliore),
:func:`set_seed`, :func:`huber_loss`, :func:`intra_locus_ranking_loss`,
:func:`pearson_cc`.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # import pesante protetto: i test CPU girano anche senza torch/GPU
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ImportError as _exc:  # pragma: no cover - ambiente senza PyTorch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment,misc]
    Tensor = Any  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR: Optional[str] = f"{type(_exc).__name__}: {_exc}"
else:
    _TORCH_IMPORT_ERROR = None

if TYPE_CHECKING:  # solo tipizzazione: nessuna dipendenza runtime da moduli in costruzione
    from src.data.schemas import PairLabel  # noqa: F401

__all__ = [
    "train_model",
    "set_seed",
    "huber_loss",
    "intra_locus_ranking_loss",
    "pearson_cc",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG_DIR = _REPO_ROOT / "results" / "logs"
_DEFAULT_CKPT_DIR = _REPO_ROOT / "results" / "checkpoints"

_TARGET_KEYS: Tuple[str, ...] = ("y", "delta", "target", "targets", "label", "labels", "y_true", "lfc")
_PRED_KEYS: Tuple[str, ...] = ("y_pred", "pred", "prediction", "predictions", "y_hat", "score", "scores", "output", "out", "logits")
_TESTED_KEYS: Tuple[str, ...] = ("tested", "is_tested", "test_mask", "tested_mask", "pu_mask")
_LOCUS_KEYS: Tuple[str, ...] = ("locus_id", "locus_ids", "locus", "loci", "group", "group_id")
_TRAIN_KEYS: Tuple[str, ...] = ("train", "training")
_VAL_KEYS: Tuple[str, ...] = ("val", "valid", "validation", "dev")


def _require_torch() -> None:
    """Solleva un errore chiaro se PyTorch non e' disponibile."""
    if torch is None:
        raise ImportError(
            "PyTorch non disponibile: il trainer SIREN richiede 'torch' "
            f"(es. 'pip install torch --index-url https://download.pytorch.org/whl/cpu'). Dettaglio import: {_TORCH_IMPORT_ERROR}"
        )


def set_seed(seed: int) -> None:
    """Fissa i seed di ``random``/``numpy`` (e ``torch``/CUDA se presenti)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _cfg_get(cfg: Optional[Any], key: str, default: Any) -> Any:
    """Legge ``key`` da un cfg Mapping oppure attribute-style, con default."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


@dataclass(frozen=True)
class _Settings:
    """Iperparametri del trainer risolti dal cfg (Mapping o oggetto)."""

    lr: float
    weight_decay: float
    optimizer: str
    max_epochs: int
    patience: int
    huber_delta: float
    lambda_rank: float
    grad_clip_norm: Optional[float]
    device: str
    model_forward: str
    ignored_index: Optional[float]
    log_dir: Path
    checkpoint_dir: Path
    resume: bool

    @classmethod
    def from_cfg(cls, cfg: Optional[Any]) -> "_Settings":
        ignored_raw = _cfg_get(cfg, "ignored_index", None)
        clip_raw = _cfg_get(cfg, "grad_clip_norm", None)
        return cls(
            lr=float(_cfg_get(cfg, "lr", 1e-3)),
            weight_decay=float(_cfg_get(cfg, "weight_decay", 0.0)),
            optimizer=str(_cfg_get(cfg, "optimizer", "adamw")).lower(),
            max_epochs=int(_cfg_get(cfg, "max_epochs", 200)),
            patience=int(_cfg_get(cfg, "patience", 10)),
            huber_delta=float(_cfg_get(cfg, "huber_delta", 1.0)),
            lambda_rank=float(_cfg_get(cfg, "lambda_rank", 0.3)),
            grad_clip_norm=None if clip_raw is None else float(clip_raw),
            device=str(_cfg_get(cfg, "device", "auto")),
            model_forward=str(_cfg_get(cfg, "model_forward", "batch")).lower(),
            ignored_index=None if ignored_raw is None else float(ignored_raw),
            log_dir=Path(str(_cfg_get(cfg, "log_dir", _DEFAULT_LOG_DIR))),
            checkpoint_dir=Path(str(_cfg_get(cfg, "checkpoint_dir", _DEFAULT_CKPT_DIR))),
            resume=bool(_cfg_get(cfg, "resume", False)),
        )


def _split_loaders(loaders: Any) -> Tuple[Any, Optional[Any]]:
    """Normalizza ``loaders`` in ``(train_loader, val_loader|None)``."""
    if isinstance(loaders, Mapping):
        train = next((loaders[k] for k in _TRAIN_KEYS if k in loaders and loaders[k] is not None), None)
        val = next((loaders[k] for k in _VAL_KEYS if k in loaders and loaders[k] is not None), None)
        if train is None and len(loaders) == 1:
            train = next(iter(loaders.values()))
        if train is None:
            raise ValueError(f"loader 'train' non trovato in loaders={list(loaders)}")
        return train, val
    if isinstance(loaders, (list, tuple)):
        if len(loaders) == 0:
            raise ValueError("loaders vuoto: serve almeno il loader di train")
        return loaders[0], (loaders[1] if len(loaders) > 1 else None)
    return loaders, None  # singolo loader/iterabile


def _pick(batch: Any, keys: Sequence[str]) -> Any:
    """Primo valore non-Nullo tra ``keys`` in un batch Mapping o attribute-style."""
    if isinstance(batch, Mapping):
        for key in keys:
            if key in batch and batch[key] is not None:
                return batch[key]
        return None
    for key in keys:
        value = getattr(batch, key, None)
        if value is not None:
            return value
    return None


def _batch_to_device(batch: Any, device: Any) -> Any:
    """Sposta su ``device`` i tensori di un batch Mapping/lista (ricorsione shallow)."""
    if isinstance(batch, Mapping):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_batch_to_device(v, device) for v in batch)
    if torch is not None and torch.is_tensor(batch):
        return batch.to(device)
    return batch


def _as_vector(x: Any, device: Any) -> "Tensor":
    """Converte ``x`` in tensore float 1-D su ``device``."""
    if not (torch.is_tensor(x)):
        x = np.asarray(x, dtype=np.float32)
    x = torch.as_tensor(x).to(device).float()
    if x.ndim > 1 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x.reshape(-1)


def _extract_supervision(batch: Any, device: Any, ignored_index: Optional[float]) -> Tuple["Tensor", "Tensor", Optional[List[Any]]]:
    """Estrae ``(target, mask_valida, loci)`` dal batch applicando la PU-mask.

    La mask combina il flag ``tested`` (se presente), l'esclusione dei target non
    finiti (NaN/inf) e del valore sentinella ``ignored_index`` (default: assente).
    """
    y_raw = _pick(batch, _TARGET_KEYS)
    if y_raw is None:
        raise KeyError(f"Nessuna chiave target nel batch; attese una tra {', '.join(_TARGET_KEYS)}")
    y = _as_vector(y_raw, device)

    tested_raw = _pick(batch, _TESTED_KEYS)
    if tested_raw is None:
        valid = torch.ones_like(y, dtype=torch.bool)
    else:
        tested = torch.as_tensor(tested_raw).to(device).reshape(-1).bool()
        if tested.numel() != y.numel():
            raise ValueError(f"'tested' ({tested.numel()}) disallineato col target ({y.numel()})")
        valid = tested

    valid = valid & torch.isfinite(y)
    if ignored_index is not None:
        valid = valid & (y != float(ignored_index))

    loci_raw = _pick(batch, _LOCUS_KEYS)
    loci: Optional[List[Any]] = None
    if loci_raw is not None:
        if torch.is_tensor(loci_raw):
            loci = [v for v in loci_raw.reshape(-1).tolist()]
        elif isinstance(loci_raw, Iterable) and not isinstance(loci_raw, (str, bytes)):
            loci = list(loci_raw)
        else:
            loci = [loci_raw]
        if len(loci) != y.numel():
            loci = None  # lunghezza incoerente: niente raggruppamento
    return y, valid, loci


def _forward_model(model: Any, batch: Any, settings: "_Settings") -> Any:
    """Chiama il modello secondo la convenzione configurata."""
    if settings.model_forward == "kwargs":
        if not isinstance(batch, Mapping):
            raise TypeError("model_forward='kwargs' richiede batch di tipo Mapping")
        return model(**batch)
    return model(batch)


def _extract_prediction(output: Any) -> "Tensor":
    """Estrae la predizione 1-D dall'output del modello (tensore/mapping/tupla)."""
    if torch.is_tensor(output):
        if output.ndim > 1 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output.reshape(-1)
    if isinstance(output, Mapping):
        picked = _pick(output, _PRED_KEYS)
        if picked is None:
            raise KeyError(f"Nessuna chiave predizione in output {list(output)}; attese una tra {', '.join(_PRED_KEYS)}")
        return _extract_prediction(picked)
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return _extract_prediction(item)
    raise TypeError(f"Output modello non interpretabile: {type(output).__name__}")


def huber_loss(pred: "Tensor", target: "Tensor", mask: Optional["Tensor"] = None, delta: float = 1.0) -> "Tensor":
    """Huber(delta) media solo sulle entrate valide (PU-mask); 0 se nessuna valida."""
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    mask = mask.reshape(-1).bool()
    if not bool(mask.any()):
        return pred.sum() * 0.0
    diff = (pred.reshape(-1)[mask] - target.reshape(-1)[mask]).abs()
    quad = diff.clamp(max=delta)
    return (0.5 * quad.pow(2) + delta * (diff - quad)).mean()


def intra_locus_ranking_loss(
    pred: "Tensor",
    target: "Tensor",
    loci: Optional[Sequence[Any]],
    mask: Optional["Tensor"] = None,
) -> "Tensor":
    """Pairwise ranking (softplus) solo entro locus, sulle coppie testate.

    Per ogni locus con >=2 coppie valide ordina le predizioni come i target:
    contributo medio ``softplus(p_j - p_i)`` sulle coppie (i, j) con t_i > t_j.
    Se ``loci`` e' assente o nessun gruppo e' utilizzabile ritorna 0.
    """
    if loci is None:
        return pred.sum() * 0.0
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    mask_list = mask.reshape(-1).bool().tolist()
    p = pred.reshape(-1)
    t = target.reshape(-1)

    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, (keep, group) in enumerate(zip(mask_list, list(loci))):
        if keep:
            groups[group].append(i)

    total = p.new_zeros(())
    n_groups = 0
    for idx in groups.values():
        if len(idx) < 2:
            continue
        pi, pj = p[idx].unsqueeze(1), p[idx].unsqueeze(0)
        ti, tj = t[idx].unsqueeze(1), t[idx].unsqueeze(0)
        want_more = ti > tj  # ci aspettiamo p_i > p_j dove t_i > t_j
        if not bool(want_more.any()):
            continue
        total = total + F.softplus(pj - pi)[want_more].mean()
        n_groups += 1
    if n_groups == 0:
        return pred.sum() * 0.0
    return total / n_groups


def pearson_cc(pred: Any, target: Any) -> float:
    """Pearson robusto (NaN-safe, varianza nulla -> NaN)."""
    a = np.asarray(pred, dtype=np.float64).reshape(-1)
    b = np.asarray(target, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return float("nan")
    sa, sb = float(a.std()), float(b.std())
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    cov = float(np.mean((a - a.mean()) * (b - b.mean())))
    return cov / (sa * sb)


def _run_epoch(loader: Iterable[Any], model: Any, optimizer: Any, device: Any, s: _Settings) -> Dict[str, float]:
    """Un'epoca di training: media delle loss pesata sulle coppie valide."""
    model.train()
    totals = {"loss": 0.0, "huber": 0.0, "rank": 0.0}
    weight = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        y, valid, loci = _extract_supervision(batch, device, s.ignored_index)
        pred = _extract_prediction(_forward_model(model, batch, s)).to(device)
        if pred.numel() != y.numel():
            raise ValueError(f"Predizione ({pred.numel()}) e target ({y.numel()}) disallineati nel batch")
        hub = huber_loss(pred, y, valid, s.huber_delta)
        rank = intra_locus_ranking_loss(pred, y, loci, valid) if s.lambda_rank > 0 else pred.sum() * 0.0
        loss = hub + s.lambda_rank * rank

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if s.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], s.grad_clip_norm)
        optimizer.step()

        bs = int(valid.sum()) or 1
        totals["loss"] += float(loss.detach())
        totals["huber"] += float(hub.detach())
        totals["rank"] += float(rank.detach())
        weight += bs
    if weight == 0:
        return {k: 0.0 for k in totals}
    return {k: v / weight for k, v in totals.items()}


def _evaluate(model: Any, loader: Iterable[Any], device: Any, s: _Settings) -> Dict[str, Optional[float]]:
    """Valutazione: PCC sulle coppie testate (PU-mask) + Huber mascherata."""
    model.eval()
    preds: List[np.ndarray] = []
    targs: List[np.ndarray] = []
    valids: List[np.ndarray] = []
    hub_sum, hub_n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            y, valid, _ = _extract_supervision(batch, device, s.ignored_index)
            pred = _extract_prediction(_forward_model(model, batch, s)).to(device)
            if pred.numel() != y.numel():
                raise ValueError(f"Predizione ({pred.numel()}) e target ({y.numel()}) disallineati nel batch")
            if bool(valid.any()):
                hub_sum += float(huber_loss(pred, y, valid, s.huber_delta)) * int(valid.sum())
                hub_n += int(valid.sum())
            preds.append(pred.detach().cpu().numpy())
            targs.append(y.detach().cpu().numpy())
            valids.append(valid.detach().cpu().numpy())
    if not preds:
        return {"val_pcc_S1": None, "val_huber": None}
    p = np.concatenate(preds)
    t = np.concatenate(targs)
    vmask = np.concatenate(valids).astype(bool)
    pcc = pearson_cc(p[vmask], t[vmask]) if int(vmask.sum()) >= 2 else float("nan")
    return {"val_pcc_S1": pcc, "val_huber": (hub_sum / hub_n) if hub_n else None}


def _snapshot(model: Any) -> Dict[str, Any]:
    """Copia CPU dello state_dict corrente (best_state portabile)."""
    state = model.state_dict()
    return {
        k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else copy.deepcopy(v))
        for k, v in state.items()
    }


def _save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    try:
        torch.save(payload, path)
    except OSError as exc:  # non blocca il training: best_state resta in memoria
        print(f"[trainer] salvataggio checkpoint fallito ({path}): {exc}", file=sys.stderr)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Appende una riga JSON per epoca (non-finito -> null per JSON valido)."""

    def _jsonable(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    line = json.dumps({k: _jsonable(v) for k, v in record.items()}, ensure_ascii=False, default=str)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        print(f"[trainer] scrittura log fallita ({path}): {exc}", file=sys.stderr)


def _current_lr(optimizer: Any) -> float:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except (KeyError, IndexError, TypeError):
        return float("nan")


def train_model(model: Any, loaders: Any, cfg: Optional[Any] = None, seed: int = 0) -> Dict[str, Any]:
    """Allena ``model`` e ritorna ``best_state`` (state_dict migliore su val_pcc_S1).

    Args:
        model: modulo PyTorch (spostato sul device scelto).
        loaders: mapping/sequenza di loader (vedi docstring del modulo).
        cfg: Mapping od oggetto con chiavi opzionali ``lr, weight_decay, optimizer,
            max_epochs (200), patience (10), huber_delta (1.0), lambda_rank (0.3),
            grad_clip_norm, device ('auto'), model_forward ('batch'|'kwargs'),
            ignored_index, log_dir, checkpoint_dir, resume``.
        seed: seed deterministico (random/numpy/torch) e suffisso file log/ckpt.

    Returns:
        Copia CPU dello state_dict con la miglior Pearson di validation (S1);
        lo stato finale se la validation e' assente o mai migliorata.
    """
    _require_torch()
    set_seed(seed)
    s = _Settings.from_cfg(cfg)

    if s.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(s.device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if s.optimizer == "adam":
        optimizer: Any = torch.optim.Adam(params, lr=s.lr, weight_decay=s.weight_decay)
    elif s.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=s.lr, weight_decay=s.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=s.lr, weight_decay=s.weight_decay)

    train_loader, val_loader = _split_loaders(loaders)
    s.log_dir.mkdir(parents=True, exist_ok=True)
    s.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = s.log_dir / f"train_{seed}.jsonl"
    best_path = s.checkpoint_dir / f"best_seed{seed}.pt"
    last_path = s.checkpoint_dir / f"last_seed{seed}.pt"

    start_epoch, stale = 0, 0
    best_pcc = float("-inf")
    best_epoch = -1
    last_epoch = start_epoch - 1
    best_state = _snapshot(model)

    if s.resume and last_path.exists():
        try:
            ckpt = torch.load(last_path, map_location=device)
            if ckpt.get("model_state"):
                model.load_state_dict(ckpt["model_state"])
            if ckpt.get("optimizer_state"):
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", 0))
            prev_best = ckpt.get("best_val_pcc_S1")
            best_pcc = float(prev_best) if prev_best is not None else float("-inf")
            stale = int(ckpt.get("patience_counter", 0))
            print(f"[train seed={seed}] ripreso dall'epoca {start_epoch} (last checkpoint)")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"[train seed={seed}] resume fallito ({exc}); riparto da zero", file=sys.stderr)
            start_epoch, stale, best_pcc = 0, 0, float("-inf")

    for epoch in range(start_epoch, s.max_epochs):
        last_epoch = epoch
        t0 = time.time()
        train_stats = _run_epoch(train_loader, model, optimizer, device, s)
        val_stats = _evaluate(model, val_loader, device, s) if val_loader is not None else {}
        val_pcc = val_stats.get("val_pcc_S1")

        improved = val_pcc is not None and math.isfinite(val_pcc) and val_pcc > best_pcc
        if improved:
            best_pcc = float(val_pcc)
            best_epoch = epoch
            stale = 0
            best_state = _snapshot(model)
            _save_checkpoint(best_path, {"kind": "best", "seed": seed, "epoch": epoch, "val_pcc_S1": best_pcc, "model_state": best_state})
        elif val_loader is not None:
            stale += 1

        _save_checkpoint(last_path, {
            "kind": "last",
            "seed": seed,
            "epoch": epoch + 1,
            "val_pcc_S1": val_pcc,
            "best_val_pcc_S1": best_pcc if math.isfinite(best_pcc) else None,
            "patience_counter": stale,
            "model_state": _snapshot(model),
            "optimizer_state": optimizer.state_dict(),
        })
        _append_jsonl(log_path, {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed,
            "epoch": epoch,
            "lr": _current_lr(optimizer),
            "train_loss": round(train_stats["loss"], 6),
            "train_huber": round(train_stats["huber"], 6),
            "train_rank": round(train_stats["rank"], 6),
            "val_pcc_S1": val_pcc,
            "val_huber": val_stats.get("val_huber"),
            "best_val_pcc_S1": best_pcc if math.isfinite(best_pcc) else None,
            "epochs_without_improvement": stale,
            "improved": bool(improved),
            "elapsed_s": round(time.time() - t0, 3),
        })
        if improved:
            print(f"[train seed={seed}] epoch {epoch}: nuovo best val_pcc_S1={best_pcc:.4f}")

        if val_loader is not None and stale >= s.patience:
            print(f"[train seed={seed}] early stopping all'epoca {epoch} (patience={s.patience}; best={best_pcc:.4f} @ epoca {best_epoch})")
            break

    if best_epoch < 0:  # mai migliorato o nessuna validazione: usa lo stato finale
        best_state = _snapshot(model)
        fallback_epoch = last_epoch if last_epoch >= 0 else start_epoch
        _save_checkpoint(best_path, {"kind": "best", "seed": seed, "epoch": fallback_epoch, "val_pcc_S1": None, "model_state": best_state})
    return best_state
