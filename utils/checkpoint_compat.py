# -*- coding: utf-8 -*-
"""Load older finetuned checkpoints into the current CausalVQAModel without shape errors.

Handles common RAD / M3AE drift:
- modality_type_embeddings: [2, H] -> [4, H] (pad extra rows from text/image rows)
- vqa_head: narrow MLP (1536->1536) -> wide (1536->3072) per current vqa_module
- vqa_head last Linear: class count or in_features mismatch (trim/pad rows, widen input)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def _first_vqa_linear_out_dim(model_sd: Dict[str, torch.Tensor]) -> Optional[int]:
    w = model_sd.get("vqa_head.0.weight")
    if w is None or w.dim() != 2:
        return None
    return int(w.shape[0])


def _last_vqa_classifier_weight_key(model_sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """Return vqa_head.<i>.weight for the final Linear (in_features == first Linear out_features)."""
    h = _first_vqa_linear_out_dim(model_sd)
    if h is None:
        return None
    best_k: Optional[str] = None
    best_i = -1
    for k, t in model_sd.items():
        if not (k.startswith("vqa_head.") and k.endswith(".weight")):
            continue
        if k == "vqa_head.0.weight":
            continue
        if t.dim() != 2:
            continue
        if int(t.shape[1]) != h:
            continue
        m = re.match(r"^vqa_head\.(\d+)\.weight$", k)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > best_i:
            best_i = idx
            best_k = k
    return best_k


def build_loadable_state_dict(
    checkpoint: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Return a state_dict subset that matches ``model`` shapes, with legacy remaps applied."""
    ckpt = _strip_module_prefix(checkpoint)
    model_sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    notes: List[str] = []
    skipped: List[str] = []

    last_w_key = _last_vqa_classifier_weight_key(model_sd)
    last_b_key = last_w_key.replace("weight", "bias") if last_w_key else None

    def _ck_val(mk: str) -> Optional[torch.Tensor]:
        if mk in ckpt:
            return ckpt[mk]
        # 最后一层 index 可能差 1（一侧有 head Dropout 时多一层）
        if last_w_key and mk == last_w_key:
            for delta in (1, -1):
                alt = re.sub(
                    r"\.(\d+)\.weight$",
                    lambda m, d=delta: f".{int(m.group(1)) + d}.weight",
                    mk,
                )
                if alt in ckpt:
                    return ckpt[alt]
        if last_b_key and mk == last_b_key:
            for delta in (1, -1):
                alt = re.sub(
                    r"\.(\d+)\.bias$",
                    lambda m, d=delta: f".{int(m.group(1)) + d}.bias",
                    mk,
                )
                if alt in ckpt:
                    return ckpt[alt]
        return None

    for mk, mv in model_sd.items():
        cv = _ck_val(mk)
        if cv is None:
            continue
        if not isinstance(cv, torch.Tensor) or not isinstance(mv, torch.Tensor):
            continue
        if cv.shape == mv.shape:
            out[mk] = cv
            continue

        # --- modality_type_embeddings: 2 rows -> 4 rows ---
        if mk == "modality_type_embeddings.weight":
            if cv.dim() == 2 and mv.dim() == 2 and cv.shape[1] == mv.shape[1]:
                if cv.shape[0] == 2 and mv.shape[0] == 4:
                    t = mv.clone()
                    t[:2] = cv
                    t[2] = cv[0]
                    t[3] = cv[1]
                    out[mk] = t
                    notes.append(f"{mk}: modality rows 2 -> 4 (dup text/image)")
                    continue
                if cv.shape[0] < mv.shape[0]:
                    t = mv.clone()
                    t[: cv.shape[0]] = cv
                    notes.append(f"{mk}: padded modality rows {cv.shape[0]} -> {mv.shape[0]}")
                    out[mk] = t
                    continue

        # --- First MLP: [1536,1536] -> [3072,1536] (hidden*2 -> hidden*4) ---
        if mk == "vqa_head.0.weight":
            if cv.shape == (1536, 1536) and mv.shape == (3072, 1536):
                w = mv.clone()
                w[:1536] = cv
                w[1536:] = cv
                out[mk] = w
                notes.append(f"{mk}: first Linear out_features 1536 -> 3072 (dup block)")
                continue
        if mk == "vqa_head.0.bias":
            if cv.shape[0] == 1536 and mv.shape[0] == 3072:
                b = mv.clone()
                b[:1536] = cv
                b[1536:] = cv
                out[mk] = b
                notes.append(f"{mk}: first Linear bias expanded 1536 -> 3072")
                continue

        # --- LayerNorm after first Linear: [1536] -> [3072] ---
        if mk in ("vqa_head.1.weight", "vqa_head.1.bias"):
            if cv.shape[0] == 1536 and mv.shape[0] == 3072:
                t = mv.clone()
                t[:1536] = cv
                t[1536:] = cv
                out[mk] = t
                notes.append(f"{mk}: LayerNorm dim 1536 -> 3072")
                continue

        # --- Final classifier: in_features 1536 -> 3072 and/or num_classes mismatch ---
        if last_w_key and mk == last_w_key:
            if cv.dim() == 2 and mv.dim() == 2 and cv.shape[1] == 1536 and mv.shape[1] == 3072:
                n_new, d_new = mv.shape
                n_ck, d_ck = cv.shape
                w = mv.clone()
                n_copy = min(n_ck, n_new)
                w[:n_copy, :d_ck] = cv[:n_copy]
                w[:n_copy, d_ck:] = w[:n_copy, :d_ck]
                out[mk] = w
                notes.append(
                    f"{mk}: classifier {tuple(cv.shape)} -> {tuple(mv.shape)} "
                    f"(rows={n_copy}, dup input tail)"
                )
                continue
            if cv.dim() == 2 and mv.dim() == 2 and cv.shape[1] == mv.shape[1] and cv.shape[0] != mv.shape[0]:
                n_new, d = mv.shape
                n_ck = cv.shape[0]
                w = mv.clone()
                n_copy = min(n_ck, n_new)
                w[:n_copy] = cv[:n_copy]
                notes.append(f"{mk}: class dim {n_ck} -> {n_new} (trim/pad rows={n_copy})")
                out[mk] = w
                continue

        if last_b_key and mk == last_b_key:
            if cv.shape[0] != mv.shape[0]:
                b = mv.clone()
                n_copy = min(cv.shape[0], mv.shape[0])
                b[:n_copy] = cv[:n_copy]
                out[mk] = b
                notes.append(f"{mk}: bias len {cv.shape[0]} -> {mv.shape[0]}")
                continue

        skipped.append(f"{mk}: ckpt{tuple(cv.shape)} vs model{tuple(mv.shape)}")

    return out, notes + ([f"skipped incompatible: {len(skipped)}"] if skipped else [])


def load_model_state_dict_relaxed(
    model: nn.Module,
    checkpoint: Dict[str, torch.Tensor],
    logger=None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``load_state_dict(strict=False)`` after shape-safe remaps; logs notes/skips."""
    loadable, notes = build_loadable_state_dict(checkpoint, model)
    log = logger.info if logger is not None else print
    for line in notes:
        if line.startswith("skipped incompatible"):
            log(f"[checkpoint_compat] {line}")
            continue
        log(f"[checkpoint_compat] {line}")
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return missing, unexpected
