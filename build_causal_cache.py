import json
import os
from typing import Dict, Optional

import torch
from tqdm import tqdm

from pipeline.causal_signal_builder import StableDOSignalBuilder


@torch.no_grad()
def rebuild_signal_cache(
    model_frozen,
    data_loader,
    tokenizer,
    intervention_bank: Optional[Dict],
    device,
    output_path: str,
    logger=None,
    ema_momentum: float = 0.9,
    ccs_text_de_scale: float = 1.0,
    store_fusion_bank: bool = True,
):
    """
    Rebuild offline causal signal cache using frozen/EMA teacher.
    Cache schema:
      {"items":[{"sample_id": "...", "CCS":..., "HCSS":..., "text_de":..., "visual_ie":..., "fusion_bank"?: [...]}, ...]}
    """
    builder = StableDOSignalBuilder(
        ema_momentum=float(ema_momentum),
        ccs_text_de_scale=float(ccs_text_de_scale),
        store_fusion_bank=bool(store_fusion_bank),
    )
    model_frozen.eval()

    items = _compute_signal_items_for_loader(
        model_frozen=model_frozen,
        data_loader=data_loader,
        tokenizer=tokenizer,
        intervention_bank=intervention_bank,
        device=device,
        builder=builder,
        sample_ratio=1.0,
        desc="rebuild_signal_cache",
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False)

    msg = f"Signal cache rebuilt: {len(items)} items -> {output_path}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _load_cache_items_as_map(cache_path: str) -> Dict[str, Dict]:
    if not cache_path or (not os.path.exists(cache_path)):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("items", []) if isinstance(raw, dict) else []
    out = {}
    for it in items:
        sid = str(it.get("sample_id", "")).strip()
        if sid:
            out[sid] = it
    return out


def _write_cache_map(cache_map: Dict[str, Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    items = list(cache_map.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False)


@torch.no_grad()
def _compute_signal_items_for_loader(
    model_frozen,
    data_loader,
    tokenizer,
    intervention_bank,
    device,
    builder: StableDOSignalBuilder,
    sample_ratio: float,
    desc: str,
):
    items = []
    ratio = max(0.0, min(1.0, float(sample_ratio)))
    for batch in tqdm(data_loader, desc=desc):
        qids_full = batch.get("qid", [])
        bsz_full = len(qids_full)
        if bsz_full == 0:
            continue
        if ratio >= 1.0:
            selected = torch.arange(bsz_full, dtype=torch.long)
        else:
            sel_mask = torch.rand(bsz_full) < ratio
            if not sel_mask.any():
                continue
            selected = torch.nonzero(sel_mask, as_tuple=False).flatten().long()
        if selected.numel() == 0:
            continue

        images = batch["images"][selected].to(device)
        q = batch["questions"]["input_ids"][selected].to(device)
        am = batch["questions"]["attention_mask"][selected].to(device)
        dq_full = batch.get("do_questions", {}).get("input_ids", None)
        dq = dq_full[selected].to(device) if dq_full is not None else None
        q_texts_full = batch.get("question_texts", None)
        q_texts = [q_texts_full[i] for i in selected.tolist()] if q_texts_full is not None else None
        img_paths_full = batch.get("image_paths", None)
        img_paths = [img_paths_full[i] for i in selected.tolist()] if img_paths_full is not None else None
        ans_types_full = batch.get("answer_types", None)
        ans_types = [ans_types_full[i] for i in selected.tolist()] if ans_types_full is not None else None
        targets_full = batch.get("targets", None)
        targets = targets_full[selected].to(device) if targets_full is not None else None
        qids = [qids_full[i] for i in selected.tolist()]

        signals = builder.build_signals(
            model_frozen=model_frozen,
            images=images,
            questions_ids=q,
            attention_mask=am,
            do_questions_ids=dq,
            common_kwargs={"training": False},
            tokenizer=tokenizer,
            intervention_bank=intervention_bank,
            question_texts=q_texts,
            image_paths=img_paths,
            answer_types=ans_types,
            targets=targets,
        )

        ccs = signals["ccs"].detach().cpu().float()
        hcss = signals["hcss"].detach().cpu().float()
        tde = signals["text_de"].detach().cpu().float()
        vie = signals["vis_ie"].detach().cpu().float()
        tie = signals.get("text_ie", torch.zeros_like(ccs)).detach().cpu().float()
        valid_mask = signals.get("valid_mask", torch.ones_like(ccs)).detach().cpu().float()
        error_reason = signals.get("error_reason", [""] * int(ccs.size(0)))
        fb_rows = signals.get("fusion_bank_rows") or []
        bsz = ccs.size(0)
        for i in range(bsz):
            sid = str(qids[i]) if i < len(qids) else str(len(items))
            row = {
                "sample_id": sid,
                "CCS": float(ccs[i].item()),
                "HCSS": float(hcss[i].item()),
                "text_de": float(tde[i].item()),
                "visual_ie": float(vie[i].item()),
                "text_ie": float(tie[i].item()),
                "valid": bool(valid_mask[i].item() > 0.5),
                "error": str(error_reason[i]) if i < len(error_reason) else "",
            }
            if i < len(fb_rows) and fb_rows[i] is not None:
                row["fusion_bank"] = fb_rows[i]
            items.append(row)
    return items


@torch.no_grad()
def partial_update_signal_cache(
    model_frozen,
    data_loader,
    tokenizer,
    intervention_bank: Optional[Dict],
    device,
    output_path: str,
    update_ratio: float = 0.25,
    logger=None,
    ema_momentum: float = 0.9,
    ccs_text_de_scale: float = 1.0,
    store_fusion_bank: bool = True,
):
    """
    Partial refresh cache:
      - sample ~update_ratio data
      - recompute signals on sampled ids
      - overwrite only sampled qids in cache
    """
    builder = StableDOSignalBuilder(
        ema_momentum=float(ema_momentum),
        ccs_text_de_scale=float(ccs_text_de_scale),
        store_fusion_bank=bool(store_fusion_bank),
    )
    model_frozen.eval()
    cache_map = _load_cache_items_as_map(output_path)
    updates = _compute_signal_items_for_loader(
        model_frozen=model_frozen,
        data_loader=data_loader,
        tokenizer=tokenizer,
        intervention_bank=intervention_bank,
        device=device,
        builder=builder,
        sample_ratio=float(update_ratio),
        desc=f"partial_update_cache(r={float(update_ratio):.2f})",
    )
    for it in updates:
        sid = str(it.get("sample_id", "")).strip()
        if sid:
            cache_map[sid] = it
    _write_cache_map(cache_map, output_path)
    msg = (
        f"Signal cache partially updated: {len(updates)} samples refreshed "
        f"(ratio={float(update_ratio):.2f}) -> {output_path}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
