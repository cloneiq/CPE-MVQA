import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import math
import os
import json
from tqdm import tqdm
import time
import logging
import itertools
import inspect
from collections import deque
from typing import Dict
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from utils.lossfc_tools import get_current_consistency_weight
def _get_mask_ratio(epoch: int, total_epochs: int = 35, schedule: str = "step") -> float:
    """mask_ratio 动态增加，与 utils.causal_schedule 一致"""
    if schedule == "cosine" and total_epochs > 1:
        progress = min(1.0, max(0.0, (epoch - 1) / (total_epochs - 1)))
        return 0.05 + 0.20 * (1.0 - math.cos(math.pi * progress)) / 2.0
    if epoch <= 5:
        return 0.15
    if epoch <= 10:
        return 0.20
    return 0.25
# Causal modules are imported lazily inside train_epoch() to avoid
# ImportError when running in baseline mode (--use_causal 0).

logger = logging.getLogger(__name__)


def _pearson_corr_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """标量 Pearson r；样本数<2 或非有限返回 nan。"""
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    if x.numel() != y.numel():
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    if d < 1e-12 or not np.isfinite(d):
        return float("nan")
    r = (x * y).sum().item() / d
    return float(r) if np.isfinite(r) else float("nan")


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_score_with_logits(logits, labels):
    """硬标签评估：正确=1，错误=0"""
    pred_idx = torch.max(logits, 1)[1].data.long().to(labels.device)
    gt_idx = torch.max(labels, 1)[1].data.long().to(labels.device)
    correct = (pred_idx == gt_idx).float()
    return correct


def filter_kwargs_for_causal_masks(func, kw):
    """只传入 ``func`` 签名中存在的关键字参数，避免因果管线 API 增减导致 test/val 报错。"""
    try:
        sig = inspect.signature(func)
        names = set(sig.parameters)
        return {k: v for k, v in kw.items() if k in names}
    except (ValueError, TypeError):
        return dict(kw)


def router_cm_topk(scores: torch.Tensor, batch_size: int, rtr: float, device: torch.device) -> torch.Tensor:
    """因果子集：仅 router score top-k（无阈值 hybrid，减少与 calibration 的耦合）。"""
    k = max(1, min(batch_size, int(batch_size * max(0.0, min(1.0, rtr)) + 1e-6)))
    cm = torch.zeros(batch_size, dtype=torch.bool, device=device)
    _, idx = torch.topk(scores, k, largest=True)
    cm[idx] = True
    return cm


def moe_probe_router_batch(model, images, questions_ids, attention_mask, fwd_kwargs, config, use_amp):
    """Bypass 前向 + router；返回 sub_indices（排序）、router_intra [B]、scores [B]。"""
    batch_size = images.size(0)
    rtr = float(config.get("router_topk_ratio", config.get("causal_ratio", 0.15)))
    rtr = max(0.0, min(1.0, rtr))
    if not (bool(int(config.get("use_moe_router", 0))) and getattr(model, "router_trunk", None) is not None):
        return list(range(batch_size)), None, None, rtr
    with torch.amp.autocast('cuda',enabled=use_amp):
        _ = model(
            images, questions_ids, attention_mask,
            causal_bypass=True, return_vision_pooled=False, **fwd_kwargs
        )
    sc, intra = model.forward_causal_router(
        model._last_mm_text_cls,
        model._last_multi_modal_cls_feats,
    )
    if sc is None:
        return list(range(batch_size)), None, None, rtr
    cm = router_cm_topk(sc, batch_size, rtr, images.device)
    sub_indices = sorted(cm.nonzero(as_tuple=True)[0].detach().cpu().tolist())
    if len(sub_indices) == 0:
        sub_indices = [int(torch.argmax(sc).item())]
    return sub_indices, intra, sc, rtr


def binary_ce_with_hard_negative(logits, targets, neg_smooth=0.3, neg_weight=0.5):
    batch_size = logits.size(0) // 2
    num_classes = logits.size(1)
    orig_logits = logits[:batch_size]
    neg_logits = logits[batch_size:]
    orig_targets = targets[:batch_size]
    neg_targets = targets[batch_size:].clone()

    orig_loss = F.binary_cross_entropy_with_logits(orig_logits, orig_targets)
    batch_score = compute_score_with_logits(orig_logits, orig_targets)

    # Create smoothed label matrix
    smoothed_targets = torch.zeros_like(neg_targets)
    positive_indices = torch.argmax(neg_targets, dim=1)
    
    # Get original value of positive class for each sample
    original_positive_values = torch.gather(neg_targets, 1, positive_indices.unsqueeze(1)).squeeze(1)
    
    # Compute value assigned to other classes
    other_value = original_positive_values.unsqueeze(1) * neg_smooth / (num_classes - 1)
    
    for i in range(batch_size):
        # Set all classes to small value
        smoothed_targets[i, :] = other_value[i]
        
        # Set positive class to original value times retention ratio
        original_value = original_positive_values[i]
        smoothed_targets[i, positive_indices[i]] = original_value * (1 - neg_smooth)

    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, smoothed_targets)
    total_loss = (1 - neg_weight) * orig_loss + neg_weight * neg_loss

    return total_loss, batch_score


def prepare_batch_data(batch, device, duplicate_text=True, duplicate_mask=True):
    processed_data = {}

    # Process image data - concatenate three types of images
    images = batch['images'].to(device)
    pos_images = batch['pos_images'].to(device)
    neg_images = batch['neg_images'].to(device)
    processed_data['combined_images'] = torch.cat([images, pos_images, neg_images], dim=0)
    processed_data['images'] = images
    processed_data['pos_images'] = pos_images
    processed_data['neg_images'] = neg_images

    # Process text data - duplicate
    questions_ids = batch['questions']['input_ids'].to(device)
    attention_mask = batch['questions']['attention_mask'].to(device)
    do_questions_ids = batch['do_questions']['input_ids'].to(device)
    do_attention_mask = batch['do_questions']['attention_mask'].to(device)

    if duplicate_text:
        processed_data['questions_ids'] = torch.cat([questions_ids, questions_ids], dim=0)
        processed_data['attention_mask'] = torch.cat([attention_mask, attention_mask], dim=0)
        processed_data['do_questions_ids'] = torch.cat([do_questions_ids, do_questions_ids], dim=0)
        processed_data['do_attention_mask'] = torch.cat([do_attention_mask, do_attention_mask], dim=0)
    else:
        processed_data['questions_ids'] = questions_ids
        processed_data['attention_mask'] = attention_mask
        processed_data['do_questions_ids'] = do_questions_ids
        processed_data['do_attention_mask'] = do_attention_mask

    targets = batch['targets'].to(device)
    if duplicate_text:
        processed_data['targets'] = torch.cat([targets, targets], dim=0)
    else:
        processed_data['targets'] = targets

    optional_fields = ['ae_images', 'maml_images', 'pattern_embedding', 'entity_embedding']
    for field in optional_fields:
        if field in batch and batch[field] is not None:
            tensor = batch.get(field).to(device)
            if duplicate_text:
                processed_data[field] = torch.cat([tensor, tensor], dim=0)
            else:
                processed_data[field] = tensor

    if 'mask' in batch and batch['mask'] is not None:
        structure_mask = batch.get('mask').to(device)
        structure_mask = structure_mask.squeeze(1)
        if duplicate_mask:
            processed_data['structure_mask'] = torch.cat([structure_mask, structure_mask], dim=0)
        else:
            processed_data['structure_mask'] = structure_mask

    processed_data['batch_size'] = processed_data['questions_ids'].size(0)

    return processed_data


def build_cls_aux_loss(
    logits, open_logits, pred_emb, concept_logits,
    batch_size, smooth_targets, batch, model, config,
    category_weights, open_embedding_loss_weight, open_loss_weight,
    open_embedding_topk_soft, open_embedding_soft_temp,
    open_embedding_align_lam, open_embedding_hybrid_weight,
    lambda_concept, logger, batch_index, category_weights_logged,
):
    """分类 BCE/CE + Open embedding 混合 + concept CE + align；供 baseline / causal skip / 单次门控前向复用。"""
    answer_types = batch.get('answer_types', [])
    answer_indices = batch.get('answer_indices', None)
    use_open_emb = (open_logits is not None and answer_indices is not None and
                    getattr(model, 'use_open_embedding_matching', False))
    if batch_index == 0 and not category_weights_logged:
        emb_w = open_embedding_loss_weight if getattr(model, 'use_open_embedding_matching', False) else open_loss_weight
        logger.info(f"  [Category Weights] Modality=1.0 Plane=1.0 Organ={category_weights.get('organ', 1.0)} Abnormality={category_weights.get('abnormality', 2.0)} Open={emb_w} Closed=1.0")
        category_weights_logged = True
        if logits.shape[1] != smooth_targets.shape[1]:
            raise ValueError(
                f"Logits/targets shape mismatch: logits={logits.shape} targets={smooth_targets.shape}. "
                "Check return_vision_pooled unpacking - logits must match num_classes."
            )
        logger.info(f"First batch: logits={logits.shape} targets={smooth_targets.shape} (num_classes={smooth_targets.shape[1]})")
        t_sum = smooth_targets.sum(dim=1)
        t_zero = (t_sum < 1e-6).sum().item()
        if t_zero > 0:
            logger.warning(f"  [Sanity] {t_zero}/{batch_size} samples have all-zero targets! Check answer_vocab/answer2idx.")
        pred_first = logits.argmax(dim=1)
        gt_first = smooth_targets.argmax(dim=1)
        match_first = (pred_first == gt_first).sum().item()
        logger.info(f"  [Sanity] First batch: pred==gt {match_first}/{batch_size}, pred_unique={len(pred_first.unique())}")
    if use_open_emb:
        ans_idx = answer_indices.to(logits.device).long().clamp(0, logits.size(1) - 1)
        per_sample = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)
        answer_emb = getattr(model, 'answer_embeddings', None)
        topk = min(open_embedding_topk_soft, open_logits.size(1))
        for j in range(batch_size):
            cat = (answer_types[j] or "").strip().lower() if j < len(answer_types) else ""
            if cat == 'open':
                if answer_emb is not None and topk > 0:
                    emb = F.normalize(answer_emb.float(), dim=-1).detach().to(open_logits.dtype).to(open_logits.device)
                    sim = emb @ emb[ans_idx[j]] / open_embedding_soft_temp
                    topk_vals, topk_idx = sim.topk(topk, dim=-1)
                    weights = F.softmax(topk_vals, dim=0)
                    soft_target = torch.zeros_like(open_logits[j])
                    soft_target[topk_idx] = weights
                    log_p = F.log_softmax(open_logits[j:j+1], dim=-1)
                    ce_open = -(soft_target.unsqueeze(0) * log_p).sum(dim=1).mean()
                else:
                    ce_open = F.cross_entropy(open_logits[j:j+1], ans_idx[j:j+1], reduction='mean')
                bce_open = F.binary_cross_entropy_with_logits(
                    logits[j:j+1], smooth_targets[j:j+1], reduction='none').mean(dim=1).squeeze(0)
                per_sample[j] = (1.0 - open_embedding_hybrid_weight) * bce_open + open_embedding_hybrid_weight * ce_open
            else:
                per_sample[j] = F.binary_cross_entropy_with_logits(
                    logits[j:j+1], smooth_targets[j:j+1], reduction='none').mean(dim=1).squeeze(0)
    else:
        per_sample = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction='none').mean(dim=1)
    w = torch.ones(batch_size, device=logits.device, dtype=per_sample.dtype)
    for j in range(min(len(answer_types), batch_size)):
        cat = (answer_types[j] or "").strip().lower()
        if use_open_emb and cat == 'open':
            w[j] = open_embedding_loss_weight
        else:
            w[j] = category_weights.get(cat, 1.0)
    loss = (per_sample * w).sum() / (w.sum() + 1e-6)
    if use_open_emb and pred_emb is not None and open_embedding_align_lam > 0:
        emb = F.normalize(answer_emb.float(), dim=-1).detach().to(pred_emb.dtype).to(pred_emb.device)
        open_mask_align = torch.tensor(
            [((answer_types[j] if j < len(answer_types) else "") or "").strip().lower() == 'open' for j in range(batch_size)],
            device=pred_emb.device, dtype=torch.bool)
        if open_mask_align.any():
            gt_emb = emb[ans_idx[open_mask_align]]
            pred_open = pred_emb[open_mask_align]
            cos_sim = (pred_open * gt_emb).sum(dim=-1)
            loss_align = (1.0 - cos_sim).mean()
            loss = loss + open_embedding_align_lam * loss_align
    if lambda_concept > 0 and concept_logits is not None:
        from utils.vqa_rad_concept import MISC_IDX
        concept_indices = batch.get("concept_indices", [])
        if len(concept_indices) >= batch_size:
            open_mask = torch.tensor(
                [((answer_types[j] if j < len(answer_types) else "") or "").strip().lower() == 'open' for j in range(batch_size)],
                device=concept_logits.device, dtype=torch.bool)
            cidx = torch.tensor(concept_indices[:batch_size], dtype=torch.long, device=concept_logits.device)
            non_misc = (cidx != MISC_IDX)
            concept_train_mask = open_mask & non_misc
            if concept_train_mask.any():
                loss_concept = F.cross_entropy(
                    concept_logits[concept_train_mask], cidx[concept_train_mask], reduction='mean')
                loss = loss + lambda_concept * loss_concept
    return loss, category_weights_logged


def _cf_shuffle_kw_tensors(kw: dict, perm: torch.Tensor, batch_size: int) -> dict:
    """Copy fwd_kwargs with first-dim batch tensors permuted (for shuffled question counterfactual)."""
    out = {}
    for k, v in kw.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
            out[k] = v[perm]
        else:
            out[k] = v
    return out


def compute_counterfactual_supervision_loss(
    model,
    logits_factual,
    images,
    questions_ids,
    attention_mask,
    targets,
    batch,
    fwd_kwargs,
    config,
    epoch,
    use_amp,
):
    """
    Margin on ground-truth logit: factual vs batch-shuffled wrong image (Q fixed) and wrong question (V fixed).
    Forwards use causal_bypass=True so CF pairs are not mixed with HCSS/CCS from factual interventions.
    """
    lam = float(config.get("lambda_counterfactual", 0.0))
    if lam <= 1e-12 or epoch < int(config.get("counterfactual_start_epoch", 10)):
        return logits_factual.new_zeros(())
    batch_size = images.size(0)
    if batch_size < 2:
        return logits_factual.new_zeros(())
    margin = float(config.get("counterfactual_margin", 0.1))
    only_closed = bool(config.get("counterfactual_only_closed", True))
    device = images.device
    gt_idx = targets.argmax(dim=1).long().clamp(0, logits_factual.size(1) - 1)
    sample_m = torch.ones(batch_size, dtype=torch.bool, device=device)
    if only_closed:
        answer_types = batch.get("answer_types", [])
        for j in range(batch_size):
            if j < len(answer_types) and (answer_types[j] or "").strip().lower() == "open":
                sample_m[j] = False
    if not sample_m.any():
        return logits_factual.new_zeros(())
    idx = torch.arange(batch_size, device=device)
    perm_v = (idx + 1) % batch_size
    # 与 perm_v 不同，避免小 batch 上「错图」与「错问」总是同一配对
    perm_q = (idx + 2) % batch_size if batch_size > 2 else (idx + 1) % batch_size

    def _first(o):
        return o[0] if isinstance(o, tuple) else o

    img_v_cf = images[perm_v]
    kw_cf = dict(fwd_kwargs)
    with torch.amp.autocast('cuda',enabled=use_amp):
        out_v = model(
            img_v_cf,
            questions_ids,
            attention_mask,
            causal_bypass=True,
            return_vision_pooled=False,
            **kw_cf,
        )
        logits_v = _first(out_v)
        kw_q = _cf_shuffle_kw_tensors(fwd_kwargs, perm_q, batch_size)
        out_q = model(
            images,
            questions_ids[perm_q],
            attention_mask[perm_q],
            causal_bypass=True,
            return_vision_pooled=False,
            **kw_q,
        )
        logits_q = _first(out_q)
    z = logits_factual.gather(1, gt_idx.unsqueeze(1)).squeeze(1)
    z_v = logits_v.gather(1, gt_idx.unsqueeze(1)).squeeze(1)
    z_q = logits_q.gather(1, gt_idx.unsqueeze(1)).squeeze(1)
    p = torch.sigmoid(z)
    p_v = torch.sigmoid(z_v)
    p_q = torch.sigmoid(z_q)
    per = F.relu(margin - (p - p_v)) + F.relu(margin - (p - p_q))
    w = sample_m.to(dtype=per.dtype, device=per.device).float()
    loss_cf = (per * w).mean()
    return lam * loss_cf


def _unpack_train_forward_out(out, need_vision_pooled, model):
    """训练前向返回值拆包（vision pooled / open embedding / concept）。"""
    open_logits = pred_emb = concept_logits = None
    v_full_vision = None
    if need_vision_pooled:
        logits = out[0]
        v_full_vision = out[1]
        open_logits = out[2] if len(out) > 2 else None
    else:
        if isinstance(out, tuple) and len(out) >= 2:
            logits = out[0]
            if getattr(model, 'use_open_embedding_matching', False):
                open_logits, pred_emb = out[1], out[2] if len(out) > 2 else None
                concept_logits = out[3] if len(out) > 3 else None
            elif getattr(model, 'use_open_concept_head', False):
                concept_logits = out[1]
            else:
                open_logits = out[1]
                pred_emb = out[2] if len(out) > 2 else None
        else:
            logits = out
    return logits, v_full_vision, open_logits, pred_emb, concept_logits


def _load_offline_causal_cache(cache_path):
    if not cache_path or (not os.path.exists(cache_path)):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and "items" in raw and isinstance(raw["items"], list):
        items = raw["items"]
    elif isinstance(raw, dict):
        items = [{"sample_id": k, **(v if isinstance(v, dict) else {})} for k, v in raw.items()]
    else:
        items = []
    for it in items:
        sid = str(it.get("sample_id", ""))
        if sid:
            out[sid] = it
    return out


def _get_cached_signals_for_batch(batch, cache, device, fusion_bank_dim: int = 0):
    qids = batch.get("qid", [])
    bsz = len(qids)
    # Neutral defaults: avoid pseudo-zero being treated as valid causal signal.
    ccs = torch.ones(bsz, device=device, dtype=torch.float32)
    hcss = torch.ones(bsz, device=device, dtype=torch.float32)
    text_de = torch.zeros(bsz, device=device, dtype=torch.float32)
    vis_ie = torch.zeros(bsz, device=device, dtype=torch.float32)
    valid_mask = torch.zeros(bsz, device=device, dtype=torch.float32)
    fusion_bank = None
    fusion_bank_valid = None
    if fusion_bank_dim and fusion_bank_dim > 0:
        fusion_bank = torch.zeros(bsz, fusion_bank_dim, device=device, dtype=torch.float32)
        fusion_bank_valid = torch.zeros(bsz, device=device, dtype=torch.float32)
    for i, q in enumerate(qids):
        qk = str(q).strip()
        item = cache.get(qk, None)
        if item is None and qk.isdigit():
            item = cache.get(str(int(qk)), None)
        if item is None:
            item = {}
        is_valid = bool(item.get("valid", True))
        if not is_valid:
            continue
        ccs[i] = float(item.get("CCS", 0.0))
        hcss[i] = float(item.get("HCSS", 0.0))
        text_de[i] = float(item.get("text_de", 0.0))
        vis_ie[i] = float(item.get("visual_ie", 0.0))
        valid_mask[i] = 1.0
        fb = item.get("fusion_bank")
        if fusion_bank is not None and isinstance(fb, list) and len(fb) == fusion_bank_dim:
            fusion_bank[i] = torch.tensor(fb, device=device, dtype=torch.float32)
            fusion_bank_valid[i] = 1.0
    out = {"ccs": ccs, "hcss": hcss, "text_de": text_de, "vis_ie": vis_ie, "valid_mask": valid_mask}
    if fusion_bank is not None:
        out["fusion_bank"] = fusion_bank
        out["fusion_bank_valid"] = fusion_bank_valid
    return out


def train_epoch(model, data_loader, criterion, optimizer, scheduler, structure_mask_generator, device, epoch,
                grad_clip=None, log_interval=10,
                config=None, topv=-1, ture_topk_ratio=0.35, tokenizer=None, intervention_bank=None,
                scaler=None, teacher_model=None):
    model.train()

    total_cls_loss = 0.0
    total_score = 0.0
    total_current = 0
    total_factor_loss_sum = 0.0
    total_factor_loss_cnt = 0
    start_time = time.time()
    label_smoothing = config.get("label_smoothing", 0.1)
    use_amp = scaler is not None
    
    # Set which epoch to start calculating mask and performing causal reasoning
    causal_start_epoch = config.get("causal_start_epoch", 5)
    use_causal = config.get("use_causal", True)
    use_offline_causal = bool(config.get("use_offline_causal", False))
    use_do_controller = bool(config.get("use_do_controller", True)) and use_causal
    enable_causal = use_causal and (epoch >= causal_start_epoch)
    if use_offline_causal:
        enable_causal = False
    if use_do_controller:
        enable_causal = False
    # Realtime causal training path is fully deprecated.
    if enable_causal:
        logger.warning("Realtime causal training path is removed; forcing pure DO-controller training.")
        enable_causal = False
    if use_do_controller and (not use_offline_causal):
        logger.warning("DO controller enabled but --use_offline_causal=0; no intervention signals will be injected.")
    
    # Teacher computes HCSS/CCS to break feedback loop; Student (model) is trained
    causal_model = teacher_model if teacher_model is not None else model
    teacher_ema_decay = float(config.get("teacher_ema_decay", 0.999)) if config else 0.999
    if enable_causal:
        _moe = bool(int(config.get("use_moe_router", 0)))
        _rtr = float(config.get("router_topk_ratio", config.get("causal_ratio", 0.15)))
        logger.info(
            f"Epoch {epoch}: Causal Reasoning ENABLED (Start Epoch: {causal_start_epoch}, "
            f"router_topk_ratio={_rtr:.3f}{' + MoE' if _moe else ' (random top-k)'})"
            + (f" | Teacher for HCSS/CCS (EMA={teacher_ema_decay})" if teacher_model is not None else " | Single-model")
        )
    else:
        if use_do_controller:
            logger.info(
                f"Epoch {epoch}: DO controller mode (offline/no_grad signals -> representation masks)"
            )
        elif not use_causal:
            logger.info(f"Epoch {epoch}: Baseline Mode (Causal Reasoning DISABLED)")
        else:
            logger.info(f"Epoch {epoch}: Base Training (Causal starts at Epoch {causal_start_epoch})")

    # Causal effectiveness statistics (epoch-level)
    causal_total_batches = 0
    causal_guided_on_batches = 0
    # 类别权重: Modality 1.0, Plane 1.0, Organ 1.2, Abnormality 2.0 (不超过2.5)
    # SLAKE/VQA-RAD: OPEN/CLOSED 统一 1.0，未知类型 fallback 1.0
    abn_loss_weight = min(float(config.get("abn_loss_weight", 2.0)), 2.5)
    organ_loss_weight = float(config.get("organ_loss_weight", 1.2))
    open_loss_weight = float(config.get("open_loss_weight", 2.0))  # OPEN 问题更难，提高权重
    open_embedding_loss_weight = float(config.get("open_embedding_loss_weight", 1.5))  # CE梯度强，embedding matching时用较低权重
    open_embedding_topk_soft = int(config.get("open_embedding_topk_soft", 5))  # Top-k soft target: lung≈pulmonary≈chest
    open_embedding_soft_temp = float(config.get("open_embedding_soft_temp", 0.07))  # soft target 温度 0.05~0.1，按语义距离加权
    open_embedding_align_lam = float(config.get("open_embedding_align_lam", 0.2))  # alignment loss: 1-cos(pred_emb, gt_emb)
    open_embedding_hybrid_weight = float(config.get("open_embedding_hybrid_weight", 0.5))  # Open: (1-w)*BCE + w*CE(open_logits)
    category_weights = {"modality": 1.0, "plane": 1.0, "organ": organ_loss_weight, "abnormality": abn_loss_weight,
                       "open": open_loss_weight, "closed": 1.0}
    category_weights_logged = False
    # Diagnostic: accumulate causal mask stats (q_mean, v_kept_ratio, HCSS/CCS/IE/DE)
    causal_q_sum, causal_q_cnt = 0.0, 0
    causal_v_kept_sum, causal_v_cnt = 0.0, 0
    causal_eff_inv_sum, causal_eff_inv_cnt = 0.0, 0
    causal_hcss_sum, causal_ccs_sum = 0.0, 0.0
    ccs_dist_list, v_keep_dist_list, q_keep_dist_list, delta_dist_list = [], [], [], []
    ccs_epoch_list = []  # 全 epoch CCS，用于连续退火 ccs_var
    hcss_epoch_list, vis_ie_epoch_list, text_de_epoch_list, text_ie_epoch_list = [], [], [], []  # 动态分位数（CEM 锚点用 text_IE）
    ccs_gt_01_cnt, ccs_gt_02_cnt, ccs_total_cnt = 0, 0, 0
    ccs_pos_cnt, ccs_neg_cnt = 0, 0  # CCS sign distribution
    causal_vis_ie_sum, causal_vis_ie_g_sum, causal_vis_ie_l_sum = 0.0, 0.0, 0.0
    causal_text_ie_sum, causal_text_de_sum = 0.0, 0.0
    causal_sign_adj_sum = 0.0
    # Offline diagnostics: track whether cached priors are actually injected.
    offline_total_cnt = 0
    offline_hit_cnt = 0
    offline_hcss_sum = 0.0
    offline_ccs_sum = 0.0
    offline_bias_mask_sum = 0.0
    offline_ccs_neg_sum = 0.0
    do_ratio_sum = 0.0
    do_mask_t_sum = 0.0
    do_mask_v_sum = 0.0
    do_delta_logits_sum = 0.0
    do_metric_cnt = 0
    diag_empty_cnt, diag_exception_cnt, diag_ccs_patches_none_cnt = 0, 0, 0
    diag_last_exception = ""
    diag_empty_reasons: Dict[str, int] = {}
    cem_vc, cem_cm, cem_tc, cem_lb, cem_vb, cem_neutral = 0, 0, 0, 0, 0, 0  # vb=Visual Bias

    if enable_causal:
        logger.info(
            "Causal mode: MoE 仅 top-k 子集 + 两阶段 warmup（先 CE+router，再 CEM ramp + CCS intra）；"
            "HCSS 不调制；阈値 hybrid 已移除"
        )
        logger.info(f"  [Log] log_interval={log_interval} (每{log_interval} batch 打印 HCSS/CCS/分布/CCS=0诊断)")
        inv_lam = config.get("invariance_lambda", 0.0)
        inv_pure = config.get("invariance_use_pure_vision", True)
        if inv_lam > 0:
            logger.info(f"Invariance: lambda={inv_lam}, use_pure_vision={inv_pure} ({'pure visual (no text)' if inv_pure else 'fusion (text-conditioned)'})")
        from models.causal_modules import HCSSComputer, CCSComputer
        from pipeline.causal_masks_intervention import compute_causal_masks_from_interventions
        hcss_computer = HCSSComputer()
        ccs_text_de_scale = config.get("ccs_text_de_scale", 1.0)
        ccs_computer = CCSComputer(text_de_scale=ccs_text_de_scale)
        pure_encoder = None
        if config.get("causal_semantic_filter", 0) > 0 and tokenizer is not None:
            try:
                from pipeline.realtime_intervention_generator import load_pure_encoder_for_interventions
                roberta_path = config.get("roberta_path", "pretrain/roberta-base")
                pure_encoder, _ = load_pure_encoder_for_interventions(roberta_path, device="cpu")
            except Exception:
                pure_encoder = None

    offline_cache = {}
    if use_offline_causal:
        cache_path = str(config.get("causal_cache_path", ""))
        offline_cache = _load_offline_causal_cache(cache_path)
        logger.info(f"Epoch {epoch}: Offline causal mode ENABLED | cache={cache_path} | items={len(offline_cache)}")

    for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} Training")):
        # =========== 1) Prepare input ===========
        images = batch['images'].to(device)
        data = prepare_batch_data(batch, device, duplicate_text=False)
        questions_ids = data['questions_ids']
        attention_mask = data['attention_mask']
        targets = data['targets']
        do_questions_ids = data['do_questions_ids']
        do_attention_mask = data['do_attention_mask']
        pattern_embedding = data.get('pattern_embedding')
        entity_embedding = data.get('entity_embedding')
        ae_images = data.get('ae_images')
        maml_images = data.get('maml_images')
        batch_size = questions_ids.size(0)

        # Current M3AE-style forward expects aligned batch sizes for image/text.
        # Do not concatenate pos/neg images here; causal guidance is handled in the
        # second guided pass with masks.
        forward_images = images

        # =========== 2) Forward (with AMP) ===========
        inv_use_pure = config.get("invariance_use_pure_vision", True)
        need_vision_pooled = enable_causal and config.get("invariance_lambda", 0.0) > 0
        return_vision_val = "pure" if (need_vision_pooled and inv_use_pure) else (True if need_vision_pooled else False)
        fwd_kwargs = dict(
            do_questions_ids=do_questions_ids, do_attention_mask=do_attention_mask,
            ae_images=ae_images, maml_images=maml_images,
            pattern_embedding=pattern_embedding, entity_embedding=entity_embedding,
            epoch=epoch, causal_start_epoch=causal_start_epoch, training=True
        )
        if use_offline_causal:
            qids_now = batch.get("qid", [])
            hits_now = 0
            for q in qids_now:
                qk = str(q).strip()
                if (qk in offline_cache) or (qk.isdigit() and (str(int(qk)) in offline_cache)):
                    hits_now += 1
            offline_total_cnt += len(qids_now)
            offline_hit_cnt += hits_now
        if label_smoothing > 0:
            smooth_targets = targets * (1 - label_smoothing) + label_smoothing / targets.size(-1)
        else:
            smooth_targets = targets
        lambda_concept = float(config.get("lambda_concept", 0.0))

        if not enable_causal:
            with torch.amp.autocast('cuda',enabled=use_amp):
                causal_signals = None
                if use_do_controller and use_offline_causal:
                    causal_signals = _get_cached_signals_for_batch(
                        batch,
                        offline_cache,
                        forward_images.device,
                        fusion_bank_dim=int(config.get("fusion_bank_dim", 0) or 0),
                    )
                    offline_hcss_sum += float(causal_signals["hcss"].mean().item()) * batch_size
                    offline_ccs_sum += float(causal_signals["ccs"].mean().item()) * batch_size
                    offline_bias_mask_sum += float(causal_signals["text_de"].abs().mean().item()) * batch_size
                    offline_ccs_neg_sum += float((causal_signals["ccs"] < 0).float().mean().item()) * batch_size
                out = model(
                    forward_images,
                    questions_ids,
                    attention_mask,
                    return_vision_pooled=return_vision_val,
                    causal_signals=causal_signals,
                    apply_do=bool(use_do_controller),
                    **fwd_kwargs,
                )
                logits, v_full_vision, open_logits, pred_emb, concept_logits = _unpack_train_forward_out(
                    out, need_vision_pooled, model)
                # DO+offline：主损走 build_cls_aux_loss（加权 BCE/CE + Open embedding 等），非 argmax CE
                loss, category_weights_logged = build_cls_aux_loss(
                    logits, open_logits, pred_emb, concept_logits,
                    batch_size, smooth_targets, batch, model, config,
                    category_weights, open_embedding_loss_weight, open_loss_weight,
                    open_embedding_topk_soft, open_embedding_soft_temp,
                    open_embedding_align_lam, open_embedding_hybrid_weight,
                    lambda_concept, logger, i, category_weights_logged,
                )
                lam_anchor = float(config.get("lambda_do_anchor", 0.0))
                if lam_anchor > 1e-12 and use_do_controller:
                    feats_do = getattr(model, "_last_multi_modal_cls_feats", None)
                    feats_base = getattr(model, "_last_fusion_s", None)
                    ccs_now = getattr(model, "_last_ccs_scalar", None)
                    if (
                        isinstance(feats_do, torch.Tensor)
                        and isinstance(feats_base, torch.Tensor)
                        and feats_do.shape == feats_base.shape
                        and isinstance(ccs_now, torch.Tensor)
                    ):
                        mse_anchor_per = (feats_do - feats_base.detach()).pow(2).mean(dim=-1)
                        w_closed = (ccs_now.detach() >= 0).float()
                        if w_closed.sum() > 1e-6:
                            loss_anchor = (mse_anchor_per * w_closed).sum() / (w_closed.sum() + 1e-6)
                            loss = loss + lam_anchor * loss_anchor
                lam_bank = float(config.get("lambda_offline_bank_align", 0.0))
                if lam_bank > 1e-12 and use_do_controller and isinstance(causal_signals, dict):
                    fb = causal_signals.get("fusion_bank")
                    fb_ok = causal_signals.get("fusion_bank_valid")
                    fs = getattr(model, "_last_fusion_s", None)
                    if (
                        isinstance(fb, torch.Tensor)
                        and isinstance(fb_ok, torch.Tensor)
                        and isinstance(fs, torch.Tensor)
                        and fb.shape == fs.shape
                    ):
                        per_b = (fs - fb.detach()).pow(2).mean(dim=-1)
                        if fb_ok.sum() > 1e-6:
                            loss = loss + lam_bank * (per_b * fb_ok).sum() / (fb_ok.sum() + 1e-6)
                if (not use_do_controller) and float(config.get("lambda_counterfactual", 0.0)) > 1e-12:
                    loss = loss + compute_counterfactual_supervision_loss(
                        model, logits, forward_images, questions_ids, attention_mask,
                        targets, batch, fwd_kwargs, config, epoch, use_amp,
                    )
            answer_types = batch.get('answer_types', [])
            answer_indices = batch.get('answer_indices', None)
            use_open_emb = (
                open_logits is not None
                and answer_indices is not None
                and getattr(model, "use_open_embedding_matching", False)
            )
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            if teacher_model is not None:
                with torch.no_grad():
                    for t_param, s_param in zip(teacher_model.parameters(), model.parameters()):
                        t_param.data.mul_(teacher_ema_decay).add_(s_param.data, alpha=1.0 - teacher_ema_decay)

            total_current += batch_size
            if use_do_controller:
                dr = getattr(model, "_last_do_ratio", None)
                mt = getattr(model, "_last_mask_t_mean", None)
                mv = getattr(model, "_last_mask_v_mean", None)
                dl = getattr(model, "_last_delta_logits", None)
                if isinstance(dr, torch.Tensor):
                    do_ratio_sum += float(dr.item()) * batch_size
                    do_mask_t_sum += float(mt.item()) * batch_size if isinstance(mt, torch.Tensor) else 0.0
                    do_mask_v_sum += float(mv.item()) * batch_size if isinstance(mv, torch.Tensor) else 0.0
                    do_delta_logits_sum += float(dl.item()) * batch_size if isinstance(dl, torch.Tensor) else 0.0
                    do_metric_cnt += batch_size
            # Open: 用 open_logits 预测；Closed: 用 logits
            if use_open_emb:
                open_mask = torch.tensor([((answer_types[j] if j < len(answer_types) else "") or "").strip().lower() == 'open' for j in range(batch_size)], device=logits.device, dtype=torch.bool)
                if open_mask.any():
                    combined = logits.clone()
                    combined[open_mask] = open_logits[open_mask]
                    total_score += compute_score_with_logits(combined, targets).sum().item()
                else:
                    total_score += compute_score_with_logits(logits, targets).sum().item()
            else:
                total_score += compute_score_with_logits(logits, targets).sum().item()
            total_cls_loss += loss.item() * batch_size

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                if use_offline_causal and offline_total_cnt > 0:
                    off_cov = offline_hit_cnt / max(1, offline_total_cnt)
                    off_h = offline_hcss_sum / max(1, total_current)
                    off_c = offline_ccs_sum / max(1, total_current)
                    off_b = offline_bias_mask_sum / max(1, total_current)
                    off_neg = offline_ccs_neg_sum / max(1, total_current)
                    ca_log = float(
                        config.get("causal_alpha", getattr(model, "causal_alpha", 0.5))
                    )
                    do_ratio = do_ratio_sum / max(1, do_metric_cnt)
                    do_mt = do_mask_t_sum / max(1, do_metric_cnt)
                    do_mv = do_mask_v_sum / max(1, do_metric_cnt)
                    do_dl = do_delta_logits_sum / max(1, do_metric_cnt)
                    logger.info(
                        f"| Batch {i + 1}/{len(data_loader)} | {elapsed * 1000 / log_interval:.2f} ms/batch | "
                        f"Loss {total_cls_loss / total_current:.4f} | "
                        f"Score {total_score / total_current * 100:.2f}% | "
                        f"OfflinePriors cov={off_cov:.2%} hcss={off_h:.4f} ccs={off_c:.4f} bias_mask={off_b:.4f} "
                        f"causal_alpha(cfg)={ca_log:.3f} ccs_neg_ratio={off_neg:.2%} | "
                        f"DO do_ratio={do_ratio:.2%} m_t={do_mt:.4f} m_v={do_mv:.4f} delta_logits={do_dl:.6f} | "
                        f"LR {lr:.2e}"
                    )
                else:
                    logger.info(
                        f"| Batch {i + 1}/{len(data_loader)} | {elapsed * 1000 / log_interval:.2f} ms/batch | "
                        f"Loss {total_cls_loss / total_current:.4f} | "
                        f"Score {total_score / total_current * 100:.2f}% | "
                        f"LR {lr:.2e}"
                    )
                start_time = time.time()
            continue
            
        # =========== Causal：MoE router top-k（推荐）或 legacy 固定比例随机 top-k；干预仅算子集一次 ===========
        causal_total_batches += 1
        inv_dissim_batch = None
        ablation_no_hcss = config.get("ablation_no_hcss", False)
        ablation_no_ccs = config.get("ablation_no_ccs", False)
        use_light_causal_weights = config.get("use_light_causal_weights", False)
        use_schedule = config.get("use_causal_schedule", False)
        if use_schedule and not use_light_causal_weights:
            from utils.causal_schedule import get_hcss_lam, get_ccs_lam
            lam_hcss_override = config.get("lambda_hcss", -1)
            lam_ccs_override = config.get("lambda_ccs", -1)
            hcss_weight = 0.0 if ablation_no_hcss else (lam_hcss_override if lam_hcss_override >= 0 else get_hcss_lam(epoch))
            ccs_weight = 0.0 if ablation_no_ccs else (lam_ccs_override if lam_ccs_override >= 0 else get_ccs_lam(epoch))
            mask_sched = config.get("mask_schedule", "step")
            total_ep = config.get("epochs", 35)
            ccs_mask_ratio = _get_mask_ratio(epoch, total_ep, mask_sched)
            is_stage1 = (ccs_weight == 0)  # epoch 1-4
            is_stage2 = (ccs_weight > 0)
        elif use_light_causal_weights:
            hcss_weight = 0.003
            ccs_weight = 0.003
            ccs_mask_ratio = config.get("ccs_mask_ratio", 0.4)
            is_stage1 = False
            is_stage2 = True
        else:
            causal_lam = config.get("causal_lam", 0.02)
            ccs_lam = config.get("ccs_lam", 0.01)
            hcss_stage1_epochs = config.get("hcss_stage1_epochs", 3)
            causal_start_epoch = config.get("causal_start_epoch", 1)
            is_stage1 = (epoch >= causal_start_epoch) and (epoch < causal_start_epoch + hcss_stage1_epochs)
            is_stage2 = (epoch >= causal_start_epoch + hcss_stage1_epochs)
            hcss_weight = causal_lam if (is_stage1 or is_stage2) else 0.0
            ccs_weight = ccs_lam if is_stage2 else 0.0
            ccs_mask_ratio = config.get("ccs_mask_ratio", 0.4)
        ccs_alpha = config.get("ccs_alpha", 4.0)
        inv_lambda = config.get("invariance_lambda", 0.0)
        causal_start_epoch = config.get("causal_start_epoch", 1)
        t = max(0, epoch - causal_start_epoch)
        inv_weight = 0.0  # Invariance disabled
        if causal_total_batches == 1:
            phase = "Stage1 (HCSS only)" if is_stage1 else ("Stage2 (HCSS+CCS)" if is_stage2 else "Base")
            logger.info(f"  [Causal Phase] {phase} | λ_hcss={hcss_weight:.4f} λ_ccs={ccs_weight:.4f} mask_ratio={ccs_mask_ratio:.2f} inv=OFF")
            if use_schedule:
                logger.info(f"  [Schedule] Epoch {epoch}: HCSS/CCS/mask 按epoch调度")
            elif is_stage1:
                stage1_end = causal_start_epoch + config.get("hcss_stage1_epochs", 3) - 1
                logger.info(f"  [Stage1] Epoch {causal_start_epoch}-{stage1_end}: Loss=CE+λ_hcss*HCSS, v_mask=1")
            elif is_stage2:
                stage1_end = causal_start_epoch + config.get("hcss_stage1_epochs", 3) - 1
                logger.info(f"  [Stage2] Epoch {stage1_end + 1}+: Loss=CE+λ_hcss*HCSS+λ_ccs*CCS, v_mask=sigmoid({ccs_alpha}*CCS)")
            _um = bool(int(config.get("use_moe_router", 0)))
            if _um:
                logger.info(
                    f"  [MoE] pure top-k | Stage1 epoch<{int(config.get('moe_warmup_epochs', 5))}: CE+router (CEM=0, no CCS intra) | "
                    f"Stage2: CEM ramp {float(config.get('moe_post_warm_cem_scale', 0.06))}→1 over {int(config.get('moe_cem_ramp_epochs', 5))} ep"
                )
            _cf_e = int(config.get("counterfactual_start_epoch", 10))
            _cg_e = int(config.get("cem_gate_align_start_epoch", 10))
            _ie_e = int(config.get("ie_reg_start_epoch", 10))
            logger.info(
                f"  [延迟约束] CF≥ep{_cf_e} gate_align≥ep{_cg_e} ie_reg≥ep{_ie_e} | "
                f"HCSS IE scale={float(config.get('hcss_ie_scale', 1.5)):.3f}"
            )
        hcss_topk_ratio = config.get("hcss_topk_ratio", 0.4)
        v_causal_topk_ratio = config.get("v_causal_topk_ratio", 0.4)
        causal_max_interventions = config.get("causal_max_interventions", 5)
        causal_mask_causal_parts = config.get("causal_mask_causal_parts", True)
        min_quality_interventions = config.get("min_quality_interventions", 1)
        min_entity_overlap = config.get("min_entity_overlap", 0.2)
        sim_low = config.get("sim_low", 0.45)
        sim_high = config.get("sim_high", 0.90)
        sim_low_strong = config.get("sim_low_strong", 0.25)
        overlap_min_strong = config.get("overlap_min_strong", 0.02)
        relax_sim_low = config.get("relax_sim_low", 0.50)
        relax_min_entity_overlap = config.get("relax_min_entity_overlap", 0.15)
        allow_last_resort_interventions = config.get("allow_last_resort_interventions", False)
        ccs_negative_weight = config.get("ccs_negative_weight", 0.0)
        ccs_negative_weight_min = config.get("ccs_negative_weight_min", 0.3)
        sign_adj_margin = config.get("sign_adj_margin", 0.05)
        if not use_schedule:
            ccs_mask_ratio = config.get("ccs_mask_ratio", 0.5)
        ccs_topk_local = config.get("ccs_topk_local", 5)
        ccs_tau = config.get("ccs_tau", 0.01)
        local_ie_alpha = config.get("local_ie_alpha", 1.0)
        ccs_use_local_ie = config.get("ccs_use_local_ie", True)
        ccs_target = config.get("ccs_target", 0.2)
        ccs_penalty_lambda = config.get("ccs_penalty_lambda", 0.08)
        inv_margin = config.get("invariance_margin", 0.1)
        inv_ccs_threshold = config.get("invariance_ccs_threshold", 0.05)
        abn_hcss_target = config.get("abn_hcss_target", 0.17)
        abn_hcss_margin = config.get("abn_hcss_margin", 0.06)
        abn_ratio_target = config.get("abn_ratio_target", 1.3)
        abn_lam = config.get("abn_lam", 0.05)
        energy_lambda = config.get("energy_lambda", 0.005)
        question_texts = batch.get("question_texts", [])
        image_paths = batch.get("image_paths", [])
        answer_types_batch = batch.get("answer_types", [])
        concepts_batch = batch.get("concepts", [])

        return_vision_causal = "pure" if (inv_lambda > 0 and inv_use_pure) else (True if inv_lambda > 0 else False)

        use_moe = bool(int(config.get("use_moe_router", 0))) and getattr(model, "router_trunk", None) is not None
        router_scores = None
        router_intra_gate = None
        if use_moe:
            sub_indices, router_intra_gate, router_scores, rtr = moe_probe_router_batch(
                model, forward_images, questions_ids, attention_mask, fwd_kwargs, config, use_amp
            )
            cm = torch.zeros(batch_size, dtype=torch.bool, device=forward_images.device)
            cm[torch.tensor(sub_indices, device=forward_images.device, dtype=torch.long)] = True
        else:
            rtr = float(config.get("router_topk_ratio", config.get("causal_ratio", 0.15)))
            rtr = max(0.0, min(1.0, rtr))
            k_sub = max(1, min(batch_size, int(batch_size * rtr + 1e-6)))
            sub_indices = random.sample(list(range(batch_size)), k_sub)
            cm = torch.zeros(batch_size, dtype=torch.bool, device=forward_images.device)
            cm[torch.tensor(sub_indices, device=forward_images.device, dtype=torch.long)] = True
        sub_size = len(sub_indices)

        fwd_kwargs_intervention = {}
        for k, v in fwd_kwargs.items():
            if k == "return_vision_pooled":
                continue
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == batch_size:
                fwd_kwargs_intervention[k] = v[sub_indices]
            else:
                fwd_kwargs_intervention[k] = v
        q_causal_mask, v_causal_mask, causal_stats, sample_weights = compute_causal_masks_from_interventions(
            model,
            forward_images[sub_indices],
            questions_ids[sub_indices],
            attention_mask[sub_indices],
            tokenizer,
            intervention_bank,
            causal_model,
            hcss_computer,
            ccs_computer,
            pure_encoder=pure_encoder,
            device=forward_images.device,
            seq_len=questions_ids.size(1),
            num_visual_patches=576,
            hcss_topk_ratio=hcss_topk_ratio,
            v_causal_topk_ratio=v_causal_topk_ratio,
            causal_mask_causal_parts=causal_mask_causal_parts,
            min_quality_interventions=min_quality_interventions,
            min_entity_overlap=min_entity_overlap,
            sim_low=sim_low,
            sim_high=sim_high,
            sim_low_strong=sim_low_strong,
            overlap_min_strong=overlap_min_strong,
            relax_sim_low=relax_sim_low,
            relax_min_entity_overlap=relax_min_entity_overlap,
            allow_last_resort_interventions=allow_last_resort_interventions,
            max_interventions=causal_max_interventions,
            question_texts=[question_texts[j] for j in sub_indices] if question_texts else None,
            image_paths=[image_paths[j] for j in sub_indices] if image_paths else None,
            targets=targets[sub_indices],
            answer_types=[answer_types_batch[j] for j in sub_indices] if answer_types_batch else None,
            fwd_kwargs=fwd_kwargs_intervention,
            ccs_negative_weight=ccs_negative_weight,
            ccs_negative_weight_min=ccs_negative_weight_min,
            sign_adj_margin=sign_adj_margin,
            ccs_mask_ratio=ccs_mask_ratio,
            ccs_topk_local=ccs_topk_local,
            ccs_tau=ccs_tau,
            local_ie_alpha=local_ie_alpha,
            ccs_use_local_ie=ccs_use_local_ie,
            ccs_target=ccs_target,
            ccs_penalty_lambda=ccs_penalty_lambda,
            use_feature_gate=config.get("use_feature_gate", False),
            gate_alpha=config.get("gate_alpha", 1.0),
            gate_beta=config.get("gate_beta", 0.8),
            hcss_ie_scale=float(config.get("hcss_ie_scale", 1.5)),
            sign_adj_temp=float(config.get("sign_adj_temp", 0.05)),
            hcss_norm_tau=float(config.get("hcss_norm_tau", 0.01)),
            hcss_floor=float(config.get("hcss_floor", 0.02)),
        )

        v_gate = causal_stats.get("v_gate")
        text_hcss_mask = causal_stats.get("text_hcss_mask")
        use_feature_gate = config.get("use_feature_gate", False)
        # 关键: schedule mask_ratio=0 时禁用 guided mask，保护早期 representation
        # effective_mask_ratio 来自 q_causal_mask，必须在此处统一绑定 schedule
        if use_schedule and ccs_mask_ratio <= 0:
            q_causal_mask = torch.ones_like(q_causal_mask, device=q_causal_mask.device, dtype=q_causal_mask.dtype)
            v_causal_mask = torch.ones_like(v_causal_mask, device=v_causal_mask.device, dtype=v_causal_mask.dtype)
            if causal_total_batches == 1:
                logger.info(f"  [Schedule] mask_ratio=0 (Epoch {epoch}<=5) → 禁用 guided mask，effective_mask_ratio≈0")
        hcss_per = causal_stats.get("hcss_per_sample", [])
        ccs_per = causal_stats.get("ccs_per_sample", [])
        visual_ie_per = causal_stats.get("visual_ie_per_sample", [])
        text_ie_per = causal_stats.get("text_ie_per_sample", [])
        text_de_per = causal_stats.get("text_de_per_sample", [])

        # 子集干预 → 扩展为全 batch；非 top-k 行保持中性 mask/统计
        if sub_size < batch_size:
            q_full = torch.ones(batch_size, questions_ids.size(1), device=forward_images.device, dtype=torch.float32)
            v_full = torch.ones(batch_size, 577, device=forward_images.device, dtype=torch.float32)
            sw_full = torch.ones(batch_size, device=forward_images.device, dtype=torch.float32)
            for idx, j in enumerate(sub_indices):
                q_full[j] = q_causal_mask[idx]
                v_full[j] = v_causal_mask[idx]
                sw_full[j] = sample_weights[idx]
            q_causal_mask = q_full
            v_causal_mask = v_full
            sample_weights = sw_full
            if v_gate is not None:
                v_gate_full = torch.ones(batch_size, 577, device=forward_images.device, dtype=torch.float32)
                for idx, j in enumerate(sub_indices):
                    v_gate_full[j] = v_gate[idx]
                v_gate = v_gate_full
            # 扩展 hcss/ccs/visual_ie/text_de 到全 batch，非 causal 样本用 0.5（中性）
            if len(hcss_per) == len(sub_indices) and len(ccs_per) == len(sub_indices):
                hcss_full = [0.5] * batch_size
                ccs_full = [0.5] * batch_size
                for idx, j in enumerate(sub_indices):
                    hcss_full[j] = float(hcss_per[idx])
                    ccs_full[j] = float(ccs_per[idx])
                hcss_per, ccs_per = hcss_full, ccs_full
            if len(visual_ie_per) == len(sub_indices) and len(text_de_per) == len(sub_indices):
                vi_full = [0.5] * batch_size
                td_full = [0.5] * batch_size
                ti_full = [0.5] * batch_size
                tie_sub = text_ie_per if len(text_ie_per) == len(sub_indices) else None
                for idx, j in enumerate(sub_indices):
                    vi_full[j] = float(visual_ie_per[idx])
                    td_full[j] = float(text_de_per[idx])
                    if tie_sub is not None:
                        ti_full[j] = float(tie_sub[idx])
                    else:
                        ti_full[j] = 0.5
                visual_ie_per, text_de_per, text_ie_per = vi_full, td_full, ti_full
            # 扩展 text_hcss_mask 到全 batch，非 causal 样本用全 1（不抑制）
            if text_hcss_mask is not None and text_hcss_mask.size(0) == sub_size:
                seq_len = text_hcss_mask.size(1)
                mask_full = torch.ones(batch_size, seq_len, device=forward_images.device, dtype=text_hcss_mask.dtype)
                for idx, j in enumerate(sub_indices):
                    mask_full[j] = text_hcss_mask[idx]
                text_hcss_mask = mask_full
            # 扩展 ccs_patches 到全 batch，非 causal 样本用全 0（patch gate 中性）
            ccs_patches_raw = causal_stats.get("ccs_patches")
            if ccs_patches_raw is not None and ccs_patches_raw.size(0) == sub_size:
                patch_dim = ccs_patches_raw.size(1)
                ccs_full = torch.zeros(batch_size, patch_dim, device=forward_images.device, dtype=ccs_patches_raw.dtype)
                for idx, j in enumerate(sub_indices):
                    ccs_full[j] = ccs_patches_raw[idx]
                causal_stats["ccs_patches"] = ccs_full
        if len(text_ie_per) != batch_size and len(text_de_per) == batch_size:
            text_ie_per = [0.5] * batch_size
        # 强消融: No HCSS/No CCS 时禁用 CEM 和 modal_competition，只用 v_gate
        use_modal_competition = use_feature_gate and len(hcss_per) == batch_size and len(ccs_per) == batch_size and not ablation_no_hcss and not ablation_no_ccs
        use_cem = use_modal_competition and len(visual_ie_per) == batch_size and len(text_ie_per) == batch_size
        # 动态分位数：从 epoch 累计数据计算（含当前 batch 前的数据，当前 batch 稍后 append）
        def _percentile(arr, p):
            if not arr or len(arr) == 0:
                return 0.0
            a = np.array(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            return float(np.percentile(a, p)) if len(a) > 0 else 0.0
        # 先 append 当前 batch 的 raw 数据（用 sub 的原始长度）
        _hp_raw = causal_stats.get("hcss_per_sample", [])
        _cp_raw = causal_stats.get("ccs_per_sample", [])
        _vp_raw = causal_stats.get("visual_ie_per_sample", [])
        _bp_raw = causal_stats.get("text_de_per_sample", [])
        _ti_raw = causal_stats.get("text_ie_per_sample", [])
        if len(_hp_raw) > 0:
            ccs_epoch_list.extend([float(x) for x in _cp_raw])
            hcss_epoch_list.extend([float(x) for x in _hp_raw])
            vis_ie_epoch_list.extend([float(x) for x in _vp_raw])
            text_de_epoch_list.extend([float(x) for x in _bp_raw])
            if len(_ti_raw) > 0:
                text_ie_epoch_list.extend([float(x) for x in _ti_raw])
            else:
                text_ie_epoch_list.extend([float(x) for x in _bp_raw])
        p25_ccs = _percentile(ccs_epoch_list, 25)
        p50_ccs = _percentile(ccs_epoch_list, 50)
        p75_ccs = _percentile(ccs_epoch_list, 75)
        p25_hcss = _percentile(hcss_epoch_list, 25)
        p50_hcss = _percentile(hcss_epoch_list, 50)
        p75_hcss = _percentile(hcss_epoch_list, 75)
        p25_vis = _percentile(vis_ie_epoch_list, 25)
        p50_vis = _percentile(vis_ie_epoch_list, 50)
        p75_vis = _percentile(vis_ie_epoch_list, 75)
        p25_text = _percentile(text_ie_epoch_list if text_ie_epoch_list else text_de_epoch_list, 25)
        p75_text = _percentile(text_ie_epoch_list if text_ie_epoch_list else text_de_epoch_list, 75)
        # 默认值防空（首 batch 无历史时）
        if p50_vis <= 0:
            p50_vis = 0.50
        if p50_hcss <= 0:
            p50_hcss = 0.14
        if p75_text <= 0:
            p75_text = 0.25
        if p75_vis <= 0:
            p75_vis = 0.60
        # Causal dropout (epoch>=20): 20% prob 关闭 visual/text causal 防单模态依赖
        causal_dropout = config.get("causal_dropout", 0.0)
        causal_dropout_start = config.get("causal_dropout_start_epoch", 20)
        if causal_dropout > 0 and epoch >= causal_dropout_start:
            for j in range(batch_size):
                if not cm[j]:
                    continue
                if random.random() < causal_dropout:
                    v_causal_mask[j].fill_(1.0)  # 关闭 visual causal
                if random.random() < causal_dropout:
                    q_causal_mask[j].fill_(1.0)  # 关闭 text causal
        # Stage1: v_mask=1 (不改变视觉). Stage2: v_mask=sigmoid(α*CCS) per-sample broadcast
        if is_stage1:
            v_mask_for_causal = torch.ones(batch_size, 577, device=forward_images.device, dtype=torch.float32)
        elif is_stage2 and len(ccs_per) == batch_size:
            ccs_t = torch.tensor([float(c) for c in ccs_per], device=forward_images.device, dtype=torch.float32)
            v_sigmoid = torch.sigmoid(ccs_alpha * ccs_t).unsqueeze(1).expand(-1, 577)
            v_mask_for_causal = v_sigmoid
        elif use_feature_gate and v_gate is not None and not use_modal_competition:
            v_mask_for_causal = torch.ones(batch_size, 577, device=forward_images.device, dtype=torch.float32)
        else:
            v_mask_for_causal = v_causal_mask
        # 干净样本：不参与 q/v 反事实注意力偏置（与纯融合一致）；并修正 Stage2 上中性 CCS 导致的 v_mask≠1
        v_mask_for_causal = torch.where(cm.unsqueeze(-1), v_mask_for_causal, torch.ones_like(v_mask_for_causal))
        q_causal_mask = torch.where(cm.unsqueeze(-1), q_causal_mask, torch.ones_like(q_causal_mask))
        # Neutral 样本弱引导: gate=0.5 不强化不抛弃，保留学习
        neutral_mask = [False] * batch_size
        if use_cem and len(hcss_per) == batch_size and len(ccs_per) == batch_size:
            for j in range(batch_size):
                c, h, v, b = float(ccs_per[j]), float(hcss_per[j]), float(visual_ie_per[j]), float(text_ie_per[j])
                is_vc = v >= p75_vis and c > 0 and h < p50_hcss
                is_vb = v >= p75_vis and c <= p25_ccs
                is_lb = b >= p75_text and v < p50_vis
                is_cm = v >= p50_vis and h >= p50_hcss and c > 0
                is_tc = h >= p50_hcss and v < p75_vis
                if not (is_vc or is_vb or is_lb or is_cm or is_tc):
                    neutral_mask[j] = True
        # CEM: runtime soft_gate（可解释）；主任务仅 logits 层 CEM 混合
        answer_type_list = batch.get("answer_types", [])
        ccs_patches_tensor = causal_stats.get("ccs_patches")
        if use_cem:
            fwd_gate = {"hcss_per_sample": hcss_per, "ccs_per_sample": ccs_per,
                        "visual_ie_per_sample": visual_ie_per, "text_ie_per_sample": text_ie_per,
                        "p50_vis": p50_vis, "p50_hcss": p50_hcss, "p50_ccs": p50_ccs, "p25_hcss": p25_hcss, "p75_text": p75_text,
                        "neutral_mask": neutral_mask}
            if len(answer_type_list) == batch_size:
                fwd_gate["answer_type_per_sample"] = answer_type_list
        elif use_modal_competition:
            fwd_gate = {"hcss_per_sample": hcss_per, "ccs_per_sample": ccs_per,
                        "p50_ccs": p50_ccs, "p25_hcss": p25_hcss}
        else:
            fwd_gate = {"v_gate": v_gate} if (use_feature_gate and v_gate is not None) else {}
        if text_hcss_mask is not None and text_hcss_mask.size(0) == batch_size:
            fwd_gate["text_hcss_mask"] = text_hcss_mask
        if ccs_patches_tensor is not None and ccs_patches_tensor.size(0) == batch_size:
            fwd_gate["ccs_patches"] = ccs_patches_tensor
        ones_v = torch.ones(batch_size, 577, device=forward_images.device, dtype=torch.float32)
        _mw = int(config.get("moe_warmup_epochs", 5))
        _cr = int(config.get("moe_cem_ramp_epochs", 5))
        _cs = float(config.get("moe_post_warm_cem_scale", 0.06))
        if not use_moe:
            cem_path_scale = 1.0
            _moe_pass_intra = True
        elif epoch < _mw:
            cem_path_scale = 0.0
            _moe_pass_intra = False
        elif _cr <= 0:
            cem_path_scale = 1.0
            _moe_pass_intra = True
        elif epoch < _mw + _cr:
            if _cr == 1:
                t_lin = 1.0
            else:
                t_lin = (epoch - _mw) / float(_cr - 1)
            t_lin = max(0.0, min(1.0, t_lin))
            cem_path_scale = _cs + (1.0 - _cs) * t_lin
            _moe_pass_intra = True
        else:
            cem_path_scale = 1.0
            _moe_pass_intra = True
        fwd_merged = {
            **fwd_kwargs, **fwd_gate,
            "causal_path_mask": cm.float(),
            "cem_path_scale": float(cem_path_scale),
            "cem_gt_indices": targets.argmax(dim=1).long(),
        }
        if router_intra_gate is not None and _moe_pass_intra:
            fwd_merged["router_intra_gate"] = router_intra_gate
        qi = questions_ids
        fi = forward_images
        q_for_model = q_causal_mask
        v_for_model = v_mask_for_causal
        if bool(config.get("use_feature_causal_probe", False)) and cm.any():
            pr_t = float(config.get("probe_text_token_ratio", 0.0))
            pr_i = float(config.get("probe_patch_zero_ratio", 0.0))
            if pr_t > 0 and tokenizer is not None:
                qi = questions_ids.clone()
                pad_id = int(getattr(tokenizer, "pad_token_id", None) or 1)
                for j in range(batch_size):
                    if not cm[j]:
                        continue
                    for pos in range(qi.size(1)):
                        if attention_mask[j, pos].item() <= 0:
                            continue
                        if random.random() < pr_t:
                            qi[j, pos] = pad_id
            if pr_i > 0:
                fi = forward_images.clone()
                _, _, Hp, Wp = fi.shape
                for j in range(batch_size):
                    if not cm[j]:
                        continue
                    if random.random() >= pr_i:
                        continue
                    eh = max(1, int(Hp * random.uniform(0.05, 0.25)))
                    ew = max(1, int(Wp * random.uniform(0.05, 0.25)))
                    top = random.randint(0, max(0, Hp - eh))
                    left = random.randint(0, max(0, Wp - ew))
                    fi[j, :, top:top + eh, left:left + ew] = 0
        with torch.amp.autocast('cuda',enabled=use_amp):
            def _first(o):
                return o[0] if isinstance(o, tuple) else o
            out_causal = model(
                fi, qi, attention_mask,
                q_mask_pre=q_for_model, v_mask=v_for_model,
                return_vision_pooled=return_vision_causal, **fwd_merged)
            if inv_lambda > 0:
                logits = out_causal[0]
                v_patch_vision = out_causal[1]
                open_logits = out_causal[2] if len(out_causal) > 2 else None
                pred_emb = concept_logits = None
                with torch.no_grad():
                    out_pure = model(
                        forward_images, questions_ids, attention_mask,
                        causal_bypass=True, return_vision_pooled="pure", **fwd_kwargs)
                    v_full_vision = out_pure[1] if isinstance(out_pure, tuple) else None
            else:
                logits, v_full_vision, open_logits, pred_emb, concept_logits = _unpack_train_forward_out(
                    out_causal, need_vision_pooled=False, model=model)
            loss, category_weights_logged = build_cls_aux_loss(
                logits, open_logits, pred_emb, concept_logits,
                batch_size, smooth_targets, batch, model, config,
                category_weights, open_embedding_loss_weight, open_loss_weight,
                open_embedding_topk_soft, open_embedding_soft_temp,
                open_embedding_align_lam, open_embedding_hybrid_weight,
                lambda_concept, logger, i, category_weights_logged,
            )
            lam_cem_guard = float(config.get("lambda_cem_guard", 0.0))
            if (
                lam_cem_guard > 1e-12
                and bool(getattr(model, "use_logits_cem", True))
                and bool(config.get("use_causal", True))
            ):
                lb = getattr(model, "_last_logits_base", None)
                if lb is not None and isinstance(lb, torch.Tensor) and lb.shape == logits.shape:
                    # KL(P_out || P_base) = sum p_out (log p_out - log p_base)；log_target=True 时 target 为 log p_out
                    loss_cem_guard = F.kl_div(
                        lb.log_softmax(dim=-1),
                        logits.log_softmax(dim=-1),
                        reduction="batchmean",
                        log_target=True,
                    )
                    loss = loss + lam_cem_guard * loss_cem_guard
            if float(config.get("lambda_counterfactual", 0.0)) > 1e-12:
                loss = loss + compute_counterfactual_supervision_loss(
                    model, logits, fi, qi, attention_mask,
                    targets, batch, fwd_kwargs, config, epoch, use_amp,
                )
            lam_cem_a_base = float(config.get("lambda_cem_align", 0.0))
            lam_cem_a = lam_cem_a_base
            if (
                bool(config.get("cem_align_schedule", False))
                and lam_cem_a_base > 1e-12
            ):
                es = int(config.get("cem_align_start_epoch", 10))
                ef = int(config.get("cem_align_full_epoch", 20))
                if epoch < es:
                    lam_cem_a = 0.0
                elif ef <= es:
                    lam_cem_a = lam_cem_a_base
                else:
                    lam_cem_a = lam_cem_a_base * max(
                        0.0, min(1.0, (epoch - es) / float(ef - es))
                    )
            if lam_cem_a > 0 and bool(config.get("use_causal_aligned_cem", False)) and cm.any() and cem_path_scale > 1e-8:
                pb = F.softmax(getattr(model, '_last_logits_base', logits).detach(), dim=-1)
                pf = F.softmax(logits, dim=-1)
                per = F.smooth_l1_loss(pf, pb, reduction="none").mean(dim=-1)
                w_align = cm.to(dtype=per.dtype, device=per.device).float()
                loss = loss + lam_cem_a * (per * w_align).sum() / (w_align.sum() + 1e-6)
            # CEM gate 与 IE 对齐（可延迟启用：epoch >= cem_gate_align_start_epoch）
            lam_cem_gate = float(config.get("lambda_cem_gate_align", 0.0))
            _eg_start = int(config.get("cem_gate_align_start_epoch", 10))
            if (
                epoch >= _eg_start
                and lam_cem_gate > 1e-12
                and use_cem
                and len(visual_ie_per) == batch_size
                and len(text_ie_per) == batch_size
            ):
                g_t = getattr(model, "_last_cem_g_t", None)
                g_v = getattr(model, "_last_cem_g_v", None)
                if g_t is not None and g_v is not None:
                    ti = torch.tensor(
                        [float(text_ie_per[j]) for j in range(batch_size)],
                        device=logits.device, dtype=logits.dtype,
                    )
                    vi = torch.tensor(
                        [float(visual_ie_per[j]) for j in range(batch_size)],
                        device=logits.device, dtype=logits.dtype,
                    )
                    den = (ti + vi).clamp(min=1e-6)
                    ti_tar = (ti / den).clamp(0.05, 0.95)
                    vi_tar = (vi / den).clamp(0.05, 0.95)
                    s_tar = ti_tar + vi_tar
                    ti_tar = ti_tar / s_tar
                    vi_tar = vi_tar / s_tar
                    w_cm = cm.to(dtype=g_t.dtype, device=g_t.device).float()
                    mse_t = (g_t - ti_tar.detach()).pow(2)
                    mse_v = (g_v - vi_tar.detach()).pow(2)
                    if w_cm.sum() > 1e-6:
                        loss_gate = ((mse_t + mse_v) * w_cm).sum() / (w_cm.sum() + 1e-6)
                        loss = loss + lam_cem_gate * loss_gate
            lam_ie_reg = float(config.get("lambda_ie_reg", 0.0))
            lam_ie_floor = float(config.get("lambda_ie_floor", 0.002))
            lam_ie_de_cpl = float(config.get("lambda_ie_de_coupling", 0.002))
            _ier_start = int(config.get("ie_reg_start_epoch", 10))
            if epoch >= _ier_start and len(text_ie_per) == batch_size:
                ti_ie = torch.tensor(
                    [float(text_ie_per[j]) for j in range(batch_size)],
                    device=logits.device, dtype=logits.dtype,
                )
                if lam_ie_reg > 1e-12:
                    loss_ie_reg = ((ti_ie - ti_ie.mean()) ** 2).mean()
                    loss = loss + lam_ie_reg * loss_ie_reg
                if lam_ie_floor > 1e-12:
                    ie_thr = float(config.get("ie_floor_threshold", 0.08))
                    loss_ie_floor = torch.relu(ie_thr - ti_ie).mean()
                    loss = loss + lam_ie_floor * loss_ie_floor
                if lam_ie_de_cpl > 1e-12 and len(text_de_per) == batch_size:
                    text_de_tensor = torch.tensor(
                        [float(text_de_per[j]) for j in range(batch_size)],
                        device=logits.device, dtype=logits.dtype,
                    ).detach()
                    # DE 作常数因子：不通过该项反传压 DE，仅调节 IE×DE（IE 可导时 DE 高则压 IE）
                    loss_ie_de_coupling = (ti_ie * text_de_tensor).mean()
                    loss = loss + lam_ie_de_cpl * loss_ie_de_coupling
            lam_re = min(float(config.get("lambda_router_entropy", 0.002)), 0.01)
            lam_cal = float(config.get("lambda_router_calib", 0.02))
            _mw_r = int(config.get("moe_warmup_epochs", 5))
            if use_moe and router_scores is not None:
                p = torch.sigmoid(router_scores)
                if lam_cal > 1e-8:
                    rt = max(1e-4, min(1.0 - 1e-4, float(rtr)))
                    kld = (
                        p * torch.log((p.clamp(1e-6, 1.0 - 1e-6)) / rt)
                        + (1.0 - p) * torch.log(((1.0 - p).clamp(1e-6, 1.0 - 1e-6)) / (1.0 - rt))
                    )
                    loss = loss + lam_cal * kld.mean()
                if lam_re > 0 and epoch >= _mw_r:
                    ent = -(
                        p * torch.log(p.clamp(1e-8, 1.0 - 1e-8))
                        + (1.0 - p) * torch.log((1.0 - p).clamp(1e-8, 1.0 - 1e-8))
                    ).mean()
                    loss = loss - lam_re * ent
            lam_feat_cons = float(config.get("lambda_causal_feat_consistency", 0.0025))
            if lam_feat_cons > 1e-12 and cm.any():
                f_causal = model._last_multi_modal_cls_feats
                with torch.no_grad():
                    _ = model(
                        fi, qi, attention_mask,
                        causal_bypass=True,
                        return_vision_pooled=False,
                        **fwd_kwargs,
                    )
                f_clean = model._last_multi_modal_cls_feats.detach()
                nf = F.normalize(f_causal, dim=-1, eps=1e-6)
                nc = F.normalize(f_clean, dim=-1, eps=1e-6)
                per_fc = (nf - nc).pow(2).mean(dim=-1)
                w_fc = cm.to(dtype=per_fc.dtype, device=per_fc.device).float()
                loss = loss + lam_feat_cons * (per_fc * w_fc).sum() / (w_fc.sum() + 1e-6)
            # L_open_refine: Open **硬 top-k** patch 仅作 λ_open·BCE 辅助；**不**用于 train acc / 推理 logits
            # 主干首轮已走 soft gate；此处只传 v_mask=v_open_refine + 基座 fwd_kwargs，不叠 q_causal / CEM / ccs 软门控（解耦）
            # 强消融 ablation_no_open_refine: 不做该 forward
            lambda_open = float(config.get("lambda_open", 0.0))
            ablation_no_open_refine = config.get("ablation_no_open_refine", False)
            open_refine_v_topk = float(config.get("open_refine_v_topk_ratio", 0.2))
            concept_focus_w = float(config.get("concept_focus_refine_weight", 1.3))
            if lambda_open > 0 and len(answer_type_list) >= batch_size and not ablation_no_ccs and not ablation_no_open_refine:
                open_mask = torch.tensor(
                    [((answer_type_list[j] or "").strip().lower() == 'open') for j in range(batch_size)],
                    device=logits.device, dtype=torch.bool
                )
                open_refine_mask = open_mask & cm
                if open_refine_mask.any() and ccs_patches_tensor is not None and ccs_patches_tensor.size(0) == batch_size:
                    num_patches = 576
                    k_strict = max(1, int(num_patches * open_refine_v_topk))
                    v_open_refine = ones_v.clone()
                    cp = ccs_patches_tensor.float().to(forward_images.device)
                    for j in range(batch_size):
                        if open_refine_mask[j]:
                            patch_scores = cp[j, 1:num_patches+1]
                            if patch_scores.numel() >= k_strict:
                                _, top_idx = torch.topk(patch_scores, k_strict)
                                v_open_refine[j] = 0.0
                                v_open_refine[j, 0] = 1.0
                                v_open_refine[j, top_idx + 1] = 1.0
                    logits_open_refine_aux = _first(model(
                        forward_images, questions_ids, attention_mask,
                        q_mask_pre=None,
                        v_mask=v_open_refine,
                        return_vision_pooled=False,
                        skip_dual_fusion=True,
                        **fwd_kwargs))
                    from utils.vqa_rad_concept import MISC_IDX
                    w_refine = torch.ones(batch_size, device=logits.device, dtype=logits.dtype)
                    concept_indices = batch.get("concept_indices", [])
                    if len(concept_indices) >= batch_size:
                        cidx = torch.tensor(concept_indices[:batch_size], device=open_refine_mask.device, dtype=torch.long)
                        concept_focus_mask = (cidx != MISC_IDX)
                        w_refine[open_refine_mask & concept_focus_mask] = concept_focus_w
                    ps_bce = F.binary_cross_entropy_with_logits(
                        logits_open_refine_aux[open_refine_mask], smooth_targets[open_refine_mask], reduction='none').mean(dim=1)
                    w_open = w_refine[open_refine_mask]
                    loss_open_refine = (ps_bce * w_open).sum() / (w_open.sum() + 1e-6)
                    loss = loss + lambda_open * loss_open_refine
            # 因果能量约束: E=mean(CCS^2)，防止CCS过分集中在少量patch
            if energy_lambda > 0:
                ccs_per_inv = causal_stats.get("ccs_per_sample", [])
                if len(ccs_per_inv) > 0:
                    if len(ccs_per_inv) >= batch_size:
                        ccs_slice = ccs_per_inv[:batch_size]
                    else:
                        ccs_slice = ccs_per_inv[: len(sub_indices)]
                    ccs_t = torch.tensor(
                        [float(c) for c in ccs_slice],
                        dtype=logits.dtype, device=logits.device,
                    )
                    energy_loss = (ccs_t ** 2).mean()
                    loss = loss + energy_lambda * energy_loss
            # 视觉 causal boost: L_vis = -λ*vis_ie，强制模型使用视觉
            # 限制 boost 幅度，避免 loss 负值过大（vis_ie 高时可能使 loss<0）
            vis_boost = config.get("vis_boost_lambda", 0.02)
            if vis_boost > 0:
                vis_ie = causal_stats.get("avg_visual_ie", 0.0)
                boost_term = min(vis_boost * vis_ie, max(0.0, loss.item() - 0.01))
                loss = loss - boost_term
            # 抑制语言 shortcut: loss += λ*text_de
            text_de_penalty = config.get("text_de_penalty_lambda", 0.002)
            if text_de_penalty > 0:
                tdp = causal_stats.get("text_de_per_sample", [])
                if len(tdp) == batch_size:
                    loss = loss + text_de_penalty * torch.tensor(
                        [float(x) for x in tdp], device=logits.device, dtype=logits.dtype
                    ).mean()
                else:
                    loss = loss + text_de_penalty * float(causal_stats.get("avg_text_de", 0.0))
            # Bias 惩罚: language_bias + visual_bias，参考 CEM 指标（batch-wise 平均）
            lambda_bias_base = config.get("lambda_bias", 0.15)
            lambda_bias_dynamic = config.get("lambda_bias_dynamic", True)
            lambda_bias = lambda_bias_base + (0.01 * epoch if lambda_bias_dynamic else 0.0)
            lambda_bias = min(lambda_bias, 0.2)  # 不超过 0.2，避免压倒 cls_loss
            if lambda_bias > 0:
                hp = causal_stats.get("hcss_per_sample", [])
                cp = causal_stats.get("ccs_per_sample", [])
                vp = causal_stats.get("visual_ie_per_sample", [])
                bp = causal_stats.get("text_de_per_sample", [])
                if len(hp) == len(cp) == len(vp) == len(bp) and len(cp) > 0:
                    lang_bias_list, vis_bias_list = [], []
                    for j in range(len(cp)):
                        c, h, v, b = float(cp[j]), float(hp[j]), float(vp[j]), float(bp[j])
                        # Language bias: b>=p75_text and v<p50_vis (软形式)
                        lb = max(0.0, b - p75_text) * max(0.0, p50_vis - v)
                        lang_bias_list.append(lb)
                        # Visual bias: v>=p75_vis and c<=p25_ccs (软形式)
                        vb = max(0.0, v - p75_vis) * max(0.0, p25_ccs - c)
                        vis_bias_list.append(vb)
                    language_bias_batch = float(np.mean(lang_bias_list))
                    visual_bias_batch = float(np.mean(vis_bias_list))
                    bias_loss = language_bias_batch + visual_bias_batch
                    loss = loss + lambda_bias * bias_loss
            # Invariance: Causal Warmup 最慢启动，inv_weight = λ * σ((t-4)/τ)
            # t<3 时 inv_weight=0，不计算/不累加 effective_inv_ratio，避免误导性 50% 等统计
            if inv_lambda > 0 and inv_weight > 0:
                v_full_teacher = v_full_vision[sub_indices].detach()
                v_patch_sub = v_patch_vision[sub_indices]
                cos_sim = F.cosine_similarity(v_patch_sub, v_full_teacher, dim=1)
                inv_dissim = 1.0 - cos_sim  # [0, 2], 0=完全对齐
                inv_dissim_batch = inv_dissim.detach()
                ccs_per_inv = causal_stats.get("ccs_per_sample", [])
                if inv_ccs_threshold > -1e8 and len(ccs_per_inv) == len(sub_indices):
                    mask = torch.tensor([c > inv_ccs_threshold for c in ccs_per_inv], device=v_patch_sub.device, dtype=torch.bool)
                    if mask.any():
                        inv_dissim_sub = inv_dissim[mask]
                        invariance_loss = inv_dissim_sub.mean()
                        effective_inv_ratio = (inv_dissim_sub > inv_margin).float().mean()
                    else:
                        invariance_loss = torch.tensor(0.0, device=v_patch_sub.device)
                        effective_inv_ratio = torch.tensor(0.0, device=v_patch_sub.device)
                else:
                    invariance_loss = inv_dissim.mean()
                    effective_inv_ratio = (inv_dissim > inv_margin).float().mean()
                total_factor_loss = invariance_loss
                ccs_arr = np.array([float(c) for c in ccs_per_inv[:len(sub_indices)]], dtype=np.float64)
                ccs_var = float(np.var(ccs_arr)) if len(ccs_arr) > 1 else 0.0
                ccs_var_cap = min(max(ccs_var, 0.0), 0.8)
                inv_weight_final = inv_weight * (1.0 - 0.5 * ccs_var_cap)
                inv_weight_final = max(inv_weight_final, inv_weight * 0.3)
                loss = loss + inv_weight_final * total_factor_loss
                total_factor_loss_sum += invariance_loss.item() * batch_size
                total_factor_loss_cnt += batch_size
                causal_eff_inv_sum += effective_inv_ratio.item() * batch_size
                causal_eff_inv_cnt += batch_size
                # 调试：打印 invariance 实际贡献
                if (i + 1) % log_interval == 0:
                    loss_inv = (inv_weight_final * total_factor_loss).item() if isinstance(total_factor_loss, torch.Tensor) else inv_weight_final * total_factor_loss
                    logger.info(f"  [Inv] inv_w={inv_weight:.4f} inv_w_final={inv_weight_final:.4f} loss_inv={loss_inv:.6f} eff_ratio={effective_inv_ratio.item():.2%}")
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        optimizer.zero_grad()
        if teacher_model is not None:
            with torch.no_grad():
                for t_param, s_param in zip(teacher_model.parameters(), model.parameters()):
                    t_param.data.mul_(teacher_ema_decay).add_(s_param.data, alpha=1.0 - teacher_ema_decay)
        total_current += batch_size
        answer_types = batch.get('answer_types', [])
        answer_indices = batch.get('answer_indices', None)
        use_open_emb = (open_logits is not None and answer_indices is not None and
                        getattr(model, 'use_open_embedding_matching', False))
        if use_open_emb:
            open_mask_causal = torch.tensor([((answer_types[j] if j < len(answer_types) else "") or "").strip().lower() == 'open' for j in range(batch_size)], device=logits.device, dtype=torch.bool)
            combined_causal = logits.clone()
            if open_mask_causal.any():
                combined_causal[open_mask_causal] = open_logits[open_mask_causal]
            total_score += compute_score_with_logits(combined_causal, targets).sum().item()
        else:
            total_score += compute_score_with_logits(logits, targets).sum().item()
        total_cls_loss += loss.item() * batch_size
        causal_guided_on_batches += 1
        diag_empty_cnt += causal_stats.get("interventions_empty_cnt", 0)
        diag_exception_cnt += causal_stats.get("exception_cnt", 0)
        diag_ccs_patches_none_cnt += causal_stats.get("ccs_patches_none_cnt", 0)
        for r, c in causal_stats.get("empty_reason_counts", {}).items():
            diag_empty_reasons[r] = diag_empty_reasons.get(r, 0) + c
        ex_msg = causal_stats.get("last_exception_msg", "")
        if ex_msg:
            diag_last_exception = ex_msg
        # Diagnostic: mask stats + HCSS/CCS/IE/DE + 分布
        with torch.no_grad():
            causal_q_sum += q_causal_mask.float().mean().item() * batch_size
            causal_q_cnt += batch_size
            v_kept = (v_causal_mask > 0.5).float().sum(dim=1) / (v_causal_mask.size(1) + 1e-6)
            causal_v_kept_sum += v_kept.mean().item() * batch_size
            causal_v_cnt += batch_size
            ccs_per = causal_stats.get("ccs_per_sample", [])
            ccs_dist_list.extend(ccs_per)
            for c in ccs_per:
                ccs_total_cnt += 1
                if c > 0:
                    ccs_pos_cnt += 1
                elif c < 0:
                    ccs_neg_cnt += 1
                if c > 0.1:
                    ccs_gt_01_cnt += 1
                if c > 0.2:
                    ccs_gt_02_cnt += 1
            v_keep_dist_list.extend(v_kept.cpu().tolist())
            q_keep_dist_list.extend(q_causal_mask.float().mean(dim=1).cpu().tolist())
            if inv_dissim_batch is not None:
                delta_dist_list.extend(inv_dissim_batch.cpu().tolist())
            causal_hcss_sum += causal_stats["hcss"] * batch_size
            causal_ccs_sum += causal_stats["ccs"] * batch_size
            causal_vis_ie_sum += causal_stats["avg_visual_ie"] * batch_size
            _vg = causal_stats.get("avg_visual_ie_g", 0.0)
            _vl = causal_stats.get("avg_visual_ie_l", 0.0)
            try:
                _vg = 0.0 if not np.isfinite(float(_vg)) else float(_vg)
                _vl = 0.0 if not np.isfinite(float(_vl)) else float(_vl)
            except (TypeError, ValueError):
                _vg, _vl = 0.0, 0.0
            causal_vis_ie_g_sum += _vg * batch_size
            causal_vis_ie_l_sum += _vl * batch_size
            causal_text_ie_sum += causal_stats["avg_text_ie"] * batch_size
            causal_text_de_sum += causal_stats["avg_text_de"] * batch_size
            causal_sign_adj_sum += causal_stats.get("sign_adj_ratio", 0.0) * batch_size
            # CEM 硬分类（动态分位数版，含 Visual Bias）；强消融时跳过，HCSS/CCS 不参与决策
            hp = causal_stats.get("hcss_per_sample", [])
            cp = causal_stats.get("ccs_per_sample", [])
            vp = causal_stats.get("visual_ie_per_sample", [])
            bp = causal_stats.get("text_de_per_sample", [])
            if len(hp) == len(cp) == len(vp) == len(bp) and len(cp) > 0 and not ablation_no_hcss and not ablation_no_ccs:
                for j in range(len(cp)):
                    c, h, v, b = float(cp[j]), float(hp[j]), float(vp[j]), float(bp[j])
                    # Visual Causal: visual>=P75, CCS>0, HCSS<P50
                    if v >= p75_vis and c > 0 and h < p50_hcss:
                        cem_vc += 1
                    # Visual Bias: visual>=P75, CCS<=P25（视觉捷径）
                    elif v >= p75_vis and c <= p25_ccs:
                        cem_vb += 1
                    # Language Bias: text_DE>=P75, visual<P50
                    elif b >= p75_text and v < p50_vis:
                        cem_lb += 1
                    # Cross-modal: visual>=P50, HCSS>=P50, CCS>0
                    elif v >= p50_vis and h >= p50_hcss and c > 0:
                        cem_cm += 1
                    # Text Causal: HCSS>=P50, visual<P75
                    elif h >= p50_hcss and v < p75_vis:
                        cem_tc += 1
                    else:
                        cem_neutral += 1
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            q_mean = causal_q_sum / max(1, causal_q_cnt)
            v_kept_r = causal_v_kept_sum / max(1, causal_v_cnt)
            # 因果相关指标用 causal_q_cnt（仅因果样本），避免被非因果 batch 稀释
            _denom = max(1, causal_q_cnt)
            hcss_avg = causal_hcss_sum / _denom
            ccs_avg = causal_ccs_sum / _denom
            vis_ie_avg = causal_vis_ie_sum / _denom
            _g = causal_vis_ie_g_sum / _denom
            _l = causal_vis_ie_l_sum / _denom
            vis_ie_g_avg = _g if (_g == _g and not np.isinf(_g)) else 0.0  # nan/inf -> 0
            vis_ie_l_avg = _l if (_l == _l and not np.isinf(_l)) else 0.0
            text_ie_avg = causal_text_ie_sum / _denom
            text_de_avg = causal_text_de_sum / _denom
            sign_adj_r = causal_sign_adj_sum / _denom
            eff_inv_r = causal_eff_inv_sum / max(1, causal_eff_inv_cnt) if causal_eff_inv_cnt > 0 else 0.0
            ccs_pos_ratio = ccs_pos_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
            ccs_neg_ratio = ccs_neg_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
            hcss_part = "" if ablation_no_hcss else f"HCSS={hcss_avg:.4f} "
            ccs_part = "" if ablation_no_ccs else f"CCS={ccs_avg:.4f} "
            ccs_sign_part = f"CCS sign: pos={ccs_pos_ratio:.0%} neg={ccs_neg_ratio:.0%} | " if not ablation_no_ccs else ""
            corr_part = ""
            gt_last = getattr(model, "_last_cem_g_t", None)
            if (
                gt_last is not None
                and len(text_ie_per) == batch_size
                and len(text_de_per) == batch_size
            ):
                ti_b = torch.tensor(
                    [float(text_ie_per[j]) for j in range(batch_size)],
                    device=gt_last.device, dtype=gt_last.dtype,
                )
                td_b = torch.tensor(
                    [float(text_de_per[j]) for j in range(batch_size)],
                    device=gt_last.device, dtype=gt_last.dtype,
                )
                r_ie = _pearson_corr_1d(gt_last, ti_b)
                r_de = _pearson_corr_1d(gt_last, td_b)
                def _fmt_corr(r):
                    return f"{r:.3f}" if isinstance(r, (int, float)) and np.isfinite(r) else "nan"
                corr_part = f" | corr(g_t,text_ie)={_fmt_corr(r_ie)} corr(g_t,text_de)={_fmt_corr(r_de)}"
            msg = (
                f"| Batch {i + 1}/{len(data_loader)} | {elapsed * 1000 / log_interval:.2f} ms/batch | "
                f"Loss {total_cls_loss / total_current:.4f} | Score {total_score / total_current * 100:.2f}% | "
                f"{hcss_part}{ccs_part}SignAdj={sign_adj_r:.2f} | "
                f"{ccs_sign_part}"
                f"vis_ie={vis_ie_avg:.4f} (IE_g={vis_ie_g_avg:.4f} IE_l={vis_ie_l_avg:.4f}) text_ie={text_ie_avg:.4f} text_de={text_de_avg:.4f} | "
                f"Phase: Causal{corr_part}"
            )
            logger.info(msg)
            tqdm.write(msg)
            sys.stdout.flush()
            # 分布/CEM/CCS诊断：每 log_diagnostic_interval 次 log 打印一次，减少 I/O 与 np 计算
            log_diag = config.get("log_diagnostic_interval", 1)
            if log_diag <= 0:
                log_diag = 1
            if (i + 1) % (log_interval * log_diag) == 0 and not ablation_no_hcss and not ablation_no_ccs:
                def _dist_str(arr):
                    if not arr:
                        return "N/A"
                    a = np.array(arr, dtype=np.float64)
                    a = a[np.isfinite(a)]
                    if len(a) == 0:
                        return "N/A"
                    return f"min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f} std={a.std():.3f} p25={np.percentile(a,25):.3f} p50={np.percentile(a,50):.3f} p75={np.percentile(a,75):.3f}"
                ccs_pos_r = ccs_pos_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
                ccs_neg_r = ccs_neg_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
                dist_msg = f"  [分布] CCS: {_dist_str(ccs_dist_list)} | CCS sign distribution pos={ccs_pos_r:.0%} neg={ccs_neg_r:.0%} | v_mask: {_dist_str(v_keep_dist_list)} | q_mask: {_dist_str(q_keep_dist_list)}"
                logger.info(dist_msg)
                tqdm.write(dist_msg)
                sys.stdout.flush()
                reasons_str = " ".join(f"{k}={v}" for k, v in sorted(diag_empty_reasons.items())) if diag_empty_reasons else ""
                diag_msg = f"  [CCS=0诊断] empty={diag_empty_cnt} exception={diag_exception_cnt} ccs_patches_none={diag_ccs_patches_none_cnt}" + (f" | reasons: {reasons_str}" if reasons_str else "") + (f" | last_exc: {diag_last_exception}" if diag_last_exception else "")
                logger.info(diag_msg)
                tqdm.write(diag_msg)
                sys.stdout.flush()
                cem_total = cem_vc + cem_cm + cem_tc + cem_lb + cem_vb + cem_neutral
                if cem_total > 0 and not ablation_no_hcss and not ablation_no_ccs:
                    cem_msg = (f"  [CEM分布] Visual_causal={cem_vc/cem_total*100:.1f}% Cross_modal={cem_cm/cem_total*100:.1f}% "
                              f"Text_causal={cem_tc/cem_total*100:.1f}% Language_bias={cem_lb/cem_total*100:.1f}% "
                              f"Visual_bias={cem_vb/cem_total*100:.1f}% Neutral={cem_neutral/cem_total*100:.1f}%")
                    logger.info(cem_msg)
                    tqdm.write(cem_msg)
                    sys.stdout.flush()
            ccs_dist_list.clear()
            v_keep_dist_list.clear()
            q_keep_dist_list.clear()
            delta_dist_list.clear()
            start_time = time.time()
            
    scheduler.step()
    avg_total_loss = total_cls_loss / max(1, total_current)
    avg_score = total_score / max(1, total_current) * 100
    
    logger.info(f"Epoch {epoch} Finished | Loss {avg_total_loss:.4f} | Score {avg_score:.2f}%")
    if enable_causal and causal_q_cnt > 0:
        guided_on_ratio = causal_guided_on_batches / max(1, causal_total_batches)
        hcss_ep = causal_hcss_sum / causal_q_cnt
        ccs_ep = causal_ccs_sum / causal_q_cnt
        vis_ie_ep = causal_vis_ie_sum / causal_q_cnt
        _g = causal_vis_ie_g_sum / causal_q_cnt
        _l = causal_vis_ie_l_sum / causal_q_cnt
        vis_ie_g_ep = _g if (_g == _g and not np.isinf(_g)) else 0.0
        vis_ie_l_ep = _l if (_l == _l and not np.isinf(_l)) else 0.0
        text_ie_ep = causal_text_ie_sum / causal_q_cnt
        text_de_ep = causal_text_de_sum / causal_q_cnt
        sign_adj_ep = causal_sign_adj_sum / causal_q_cnt
        eff_inv_ep = causal_eff_inv_sum / max(1, causal_eff_inv_cnt) if causal_eff_inv_cnt > 0 else 0.0
        ccs_gt_01_ratio = ccs_gt_01_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
        ccs_gt_02_ratio = ccs_gt_02_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
        diag_ep = f"empty={diag_empty_cnt} exc={diag_exception_cnt} ccs_none={diag_ccs_patches_none_cnt}"
        if diag_empty_reasons:
            diag_ep += f" | reasons: {' '.join(f'{k}={v}' for k,v in sorted(diag_empty_reasons.items()))}"
        if diag_last_exception:
            diag_ep += f" | last_exc: {diag_last_exception[:60]}"
        ccs_pos_ep = ccs_pos_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
        ccs_neg_ep = ccs_neg_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
        _no_h = config.get("ablation_no_hcss", False)
        _no_c = config.get("ablation_no_ccs", False)
        ccs_stats = "" if _no_c else (f"avg_ccs={ccs_ep:.4f} CCS sign pos={ccs_pos_ep:.0%} neg={ccs_neg_ep:.0%} | "
            f"ccs>0.1={ccs_gt_01_ratio:.1%} ccs>0.2={ccs_gt_02_ratio:.1%} | ")
        hcss_stats = "" if _no_h else f"HCSS={hcss_ep:.4f} "
        logger.info(
            f"Epoch {epoch} Causal Stats | Batches: {causal_guided_on_batches}/{causal_total_batches} ({guided_on_ratio:.2%}) | "
            f"{ccs_stats}{hcss_stats}SignAdj={sign_adj_ep:.2f} | "
            f"vis_ie={vis_ie_ep:.4f} (IE_g={vis_ie_g_ep:.4f} IE_l={vis_ie_l_ep:.4f}) text_ie={text_ie_ep:.4f} text_de={text_de_ep:.4f}"
        )
        if not _no_c:
            logger.info(f"Epoch {epoch} CCS=0诊断: {diag_ep}")
        cem_total_ep = cem_vc + cem_cm + cem_tc + cem_lb + cem_vb + cem_neutral
        if cem_total_ep > 0 and not _no_h and not _no_c:
            logger.info(
                f"Epoch {epoch} CEM分布 | Visual_causal={cem_vc/cem_total_ep*100:.1f}% Cross_modal={cem_cm/cem_total_ep*100:.1f}% "
                f"Text_causal={cem_tc/cem_total_ep*100:.1f}% Language_bias={cem_lb/cem_total_ep*100:.1f}% "
                f"Visual_bias={cem_vb/cem_total_ep*100:.1f}% Neutral={cem_neutral/cem_total_ep*100:.1f}%"
            )
        eff_mask = (1.0 - causal_q_sum / causal_q_cnt) if causal_q_cnt > 0 else 0.0
        _ms = config.get("mask_schedule", "step")
        _te = config.get("epochs", 35)
        sched_mask = _get_mask_ratio(epoch, _te, _ms) if (config.get("use_causal_schedule", False) and not config.get("use_light_causal_weights", False)) else config.get("ccs_mask_ratio", 0)
        sum_ccs = "" if _no_c else f"avg_ccs={ccs_ep:.4f} ccs>0.1={ccs_gt_01_ratio:.1%} ccs>0.2={ccs_gt_02_ratio:.1%} | "
        tqdm.write(
            f"Epoch {epoch} Summary | Score={avg_score:.2f}% | {sum_ccs}"
            f"schedule_mask={sched_mask:.2f} effective_mask={eff_mask:.3f}"
        )
    else:
        guided_on_ratio = 0.0
    
    avg_total_factor_loss = total_factor_loss_sum / total_factor_loss_cnt if total_factor_loss_cnt > 0 else 0.0
    ccs_gt_01_r = ccs_gt_01_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
    ccs_gt_02_r = ccs_gt_02_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0
    # CEM 分布百分比（用于 JSON 记录）
    cem_total_ep = cem_vc + cem_cm + cem_tc + cem_lb + cem_vb + cem_neutral
    cem_pct = (lambda x: (x / cem_total_ep * 100) if cem_total_ep > 0 else 0.0)
    loss_info = {
        'epoch': epoch,
        'total_cls_loss': avg_total_loss,
        'total_factor_loss': avg_total_factor_loss,
        'total_loss': avg_total_loss,
        'accuracy': avg_score,
        'enable_causal': enable_causal,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'guided_on_ratio': guided_on_ratio,
        'effective_mask_ratio': (1.0 - causal_q_sum / causal_q_cnt) if causal_q_cnt > 0 else 0.0,
        'avg_ccs': causal_ccs_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'ccs_gt_01_ratio': ccs_gt_01_r,
        'ccs_gt_02_ratio': ccs_gt_02_r,
        'avg_visual_ie': causal_vis_ie_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'avg_text_de': causal_text_de_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'avg_text_ie': causal_text_ie_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'avg_hcss': causal_hcss_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'sign_adj_ratio': causal_sign_adj_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'effective_inv_ratio': causal_eff_inv_sum / causal_eff_inv_cnt if causal_eff_inv_cnt > 0 else 0.0,
        'ccs_var': float(np.var(ccs_epoch_list)) if len(ccs_epoch_list) > 1 else 0.0,
        'schedule_mask_ratio': (_get_mask_ratio(epoch, config.get("epochs", 35), config.get("mask_schedule", "step")) if (config.get("use_causal_schedule", False) and not config.get("use_light_causal_weights", False)) else config.get("ccs_mask_ratio", 0)) if enable_causal else 0.0,
        # Causal Stats（用于 JSON）
        'causal_guided_on_batches': causal_guided_on_batches,
        'causal_total_batches': causal_total_batches,
        'ccs_pos_ratio': ccs_pos_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0,
        'ccs_neg_ratio': ccs_neg_cnt / ccs_total_cnt if ccs_total_cnt > 0 else 0.0,
        'avg_visual_ie_g': causal_vis_ie_g_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        'avg_visual_ie_l': causal_vis_ie_l_sum / causal_q_cnt if causal_q_cnt > 0 else 0.0,
        # CCS=0 诊断
        'diag_empty_cnt': diag_empty_cnt,
        'diag_exception_cnt': diag_exception_cnt,
        'diag_ccs_patches_none_cnt': diag_ccs_patches_none_cnt,
        'diag_empty_reasons': dict(diag_empty_reasons),
        'diag_last_exception': (diag_last_exception[:200] if diag_last_exception else ""),
        # CEM 分布
        'cem_visual_causal_pct': round(cem_pct(cem_vc), 1),
        'cem_cross_modal_pct': round(cem_pct(cem_cm), 1),
        'cem_text_causal_pct': round(cem_pct(cem_tc), 1),
        'cem_language_bias_pct': round(cem_pct(cem_lb), 1),
        'cem_visual_bias_pct': round(cem_pct(cem_vb), 1),
        'cem_neutral_pct': round(cem_pct(cem_neutral), 1),
    }
    if use_offline_causal and offline_total_cnt > 0:
        # In offline mode, realtime causal branch is disabled by design.
        # Override summary fields with actual injected offline prior stats.
        loss_info['offline_prior_coverage'] = offline_hit_cnt / max(1, offline_total_cnt)
        loss_info['avg_offline_hcss'] = offline_hcss_sum / max(1, total_current)
        loss_info['avg_offline_ccs'] = offline_ccs_sum / max(1, total_current)
        loss_info['avg_offline_bias_mask'] = offline_bias_mask_sum / max(1, total_current)
        loss_info['offline_ccs_neg_ratio'] = offline_ccs_neg_sum / max(1, total_current)
        _ca_h = float(config.get("causal_alpha", getattr(model, "causal_alpha", 0.5)))
        loss_info["causal_alpha_hyper"] = _ca_h
        # Legacy keys: 旧版未累加 offline_alpha_*，易误读为 0；现改为与表征融合 α 一致
        loss_info["offline_alpha_mean"] = _ca_h
        loss_info["offline_alpha_max"] = _ca_h
        loss_info['avg_hcss'] = loss_info['avg_offline_hcss']
        loss_info['avg_ccs'] = loss_info['avg_offline_ccs']
        # effective_mask_ratio is a realtime q-mask metric; use offline bias-mask mean as proxy in offline mode.
        loss_info['effective_mask_ratio'] = loss_info['avg_offline_bias_mask']
    if do_metric_cnt > 0:
        loss_info['do_ratio'] = do_ratio_sum / do_metric_cnt
        loss_info['do_mask_t_mean'] = do_mask_t_sum / do_metric_cnt
        loss_info['do_mask_v_mean'] = do_mask_v_sum / do_metric_cnt
        loss_info['do_delta_logits'] = do_delta_logits_sum / do_metric_cnt
        if use_do_controller:
            # Replace misleading legacy ratio in DO mode.
            loss_info['guided_on_ratio'] = loss_info['do_ratio']
            loss_info['effective_inv_ratio'] = loss_info['do_ratio']
    # `enable_causal` 仅表示「图内实时 HCSS/CCS/干预」分支；离线 DO 下恒为 False，不代表未训因果
    loss_info["offline_do_active"] = bool(use_offline_causal and use_do_controller and use_causal)
    loss_info["realtime_causal_branch"] = bool(enable_causal)

    return avg_total_loss, avg_score, loss_info


def validate(model, data_loader, criterion, device, use_amp=False,
            use_causal_gate=False, tokenizer=None, intervention_bank=None,
            config=None):
    """Evaluate model performance on validation set with per-category breakdown.
    use_causal_gate=True: 验证也跑实时干预 + HCSS/CCS gate（与 Train 同结构，慢）。
    main 默认 use_causal_gate_in_val=0：验证无干预，仅前向（与关闭 gate 的 test 一致）。
    """
    model.eval()
    total_loss = 0.0
    total_score = 0.0
    total_samples = 0
    total_concept_correct = 0
    total_concept_count = 0

    # MedVQA2019: modality/plane/organ/abnormality; SLAKE/VQA-RAD: open/closed
    category_names = ['modality', 'plane', 'organ', 'abnormality', 'open', 'closed']
    cat_correct = {c: 0.0 for c in category_names}
    cat_total = {c: 0 for c in category_names}

    use_offline_causal = bool(config.get("use_offline_causal", False)) if config else False
    offline_cache = {}
    if use_offline_causal:
        cache_path = str(config.get("causal_cache_path_val") or config.get("causal_cache_path", ""))
        offline_cache = _load_offline_causal_cache(cache_path)
        logger.info(f"Validation: offline causal priors enabled | cache={cache_path} | items={len(offline_cache)}")

    # 验证/测试时 causal gate 所需依赖
    do_causal_gate = False
    if do_causal_gate:
        from models.causal_modules import HCSSComputer, CCSComputer
        from pipeline.causal_masks_intervention import compute_causal_masks_from_interventions
        hcss_computer = HCSSComputer()
        ccs_computer = CCSComputer(text_de_scale=config.get("ccs_text_de_scale", 1.0))
        max_inv_val = config.get("causal_max_interventions_val_test", config.get("causal_max_interventions", 3))
        logger.info(f"Validation: using causal gate (HCSS+CCS) for Train=Val structure, max_interventions={max_inv_val}")

    logger.info("Start validation...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            data = prepare_batch_data(batch, device, duplicate_text=False)
            images = data['images']
            questions = data['questions_ids']
            attention_mask = data['attention_mask']
            do_questions = data['do_questions_ids']
            do_attention_mask = data['do_attention_mask']
            targets = data['targets']
            pattern_embedding = data.get('pattern_embedding', None)
            entity_embedding = data.get('entity_embedding', None)
            ae_images = data.get('ae_images', None)
            maml_images = data.get('maml_images', None)
            if ae_images is not None:
                ae_images = ae_images.to(device)
            if maml_images is not None:
                maml_images = maml_images.to(device)
            answer_types = batch.get('answer_types', [])
            batch_size = images.size(0)

            fwd_gate = {}
            router_intra = None
            cm = None
            if do_causal_gate:
                try:
                    fwd_kwargs = dict(
                        do_questions_ids=do_questions, do_attention_mask=do_attention_mask,
                    ae_images=ae_images, maml_images=maml_images,
                        pattern_embedding=pattern_embedding, entity_embedding=entity_embedding,
                        epoch=1, causal_start_epoch=1, training=False
                    )
                    use_moe_val = bool(int(config.get("use_moe_router", 0))) and getattr(model, "router_trunk", None) is not None
                    if use_moe_val:
                        sub_indices, router_intra, _, _ = moe_probe_router_batch(
                            model, images, questions, attention_mask, fwd_kwargs, config, use_amp
                        )
                        cm = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
                        cm[torch.tensor(sub_indices, device=images.device, dtype=torch.long)] = True
                    else:
                        sub_indices = list(range(batch_size))
                        cm = torch.ones(batch_size, dtype=torch.bool, device=images.device)
                    sub_size = len(sub_indices)
                    fwd_kwargs_intervention = {}
                    for kk, vv in fwd_kwargs.items():
                        if isinstance(vv, torch.Tensor) and vv.dim() > 0 and vv.size(0) == batch_size:
                            fwd_kwargs_intervention[kk] = vv[sub_indices]
                        else:
                            fwd_kwargs_intervention[kk] = vv
                    q_causal, v_causal, causal_stats, sample_weights = compute_causal_masks_from_interventions(
                        model, images[sub_indices], questions[sub_indices], attention_mask[sub_indices], tokenizer, intervention_bank,
                        model, hcss_computer, ccs_computer, pure_encoder=None,
                        device=str(device), seq_len=questions.size(1), num_visual_patches=576,
                        hcss_topk_ratio=config.get("hcss_topk_ratio", 0.4),
                        v_causal_topk_ratio=config.get("v_causal_topk_ratio", 0.4),
                        causal_mask_causal_parts=config.get("causal_mask_causal_parts", True),
                        min_quality_interventions=config.get("min_quality_interventions", 1),
                        min_entity_overlap=config.get("min_entity_overlap", 0.2),
                        sim_low=config.get("sim_low", 0.45), sim_high=config.get("sim_high", 0.90),
                        sim_low_strong=config.get("sim_low_strong", 0.25),
                        overlap_min_strong=config.get("overlap_min_strong", 0.02),
                        relax_sim_low=config.get("relax_sim_low", 0.50),
                        relax_min_entity_overlap=config.get("relax_min_entity_overlap", 0.15),
                        allow_last_resort_interventions=config.get("allow_last_resort_interventions", False),
                        max_interventions=config.get("causal_max_interventions_val_test", config.get("causal_max_interventions", 3)),
                        question_texts=[(batch.get("question_texts") or [])[j] for j in sub_indices] if (batch.get("question_texts") or []) else None,
                        image_paths=[batch.get("image_paths")[j] for j in sub_indices] if batch.get("image_paths") is not None else None,
                        targets=targets[sub_indices],
                        answer_types=[answer_types[j] for j in sub_indices] if answer_types else None,
                        fwd_kwargs=fwd_kwargs_intervention,
                        ccs_mask_ratio=config.get("ccs_mask_ratio", 0.4),
                        ccs_topk_local=config.get("ccs_topk_local", 5),
                        ccs_tau=config.get("ccs_tau", 0.01),
                        local_ie_alpha=config.get("local_ie_alpha", 1.0),
                        ccs_use_local_ie=config.get("ccs_use_local_ie", True),
                        ccs_target=config.get("ccs_target", 0.2),
                        ccs_penalty_lambda=config.get("ccs_penalty_lambda", 0.08),
                        use_feature_gate=True,
                        gate_alpha=config.get("gate_alpha", 1.0),
                        gate_beta=config.get("gate_beta", 0.8),
                        hcss_ie_scale=float(config.get("hcss_ie_scale", 1.5)),
                        sign_adj_temp=float(config.get("sign_adj_temp", 0.05)),
                        hcss_norm_tau=float(config.get("hcss_norm_tau", 0.01)),
                        hcss_floor=float(config.get("hcss_floor", 0.02)),
                    )
                    v_gate = causal_stats.get("v_gate")
                    hcss_per = causal_stats.get("hcss_per_sample", [])
                    ccs_per = causal_stats.get("ccs_per_sample", [])
                    visual_ie_per = causal_stats.get("visual_ie_per_sample", [])
                    text_ie_per = causal_stats.get("text_ie_per_sample", [])
                    text_de_per = causal_stats.get("text_de_per_sample", [])
                    text_hcss_mask = causal_stats.get("text_hcss_mask")
                    if sub_size < batch_size:
                        q_full = torch.ones(batch_size, questions.size(1), device=images.device, dtype=torch.float32)
                        v_full = torch.ones(batch_size, 577, device=images.device, dtype=torch.float32)
                        sw_full = torch.ones(batch_size, device=images.device, dtype=torch.float32)
                        for idx, j in enumerate(sub_indices):
                            q_full[j] = q_causal[idx]
                            v_full[j] = v_causal[idx]
                            sw_full[j] = sample_weights[idx]
                        q_causal, v_causal = q_full, v_full
                        sample_weights = sw_full
                        if v_gate is not None:
                            v_gate_full = torch.ones(batch_size, 577, device=images.device, dtype=torch.float32)
                            for idx, j in enumerate(sub_indices):
                                v_gate_full[j] = v_gate[idx]
                            v_gate = v_gate_full
                        if len(hcss_per) == len(sub_indices) and len(ccs_per) == len(sub_indices):
                            hcss_full = [0.5] * batch_size
                            ccs_full = [0.5] * batch_size
                            for idx, j in enumerate(sub_indices):
                                hcss_full[j] = float(hcss_per[idx])
                                ccs_full[j] = float(ccs_per[idx])
                            hcss_per, ccs_per = hcss_full, ccs_full
                        if len(visual_ie_per) == len(sub_indices) and len(text_de_per) == len(sub_indices):
                            vi_full = [0.5] * batch_size
                            td_full = [0.5] * batch_size
                            ti_full = [0.5] * batch_size
                            tie_sub = text_ie_per if len(text_ie_per) == len(sub_indices) else None
                            for idx, j in enumerate(sub_indices):
                                vi_full[j] = float(visual_ie_per[idx])
                                td_full[j] = float(text_de_per[idx])
                                if tie_sub is not None:
                                    ti_full[j] = float(tie_sub[idx])
                                else:
                                    ti_full[j] = 0.5
                            visual_ie_per, text_de_per, text_ie_per = vi_full, td_full, ti_full
                        if text_hcss_mask is not None and text_hcss_mask.size(0) == sub_size:
                            seq_len = text_hcss_mask.size(1)
                            mask_full = torch.ones(batch_size, seq_len, device=images.device, dtype=text_hcss_mask.dtype)
                            for idx, j in enumerate(sub_indices):
                                mask_full[j] = text_hcss_mask[idx]
                            text_hcss_mask = mask_full
                        ccs_patches_raw = causal_stats.get("ccs_patches")
                        if ccs_patches_raw is not None and ccs_patches_raw.size(0) == sub_size:
                            patch_dim = ccs_patches_raw.size(1)
                            ccs_full = torch.zeros(batch_size, patch_dim, device=images.device, dtype=ccs_patches_raw.dtype)
                            for idx, j in enumerate(sub_indices):
                                ccs_full[j] = ccs_patches_raw[idx]
                            causal_stats["ccs_patches"] = ccs_full
                    if len(text_ie_per) != batch_size and len(text_de_per) == batch_size:
                        text_ie_per = [0.5] * batch_size
                    ccs_patches_tensor = causal_stats.get("ccs_patches")
                    def _p(arr, p):
                        if not arr: return 0.0
                        a = np.array(arr, dtype=np.float64)
                        a = a[np.isfinite(a)]
                        return float(np.percentile(a, p)) if len(a) > 0 else 0.0
                    ablation_no_hcss = config.get("ablation_no_hcss", False)
                    ablation_no_ccs = config.get("ablation_no_ccs", False)
                    use_cem = (len(hcss_per) == batch_size and len(ccs_per) == batch_size and
                              len(visual_ie_per) == batch_size and len(text_ie_per) == batch_size and
                              not ablation_no_hcss and not ablation_no_ccs)
                    use_modal = len(hcss_per) == batch_size and len(ccs_per) == batch_size and not ablation_no_hcss and not ablation_no_ccs
                    if use_cem:
                        p50_vis, p50_hcss, p50_ccs = _p(visual_ie_per, 50), _p(hcss_per, 50), _p(ccs_per, 50)
                        p25_hcss, p75_text = _p(hcss_per, 25), _p(text_ie_per, 75)
                        p50_vis = p50_vis if p50_vis > 0 else 0.50
                        p50_hcss = p50_hcss if p50_hcss > 0 else 0.14
                        p75_text = p75_text if p75_text > 0 else 0.25
                        fwd_gate = {"hcss_per_sample": hcss_per, "ccs_per_sample": ccs_per,
                                   "visual_ie_per_sample": visual_ie_per, "text_ie_per_sample": text_ie_per,
                                   "p50_vis": p50_vis, "p50_hcss": p50_hcss, "p50_ccs": p50_ccs,
                                   "p25_hcss": p25_hcss, "p75_text": p75_text,
                                   "neutral_mask": [False] * batch_size}
                    elif use_modal:
                        p50_ccs = _p(ccs_per, 50) if ccs_per else 0.0
                        p25_hcss = _p(hcss_per, 25) if hcss_per else 0.0
                        fwd_gate = {"hcss_per_sample": hcss_per, "ccs_per_sample": ccs_per,
                                   "p50_ccs": p50_ccs, "p25_hcss": p25_hcss}
                    else:
                        v_gate_val = causal_stats.get("v_gate")
                        fwd_gate = {"v_gate": v_gate_val} if (config.get("use_feature_gate", False) and v_gate_val is not None) else {}
                    if text_hcss_mask is not None and text_hcss_mask.size(0) == batch_size:
                        fwd_gate["text_hcss_mask"] = text_hcss_mask
                    if ccs_patches_tensor is not None and ccs_patches_tensor.size(0) == batch_size:
                        fwd_gate["ccs_patches"] = ccs_patches_tensor
                    # 与训练一致：传入 q_mask_pre/v_mask，否则模型只用 feature gate 无 attention mask
                    use_fg = config.get("use_feature_gate", False)
                    fwd_gate["q_mask_pre"] = q_causal
                    fwd_gate["v_mask"] = torch.ones(batch_size, 577, device=images.device, dtype=torch.float32) if use_fg else v_causal
                    if cm is not None:
                        fwd_gate["causal_path_mask"] = cm.float()
                        fwd_gate["cem_path_scale"] = 1.0
                        if router_intra is not None:
                            fwd_gate["router_intra_gate"] = router_intra
                except Exception as e:
                    logger.warning(f"Validation causal gate failed: {e}, using default")
                    fwd_gate = {}

            fwd = dict(ae_images=ae_images, maml_images=maml_images,
                      pattern_embedding=pattern_embedding, entity_embedding=entity_embedding,
                      do_questions_ids=do_questions, do_attention_mask=do_attention_mask,
                      epoch=1, causal_start_epoch=1, training=False,
                      cem_gt_indices=targets.argmax(dim=1).long(),
                      **fwd_gate)
            if bool(config.get("use_do_controller", True)) and use_offline_causal:
                cached_signals = _get_cached_signals_for_batch(
                    batch,
                    offline_cache,
                    images.device,
                    fusion_bank_dim=int(config.get("fusion_bank_dim", 0) or 0),
                )
                fwd.update(causal_signals=cached_signals, apply_do=True)
            with torch.amp.autocast('cuda',enabled=use_amp):
                out = model(images, questions, attention_mask, **fwd)
                if isinstance(out, tuple) and len(out) >= 2:
                    logits = out[0]
                    if getattr(model, 'use_open_embedding_matching', False):
                        open_logits = out[1]
                        concept_logits = out[3] if len(out) > 3 else None
                    elif getattr(model, 'use_open_concept_head', False):
                        open_logits = None
                        concept_logits = out[1]
                    else:
                        open_logits = out[1]
                        concept_logits = None
                else:
                    logits, open_logits, concept_logits = out, None, None
            use_open_emb = (open_logits is not None and getattr(model, 'use_open_embedding_matching', False))
            if use_open_emb:
                open_mask = torch.tensor([((answer_types[i] if i < len(answer_types) else "") or "").strip().lower() == 'open' for i in range(batch_size)], device=logits.device, dtype=torch.bool)
                combined = logits.clone()
                if open_mask.any():
                    combined[open_mask] = open_logits[open_mask]
                loss = criterion(combined, targets)
                score_logits = combined
            else:
                loss = criterion(logits, targets)
                score_logits = logits

            batch_scores = compute_score_with_logits(score_logits, targets)
            batch_score = batch_scores.sum().item()

            total_loss += loss.item() * batch_size
            total_score += batch_score
            total_samples += batch_size

            for i, score in enumerate(batch_scores):
                s = score.sum().item() if score.dim() > 0 else score.item()
                if i < len(answer_types):
                    cat_key = answer_types[i].lower().strip()
                    if cat_key in cat_correct:
                        cat_correct[cat_key] += s
                        cat_total[cat_key] += 1

            # Concept accuracy (open & non-misc, 硬标签)
            if concept_logits is not None:
                from utils.vqa_rad_concept import MISC_IDX
                concept_indices = batch.get("concept_indices", [])
                if len(concept_indices) >= batch_size:
                    open_mask = torch.tensor([((answer_types[i] if i < len(answer_types) else "") or "").strip().lower() == 'open' for i in range(batch_size)], device=concept_logits.device, dtype=torch.bool)
                    cidx = torch.tensor(concept_indices[:batch_size], dtype=torch.long, device=concept_logits.device)
                    non_misc = (cidx != MISC_IDX)
                    eval_mask = open_mask & non_misc
                    if eval_mask.any():
                        pred_c = concept_logits[eval_mask].argmax(dim=1)
                        gt_c = cidx[eval_mask]
                        total_concept_correct += (pred_c == gt_c).sum().item()
                        total_concept_count += eval_mask.sum().item()

    avg_loss = total_loss / total_samples
    avg_score = total_score / total_samples * 100

    logger.info(f"Validation: Loss {avg_loss:.4f} | All {avg_score:.2f}%")
    if total_concept_count > 0:
        concept_acc = total_concept_correct / total_concept_count * 100
        logger.info(f"  Open concept accuracy: {concept_acc:.2f}% ({total_concept_correct}/{total_concept_count})")
    parts = []
    for cat in category_names:
        if cat_total[cat] > 0:
            acc = cat_correct[cat] / cat_total[cat] * 100
            parts.append(f"{cat.capitalize()} {acc:.1f}%")
    if parts:
        logger.info(f"  Per-category: {' | '.join(parts)}")

    return avg_loss, avg_score
