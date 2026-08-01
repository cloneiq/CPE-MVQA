#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 interventions 的因果 mask 构造
使用 realtime_intervention_generator 实时生成反事实，realtime_causal_effects 计算 HCSS(混合定义)/CCS
"""
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

def compute_causal_masks_from_interventions(
    model,
    images,
    questions_ids,
    attention_mask,
    tokenizer,
    intervention_bank: Optional[Dict],
    causal_model,
    hcss_computer,
    ccs_computer,
    pure_encoder=None,
    device: str = "cuda",
    seq_len: int = 32,
    num_visual_patches: int = 576,
    hcss_topk_ratio: float = 0.4,
    v_causal_topk_ratio: float = 0.4,
    causal_mask_causal_parts: bool = True,
    min_quality_interventions: int = 1,
    min_entity_overlap: float = 0.2,
    sim_low: float = 0.45,
    sim_high: float = 0.90,
    sim_low_strong: float = 0.25,
    overlap_min_strong: float = 0.02,
    relax_sim_low: float = 0.50,
    relax_min_entity_overlap: float = 0.15,
    allow_last_resort_interventions: bool = False,
    max_interventions: int = 5,
    question_texts: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    targets: Optional[torch.Tensor] = None,
    answer_types: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
    fwd_kwargs: Optional[Dict] = None,
    ccs_negative_weight: float = 0.0,
    ccs_negative_weight_min: float = 0.3,
    sign_adj_margin: float = 0.05,
    ccs_mask_ratio: float = 0.5,
    ccs_topk_local: int = 5,
    ccs_tau: float = 0.01,
    local_ie_alpha: float = 1.0,
    ccs_use_local_ie: bool = True,
    ccs_target: float = 0.2,
    ccs_penalty_lambda: float = 0.08,
    use_feature_gate: bool = False,
    gate_alpha: float = 1.0,
    gate_beta: float = 0.8,
    ablation_no_hcss: bool = False,
    ablation_no_ccs: bool = False,
    hcss_ie_scale: float = 1.0,
    sign_adj_temp: float = 0.05,
    hcss_norm_tau: float = 0.01,
    hcss_floor: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
    """
    用 interventions + realtime HCSS/CCS 构造 Q_causal 和 V_causal
    诊断: stats 含 interventions_empty_cnt, exception_cnt, ccs_patches_none_cnt
    公式 B: HCSS = SignAdj × clip(core)；core = μ·IE_mean·(1−DE_mean)·hcss_ie_scale + βΔH' − θVar（DE 为文本 DE 均值，压语言捷径）
    - SignAdj=0: 因果方向不一致，token 关闭（d_mean < -margin）
    - SignAdj=1: 因果方向一致，F_i^text = HCSS_i · F_i（d_mean >= -margin）
    CCS：样本级/patch 级均为 compute_CCS(·,·) 第二路传入文本 DE（与 HCSS 的 ie_mean 分离）。
    CCS<0: 视觉路径不成立，样本 loss 权重=max(ccs_negative_weight, ccs_negative_weight_min)
    避免 SignAdj=0 且 CCS<0 时双重关闭导致梯度过弱

    Returns:
        q_causal_mask: (B, seq_len) 1=保留
        v_causal_mask: (B, 577) 1=保留；use_feature_gate 时可为全1（由 v_gate 接管）
        stats: {hcss, ccs, ..., v_gate} 样本均值；v_gate 为 [B,577] 当 use_feature_gate
        sample_weights: (B,) CCS>=0 为 1.0，CCS<0 为 ccs_negative_weight
    """
    from pipeline.realtime_intervention_generator import get_interventions_for_sample_realtime
    from pipeline.realtime_causal_effects import compute_hcss_realtime, compute_ccs_realtime

    batch_size = images.size(0)
    fwd_kwargs = fwd_kwargs or {}
    q_list = []
    v_list = []
    sample_weights_list = []
    stats_hcss, stats_ccs, stats_vis_ie, stats_vis_ie_g, stats_vis_ie_l, stats_text_ie, stats_text_de, stats_sign_adj = [], [], [], [], [], [], [], []
    text_hcss_list: List[torch.Tensor] = []
    ccs_patches_list: List[torch.Tensor] = []
    interventions_empty_cnt, exception_cnt, ccs_patches_none_cnt = 0, 0, 0
    last_exception_msg = ""
    empty_reason_counts: Dict[str, int] = {}
    v_gate_list: List[torch.Tensor] = [] if use_feature_gate else []

    for i in range(batch_size):
        try:
            sample_q = ""
            if question_texts is not None and i < len(question_texts):
                sample_q = (question_texts[i] or "").strip()
            if not sample_q and tokenizer is not None:
                sample_q = tokenizer.decode(questions_ids[i], skip_special_tokens=True)

            ans_idx = 0
            if targets is not None:
                ans_idx = torch.argmax(targets[i]).item()

            image_name = ""
            if image_paths is not None and i < len(image_paths):
                image_name = os.path.basename(image_paths[i] or "")

            # answer_type 必须为 GT（来自 batch.answer_types / 数据集 annotation），不可用预测
            # 不修改 answer_type：重标为 abnormality 会破坏训练/评估一致性，用 concept_focus_flag 代替
            answer_type = ""
            if answer_types is not None and i < len(answer_types):
                answer_type = (answer_types[i] or "").strip().lower()

            resp = get_interventions_for_sample_realtime(
                image_name,
                sample_q,
                intervention_bank,
                answer_type=answer_type,
                model=causal_model,
                tokenizer=tokenizer,
                pure_encoder=pure_encoder,
                device="cpu",
                max_interventions=max_interventions,
                use_realtime_fallback=True,
                min_quality_interventions=min_quality_interventions,
                min_entity_overlap=min_entity_overlap,
                sim_low=sim_low,
                sim_high=sim_high,
                sim_low_strong=sim_low_strong,
                overlap_min_strong=overlap_min_strong,
                relax_on_empty=True,
                relax_sim_low=relax_sim_low,
                relax_min_entity_overlap=relax_min_entity_overlap,
                allow_last_resort_interventions=allow_last_resort_interventions,
                return_metadata=True,
            )
            interventions = resp.get("interventions", []) if isinstance(resp, dict) else []
            no_update = resp.get("no_causal_update", False) if isinstance(resp, dict) else False

            if no_update or not interventions:
                interventions_empty_cnt += 1
                if use_feature_gate:
                    v_gate_list.append(torch.ones(1, num_visual_patches + 1, device=device, dtype=torch.float32))
                text_hcss_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))
                reason = (resp or {}).get("diag_empty_reason", "unknown")
                empty_reason_counts[reason] = empty_reason_counts.get(reason, 0) + 1
                ccs_patches_list.append(torch.zeros(1, num_visual_patches + 1, device=device, dtype=torch.float32))
                q_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))
                v_list.append(torch.ones(1, num_visual_patches + 1, device=device, dtype=torch.float32))
                sample_weights_list.append(1.0)
                stats_hcss.append(0.0)
                stats_ccs.append(0.0)
                stats_vis_ie.append(0.0)
                stats_vis_ie_g.append(0.0)
                stats_vis_ie_l.append(0.0)
                stats_text_ie.append(0.0)
                stats_text_de.append(0.0)
                stats_sign_adj.append(0.0)
                continue

            img_i = images[i : i + 1]
            q_i = questions_ids[i : i + 1]
            attn_i = attention_mask[i : i + 1]
            fwd_i = {}
            for k, v in fwd_kwargs.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == images.size(0):
                    fwd_i[k] = v[i : i + 1]
                else:
                    fwd_i[k] = v
            with torch.no_grad():
                out_i = causal_model(img_i, q_i, attn_i, **fwd_i)
                logits_i = out_i[0] if isinstance(out_i, tuple) else out_i

            hcss_res = compute_hcss_realtime(
                causal_model,
                img_i,
                sample_q,
                interventions,
                tokenizer,
                ans_idx,
                logits_i,
                hcss_computer,
                answer_group=answer_type or None,
                answer_idx_in_type=None,
                device=device,
                min_interventions=1,
                max_interventions=max_interventions,
                sign_adj_margin=sign_adj_margin,
                hcss_ie_scale=hcss_ie_scale,
                sign_adj_temp=sign_adj_temp,
                hcss_norm_tau=hcss_norm_tau,
                hcss_floor=hcss_floor,
            )
            if ablation_no_ccs:
                # 强消融 No CCS: 不计算 CCS，v_mask=全1，v_gate 仅用 HCSS
                ccs_res = {
                    "ccs_patches": None,
                    "ccs": 0.0,
                    "visual_ie": 0.0,
                    "visual_ie_g": 0.0,
                    "visual_ie_l": 0.0,
                    "text_de": 0.0,
                }
            else:
                ccs_res = compute_ccs_realtime(
                    causal_model,
                    img_i,
                    sample_q,
                    interventions,
                    tokenizer,
                    ans_idx,
                    logits_i,
                    ccs_computer,
                    answer_group=answer_type or None,
                    answer_idx_in_type=None,
                    mask_ratio_global=ccs_mask_ratio,
                    topk_local=ccs_topk_local,
                    tau=ccs_tau,
                    local_ie_alpha_scale=local_ie_alpha,
                    use_local_ie=ccs_use_local_ie,
                    device=device,
                    max_interventions=max_interventions,
                    precomputed_text_de=hcss_res.get("de_mean"),
                )

            # 公式 B: SignAdj=0 -> token 关闭；SignAdj=1 -> F_i^text = HCSS_i · F_i
            # 强消融 No HCSS: q_mask=全1，text_hcss=全1，HCSS 不参与 gate/loss/CEM
            if ablation_no_hcss:
                q_mask = torch.ones(1, seq_len, device=device, dtype=torch.float32)
                text_hcss_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))
            else:
                sign_adj = float(hcss_res.get("sign_adj", 1.0))
                hcss_core = float(hcss_res.get("hcss_core", 0.0))
                text_hcss = hcss_res.get("text_hcss")
                if sign_adj <= 0.0:
                    q_mask = torch.zeros(1, seq_len, device=device, dtype=torch.float32)
                    q_mask[:, 0] = 1.0
                    text_hcss_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))
                elif text_hcss is not None:
                    th = text_hcss.squeeze().to(device)
                    if th.dim() == 1:
                        th = th[:seq_len]
                    pad_len = seq_len - th.size(0)
                    if pad_len > 0:
                        th = F.pad(th, (0, pad_len), value=0.0)
                    # SignAdj=1: 使用 HCSS 缩放，map [0, 0.5] -> [0.5, 1.0]
                    th_scaled = (0.5 + th.clamp(0.0, 0.5)).clamp(0.0, 1.0)
                    q_mask = th_scaled.unsqueeze(0)
                    q_mask[:, 0] = 1.0
                    # Text Debias Mask: sigmoid(4*(token_hcss-0.15))，医学文本短、过强 debias 会删有用语义
                    debias_mask = torch.sigmoid(4.0 * (th.clamp(0.0, 1.0) - 0.15)).unsqueeze(0)
                    debias_mask[:, 0] = 1.0  # CLS 保留
                    text_hcss_list.append(debias_mask)
                else:
                    hcss_scalar = float(hcss_res.get("hcss_scalar", 0.0))
                    hcss_norm = (np.tanh(hcss_scalar) + 1.0) / 2.0
                    q_keep = 0.3 + 0.7 * hcss_norm
                    q_mask = torch.full((1, seq_len), q_keep, device=device, dtype=torch.float32)
                    q_mask[:, 0] = 1.0
                    if attn_i is not None:
                        valid = attn_i[0].float().to(device)
                        q_mask = q_mask * valid
                    text_hcss_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))

            ccs_patches = ccs_res.get("ccs_patches")
            hcss_scalar = float(hcss_res.get("hcss_scalar", 0.0))
            if ccs_patches is None or ccs_patches.numel() == 0:
                ccs_patches_none_cnt += 1
            # CCS<0: 视觉路径不成立，样本 loss 权重降低（提前获取 ccs_val 供 gate 使用）
            ccs_val = float(ccs_res.get("ccs", 0.0))
            if use_feature_gate:
                if ccs_patches is not None and ccs_patches.numel() > 0:
                    cp = ccs_patches.squeeze().float().to(device)
                    if cp.dim() == 0:
                        cp = cp.unsqueeze(0)
                    n = cp.size(0)
                    if n > num_visual_patches:
                        cp = cp[:num_visual_patches]
                    elif n < num_visual_patches:
                        cp = F.pad(cp, (0, num_visual_patches - n), value=0.0)
                    hcss_broadcast = torch.full_like(cp, hcss_scalar, device=device, dtype=torch.float32)
                    # 强消融: No HCSS 时 gate 仅用 CCS (gate_beta=0); No CCS 时 gate 仅用 HCSS (gate_alpha=0)
                    _ga = 0.0 if ablation_no_ccs else gate_alpha
                    _gb = 0.0 if ablation_no_hcss else gate_beta
                    gate_patches = torch.sigmoid(_ga * cp + _gb * hcss_broadcast)
                    # gate 基于分位数: scale = 1 + 0.3*(CCS-P50)/(P75-P25)；No CCS 时 scale=1 不随 CCS 变化
                    scale_val = 1.0
                    if not ablation_no_ccs:
                        P25, P50, P75 = 0.25, 0.40, 0.55
                        scale_val = 1.0 + 0.3 * (ccs_val - P50) / max(P75 - P25, 1e-6)
                        scale_val = max(0.85, min(1.15, scale_val))
                    gate_patches = (gate_patches * scale_val).clamp(0.05, 1.0)  # 饱和保护防全0
                    gate_cls = torch.ones(1, 1, device=device, dtype=torch.float32)
                    v_g = torch.cat([gate_cls, gate_patches.unsqueeze(0)], dim=1)
                elif ablation_no_ccs:
                    # No CCS: v_gate 仅用 HCSS
                    hcss_broadcast = torch.full((num_visual_patches,), hcss_scalar, device=device, dtype=torch.float32)
                    gate_patches = torch.sigmoid(gate_beta * hcss_broadcast).clamp(0.05, 1.0)
                    gate_cls = torch.ones(1, 1, device=device, dtype=torch.float32)
                    v_g = torch.cat([gate_cls, gate_patches.unsqueeze(0)], dim=1)
                else:
                    v_g = torch.ones(1, num_visual_patches + 1, device=device, dtype=torch.float32)
                v_gate_list.append(v_g)
            if ccs_patches is not None and ccs_patches.numel() > 0:
                cp = ccs_patches.squeeze().to(device)
                if cp.dim() == 0:
                    cp = cp.unsqueeze(0)
                n = cp.size(0)
                if n > num_visual_patches:
                    cp = cp[:num_visual_patches]
                elif n < num_visual_patches:
                    cp = F.pad(cp, (0, num_visual_patches - n), value=float("-inf"))
                k = max(1, int(num_visual_patches * v_causal_topk_ratio))
                _, top_idx = torch.topk(cp, k)
                v_mask = torch.zeros(1, num_visual_patches + 1, device=device, dtype=torch.float32)
                v_mask[:, 0] = 1.0
                patch_indices = (top_idx + 1).long()
                if causal_mask_causal_parts:
                    v_mask[:, patch_indices] = 1.0
                else:
                    v_mask[:, 1:] = 1.0
                    v_mask[:, patch_indices] = 0.0
            else:
                v_mask = torch.ones(1, num_visual_patches + 1, device=device, dtype=torch.float32)

            # 收集 ccs_patches 用于 patch gate（soft_gate(ccs_patch, 4.0, 0.0)）
            if ccs_patches is not None and ccs_patches.numel() > 0:
                cp = ccs_patches.squeeze().float().to(device)
                if cp.dim() == 0:
                    cp = cp.unsqueeze(0)
                n = cp.size(0)
                if n >= num_visual_patches + 1:
                    cp = cp[:num_visual_patches + 1]
                else:
                    # 通常 n=576，需 prepend CLS(1.0) 得到 [577]
                    cls_val = torch.ones(1, device=cp.device, dtype=cp.dtype)
                    cp = torch.cat([cls_val, cp], dim=0)
                    if cp.size(0) < num_visual_patches + 1:
                        cp = F.pad(cp, (0, num_visual_patches + 1 - cp.size(0)), value=0.0)
                ccs_patches_list.append(cp.unsqueeze(0))
            else:
                ccs_patches_list.append(torch.zeros(1, num_visual_patches + 1, device=device, dtype=torch.float32))

            # CCS<0: 视觉路径不成立，样本 loss 权重降低（ccs_val 已在上方 gate 处获取）
            if ccs_val >= 0:
                sw = 1.0
            else:
                sw = max(ccs_negative_weight, ccs_negative_weight_min)
            # ccs_target + ccs_penalty_lambda: 偏离目标的样本降权
            if ccs_penalty_lambda > 0:
                penalty_scale = max(0.5, 1.0 - ccs_penalty_lambda * (ccs_val - ccs_target) ** 2)
                sw = sw * penalty_scale
            sample_weights_list.append(sw)

            q_list.append(q_mask)
            v_list.append(v_mask)
            stats_hcss.append(float(hcss_res.get("hcss_scalar", 0.0)))
            stats_text_ie.append(float(hcss_res.get("ie_mean", 0.0)))
            stats_ccs.append(ccs_val)
            stats_vis_ie.append(float(ccs_res.get("visual_ie", 0.0)))
            vg = float(ccs_res.get("visual_ie_g", 0.0))
            vl = float(ccs_res.get("visual_ie_l", 0.0))
            stats_vis_ie_g.append(0.0 if (np.isnan(vg) or np.isinf(vg)) else vg)
            stats_vis_ie_l.append(0.0 if (np.isnan(vl) or np.isinf(vl)) else vl)
            stats_text_de.append(float(ccs_res.get("text_de", 0.0)))
            stats_sign_adj.append(sign_adj)
        except Exception as e:
            exception_cnt += 1
            last_exception_msg = f"{type(e).__name__}: {str(e)[:80]}"
            ccs_patches_list.append(torch.zeros(1, num_visual_patches + 1, device=device, dtype=torch.float32))
            q_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))
            v_list.append(torch.ones(1, num_visual_patches + 1, device=device, dtype=torch.float32))
            text_hcss_list.append(torch.ones(1, seq_len, device=device, dtype=torch.float32))
            if use_feature_gate:
                v_gate_list.append(torch.ones(1, num_visual_patches + 1, device=device, dtype=torch.float32))
            sample_weights_list.append(1.0)
            stats_hcss.append(0.0)
            stats_ccs.append(0.0)
            stats_vis_ie.append(0.0)
            stats_vis_ie_g.append(0.0)
            stats_vis_ie_l.append(0.0)
            stats_text_ie.append(0.0)
            stats_text_de.append(0.0)
            stats_sign_adj.append(0.0)

    q_causal = torch.cat(q_list, dim=0)
    v_causal = torch.cat(v_list, dim=0)
    v_gate_tensor = torch.cat(v_gate_list, dim=0) if use_feature_gate and len(v_gate_list) == batch_size else None
    text_hcss_mask = torch.cat(text_hcss_list, dim=0) if len(text_hcss_list) == batch_size else None
    sample_weights = torch.tensor(sample_weights_list, device=device, dtype=torch.float32)
    n = len(stats_hcss)
    # 使用 nanmean 防止 nan 污染；空列表时返回 0
    def _safe_mean(arr):
        if not arr:
            return 0.0
        a = np.array(arr, dtype=np.float64)
        return float(np.nanmean(a)) if np.any(np.isfinite(a)) else 0.0
    stats = {
        "hcss": _safe_mean(stats_hcss),
        "ccs": _safe_mean(stats_ccs),
        "ccs_per_sample": stats_ccs,  # 用于 selective invariance、abnormal ratio
        "hcss_per_sample": stats_hcss,  # 用于 abnormal HCSS 软区间、ratio
        "visual_ie_per_sample": stats_vis_ie,  # CEM: 视觉证据
        "text_de_per_sample": stats_text_de,  # 诊断 / 旧兼容；CEM 主通路用 text_ie
        "text_ie_per_sample": stats_text_ie,  # CEM 与 Ĉ 监督：文本 IE（非 DE）
        "sign_adj_per_sample": stats_sign_adj,  # 每样本 SignAdj（0/1）
        "v_gate": v_gate_tensor,
        "text_hcss_mask": text_hcss_mask,  # [B, seq_len] Text Debias: 抑制非因果 token
        "ccs_patches": torch.cat(ccs_patches_list, dim=0) if len(ccs_patches_list) == batch_size else None,  # [B, 577] for patch gate
        "interventions_empty_cnt": interventions_empty_cnt,
        "exception_cnt": exception_cnt,
        "ccs_patches_none_cnt": ccs_patches_none_cnt,
        "last_exception_msg": last_exception_msg,
        "empty_reason_counts": empty_reason_counts,
        "avg_visual_ie": _safe_mean(stats_vis_ie),
        "avg_visual_ie_g": _safe_mean(stats_vis_ie_g),
        "avg_visual_ie_l": _safe_mean(stats_vis_ie_l),
        "avg_text_ie": _safe_mean(stats_text_ie),
        "avg_text_de": _safe_mean(stats_text_de),
        "sign_adj_ratio": _safe_mean(stats_sign_adj),
    }
    return q_causal, v_causal, stats, sample_weights
