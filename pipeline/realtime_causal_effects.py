#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时计算因果效应模块 (适配 vqa_module.py + Two-Stage Logic)
用于在训练过程中实时计算文本直接效应(DE)、文本间接效应(IE)和视觉间接效应(Visual IE)
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import os

from models.causal_modules import HCSSComputer, CCSComputer, GateNetwork
from interventions.gradcam_intervention import generate_masked_embeddings, mask_image_embeds

# ------------------------ 核心：手动两阶段前向传播辅助函数 ------------------------
def get_model_layers(model):
    """获取并将层分为两个阶段 (0-2层, 3-5层)"""
    text_layers = model.multi_modal_language_layers
    vision_layers = model.multi_modal_vision_layers
    
    num_layers = len(text_layers)
    mid = num_layers // 2 
    
    stage1_layers = list(zip(text_layers[:mid], vision_layers[:mid]))
    stage2_layers = list(zip(text_layers[mid:], vision_layers[mid:]))
    
    return stage1_layers, stage2_layers

def manual_encode_text(model, input_ids, attention_mask):
    """手动执行文本编码 + Modality Embedding"""
    uni_modal_text_feats = model.language_encoder.embeddings(input_ids=input_ids)
    text_input_shape = attention_mask.size()
    extended_text_masks = model.language_encoder.get_extended_attention_mask(
        attention_mask, text_input_shape, input_ids.device
    )
    for layer in model.language_encoder.encoder.layer:
        uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
    uni_modal_text_feats = model.multi_modal_language_proj(uni_modal_text_feats)
    
    uni_modal_text_feats = uni_modal_text_feats + \
        model.modality_type_embeddings(torch.zeros_like(attention_mask))
        
    return uni_modal_text_feats, extended_text_masks

def manual_encode_image(model, images):
    """手动执行图像编码 + Modality Embedding"""
    uni_modal_image_feats = model.vision_encoder(images)
    uni_modal_image_feats = model.multi_modal_vision_proj(uni_modal_image_feats)
    
    image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), 
                             dtype=torch.long, device=images.device)
    extended_image_masks = model.language_encoder.get_extended_attention_mask(
        image_masks, image_masks.size(), images.device
    )
    
    image_token_type_idx = 1
    uni_modal_image_feats = uni_modal_image_feats + \
        model.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
        
    return uni_modal_image_feats, extended_image_masks

def run_fusion_layers(layers_list, x, y, ext_text_masks, ext_image_masks):
    """运行指定的一组 Co-Attention 层"""
    for text_layer, image_layer in layers_list:
        x1 = text_layer(x, y, ext_text_masks, ext_image_masks, output_attentions=True)
        y1 = image_layer(y, x, ext_image_masks, ext_text_masks, output_attentions=True)
        x, y = x1[0], y1[0]
    return x, y

def final_prediction(model, x, y, ext_text_masks, ext_image_masks, answer_group=None, device='cuda'):
    """执行后处理、池化和分类"""
    if hasattr(model, "multi_modal_vision_post_layers") and len(model.multi_modal_vision_post_layers) > 0:
        for post_image_layer in model.multi_modal_vision_post_layers:
            y1 = post_image_layer(y, x, ext_image_masks, ext_text_masks)
            y = y1[0]
        
    multi_modal_text_cls_feats = model.multi_modal_language_pooler(x)
    multi_modal_image_cls_feats = model.multi_modal_vision_pooler(y)
    
    multi_modal_cls_feats = torch.cat(
        [multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
        
    # Current model uses a single unified classifier head.
    # Keep answer_group arg for compatibility with older call sites.
    logits = model.vqa_head(multi_modal_cls_feats)
    return logits

# ------------------------ 1. 计算 Text Direct Effect (DE) ------------------------
def compute_text_direct_effect(
    model,
    image: torch.Tensor,
    question: str,
    q_star: str,
    tokenizer,
    answer_idx: int,
    logits_orig_target: torch.Tensor,
    answer_group: Optional[str] = None,
    answer_idx_in_type: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    两阶段逻辑 Text DE:
    Stage 1: Fusion(Orig Vision, Orig Text) -> x_mid, y_mid
    Stage 2: Fusion(y_mid [Vision], Intervened Text [Text]) -> Logits
    """
    with torch.no_grad(): # 训练时不需要此处的梯度，但模型参数是实时的
        stage1_layers, stage2_layers = get_model_layers(model)
        
        # 1. 编码
        feat_v_orig, ext_mask_v = manual_encode_image(model, image)
        
        enc_orig = tokenizer(question, return_tensors='pt', padding='longest', truncation=True, max_length=64).to(device)
        feat_q_orig, ext_mask_q_orig = manual_encode_text(model, enc_orig.input_ids, enc_orig.attention_mask)
        
        enc_star = tokenizer(q_star, return_tensors='pt', padding='longest', truncation=True, max_length=64).to(device)
        feat_q_star, ext_mask_q_star = manual_encode_text(model, enc_star.input_ids, enc_star.attention_mask)
        
        # 2. Stage 1: Orig Vision + Orig Text
        x1, y1 = run_fusion_layers(stage1_layers, feat_q_orig, feat_v_orig, ext_mask_q_orig, ext_mask_v)
        
        # 3. Stage 2: Stage 1 Vision Output (y1) + Intervened Text (feat_q_star)
        x2, y2 = run_fusion_layers(stage2_layers, feat_q_star, y1, ext_mask_q_star, ext_mask_v)
        
        # 4. 预测
        logits_de = final_prediction(model, x2, y2, ext_mask_q_star, ext_mask_v, answer_group, device)
        
        # 5. 提取结果
        idx = answer_idx_in_type if answer_idx_in_type is not None else answer_idx
        idx = torch.clamp(torch.tensor(idx, device=device), 0, logits_de.shape[1]-1).item()
        
        de_logit_target = logits_de[:, idx].squeeze()
        if logits_orig_target.dim() == 2:
            orig_logit_target = logits_orig_target[:, idx].squeeze()
        else:
            orig_logit_target = logits_orig_target.squeeze()
        de_diff = de_logit_target - orig_logit_target
        de_prob = F.softmax(logits_de, dim=-1).squeeze(0)
        
        return {"logit": de_logit_target, "diff": de_diff, "prob": de_prob}

# ------------------------ 2. 计算 Text Indirect Effect (IE) ------------------------
def compute_text_indirect_effect(
    model,
    image: torch.Tensor,
    question: str,
    q_star: str,
    tokenizer,
    answer_idx: int,
    logits_orig_target: torch.Tensor,
    answer_group: Optional[str] = None,
    answer_idx_in_type: Optional[int] = None,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    两阶段逻辑 Text IE:
    Stage 1: Fusion(Orig Vision, Intervened Text) -> x_mid, y_mid
    Stage 2: Fusion(y_mid [Vision], Orig Text [Text]) -> Logits
    """
    with torch.no_grad():
        stage1_layers, stage2_layers = get_model_layers(model)
        
        # 1. 编码
        feat_v_orig, ext_mask_v = manual_encode_image(model, image)
        
        enc_orig = tokenizer(question, return_tensors='pt', padding='longest', truncation=True, max_length=64).to(device)
        feat_q_orig, ext_mask_q_orig = manual_encode_text(model, enc_orig.input_ids, enc_orig.attention_mask)
        
        enc_star = tokenizer(q_star, return_tensors='pt', padding='longest', truncation=True, max_length=64).to(device)
        feat_q_star, ext_mask_q_star = manual_encode_text(model, enc_star.input_ids, enc_star.attention_mask)
        
        # 2. Stage 1: Orig Vision + Intervened Text (Vision被误导)
        x1, y1 = run_fusion_layers(stage1_layers, feat_q_star, feat_v_orig, ext_mask_q_star, ext_mask_v)
        
        # 3. Stage 2: Stage 1 Vision Output (y1) + Orig Text (Vision带着偏见回来找原问题)
        x2, y2 = run_fusion_layers(stage2_layers, feat_q_orig, y1, ext_mask_q_orig, ext_mask_v)
        
        # 4. 预测
        logits_ie = final_prediction(model, x2, y2, ext_mask_q_orig, ext_mask_v, answer_group, device)
        
        # 5. 提取结果
        idx = answer_idx_in_type if answer_idx_in_type is not None else answer_idx
        idx = torch.clamp(torch.tensor(idx, device=device), 0, logits_ie.shape[1]-1).item()
        
        ie_logit_target = logits_ie[:, idx].squeeze()
        if logits_orig_target.dim() == 2:
            orig_logit_target = logits_orig_target[:, idx].squeeze()
        else:
            orig_logit_target = logits_orig_target.squeeze()
        ie_diff = ie_logit_target - orig_logit_target
        ie_prob = F.softmax(logits_ie, dim=-1).squeeze(0)
        
        return {"logit": ie_logit_target, "diff": ie_diff, "prob": ie_prob}

# ------------------------ 3. 计算 Visual Indirect Effect (Visual IE) ------------------------
def compute_visual_indirect_effect(
    model,
    image: torch.Tensor,
    question: str,
    tokenizer,
    answer_idx: int,
    answer_group: Optional[str] = None,
    answer_idx_in_type: Optional[int] = None,
    mask_ratio_global: float = 0.5,
    topk_local: int = 5,
    tau: float = 0.01,
    local_ie_alpha_scale: float = 1.0,
    use_local_ie: bool = True,
    mask_mode: str = "zero",
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Visual IE = (IE_g + alpha * IE_l) / (1 + alpha)
    - IE_g: Global (mask top 50%)
    - IE_l: Local (mask top-k patches, stable gating), 可关闭以节省 ~1 次 forward
    - alpha: 2.0 if abnormality else 0.5

    ⚠ answer_group 必须为 GT（来自 batch.answer_types），不能用预测类别。
    """
    def _target_prob_from_logits(logits: torch.Tensor, idx: int) -> float:
        probs = F.softmax(logits, dim=-1)
        if probs.dim() == 2:
            return probs[0, idx].item()
        return probs[idx].item()

    def _run_masked_forward(raw_v, patch_scores, mask_ratio=None, topk=None):
        raw_v_no_cls = raw_v[:, 1:, :]
        masked_v_no_cls = mask_image_embeds(
            raw_v_no_cls, patch_scores, mask_ratio=mask_ratio or 0.5, mask_mode=mask_mode, topk=topk
        )
        masked_raw_v = torch.cat([raw_v[:, 0:1, :], masked_v_no_cls], dim=1)
        feat_v_masked = model.multi_modal_vision_proj(masked_raw_v)
        image_masks = torch.ones((feat_v_masked.size(0), feat_v_masked.size(1)), dtype=torch.long, device=device)
        extended_mask_v_masked = model.language_encoder.get_extended_attention_mask(image_masks, image_masks.size(), device)
        feat_v_masked = feat_v_masked + model.modality_type_embeddings(torch.full_like(image_masks, 1))
        x1, y1 = run_fusion_layers(stage1_layers, feat_q_orig, feat_v_masked, ext_mask_q_orig, extended_mask_v_masked)
        x2, y2 = run_fusion_layers(stage2_layers, x1, feat_v_orig, ext_mask_q_orig, ext_mask_v_orig)
        return final_prediction(model, x2, y2, ext_mask_q_orig, ext_mask_v_orig, answer_group, device)

    with torch.no_grad():
        stage1_layers, stage2_layers = get_model_layers(model)
        
        # 1. 编码文本
        enc_orig = tokenizer(question, return_tensors='pt', padding='longest', truncation=True, max_length=64).to(device)
        feat_q_orig, ext_mask_q_orig = manual_encode_text(model, enc_orig.input_ids, enc_orig.attention_mask)
        
        # 2. 原始图像预测
        feat_v_orig, ext_mask_v_orig = manual_encode_image(model, image)
        x_std, y_std = run_fusion_layers(stage1_layers, feat_q_orig, feat_v_orig, ext_mask_q_orig, ext_mask_v_orig)
        x_std, y_std = run_fusion_layers(stage2_layers, feat_q_orig, y_std, ext_mask_q_orig, ext_mask_v_orig)
        logits_orig = final_prediction(model, x_std, y_std, ext_mask_q_orig, ext_mask_v_orig, answer_group, device)
        
        idx = answer_idx_in_type if answer_idx_in_type is not None else answer_idx
        idx = torch.clamp(torch.tensor(idx, device=device), 0, logits_orig.shape[1]-1).item()
        orig_prob = _target_prob_from_logits(logits_orig, idx)
        
        # 3. GradCAM patch scores
        q_enc_dict = {"input_ids": enc_orig.input_ids, "attention_mask": enc_orig.attention_mask}
        sample_data = [{"answer_idx_in_type": idx, "answer_group": answer_group, "meta_batch_idx": 0}]
        mask_result = generate_masked_embeddings(
            model=model, image=image, q_enc=q_enc_dict,
            mask_ratio=mask_ratio_global, mask_mode=mask_mode, device=device,
            answer_targets=torch.tensor([idx], device=device),
            tokenizer=tokenizer, sample_data=sample_data
        )
        patch_scores = mask_result.get("patch_scores")
        mask_meta = mask_result.get("meta", {}) if isinstance(mask_result, dict) else {}
        gradcam_success = bool(mask_meta.get("success", False))
        patch_score_source = "gradcam" if gradcam_success else "fallback"
        
        raw_v = model.vision_encoder(image)
        # Fallback: if GradCAM score is missing/invalid, still compute IE with a deterministic proxy score.
        # This avoids hard-zero visual_ie due to one failed attribution call.
        if (
            patch_scores is None
            or (not torch.is_tensor(patch_scores))
            or patch_scores.numel() == 0
            or (not torch.isfinite(patch_scores).all())
        ):
            patch_scores = raw_v[:, 1:, :].detach().abs().mean(dim=-1)
            gradcam_success = False
            patch_score_source = "fallback_l1"

        IE_g, IE_l = 0.0, 0.0
        # IE_g: mask top ratio
        logits_g = _run_masked_forward(raw_v, patch_scores, mask_ratio=mask_ratio_global)
        prob_g = _target_prob_from_logits(logits_g, idx)
        IE_g = max(orig_prob - prob_g, 0.0)
        if not np.isfinite(IE_g):
            IE_g = 0.0
        
        if use_local_ie:
            # IE_l: mask top-k patches (多 1 次 forward，epoch 慢时可关闭)
            logits_l = _run_masked_forward(raw_v, patch_scores, topk=topk_local)
            prob_l = _target_prob_from_logits(logits_l, idx)
            IE_l_raw = max(orig_prob - prob_l, 0.0)
            IE_l = max(IE_l_raw - tau, 0.0)  # stable gating
            if not np.isfinite(IE_l):
                IE_l = 0.0
            # alpha: GT category，abnormality -> 2.0, else 0.5；local_ie_alpha_scale 提高局部权重
            alpha_base = 2.0 if (answer_group or "").strip().lower() == "abnormality" else 0.5
            alpha = local_ie_alpha_scale * alpha_base
            visual_ie = (IE_g + alpha * IE_l) / (1.0 + alpha)
        else:
            prob_l = prob_g
            visual_ie = IE_g
            IE_l = 0.0  # 未启用 local
        
        visual_ie_patches = None
        if patch_scores is not None and visual_ie > 1e-6:
            patch_scores_norm = F.softmax(patch_scores.squeeze(0), dim=0)
            visual_ie_patches = visual_ie * patch_scores_norm
        
        # 确保为 Python float，避免 numpy 标量导致 nan 传播
        IE_g_out = float(IE_g) if np.isfinite(IE_g) else 0.0
        IE_l_out = float(IE_l) if np.isfinite(IE_l) else 0.0
        visual_ie_out = float(visual_ie) if np.isfinite(visual_ie) else 0.0
             
        return {
            "visual_ie": visual_ie_out,
            "visual_ie_g": IE_g_out,
            "visual_ie_l": IE_l_out,
            "visual_ie_patches": visual_ie_patches,
            "patch_scores": patch_scores.squeeze(0) if patch_scores is not None else None,
            "gradcam_success": gradcam_success,
            "patch_score_source": patch_score_source,
            "orig_prob": orig_prob,
            "masked_prob": prob_g,
            "masked_prob_local": prob_l if patch_scores is not None else None
        }

# ------------------------ 4. 高级接口 (HCSS & CCS) ------------------------

def _normalize_interventions(interventions) -> Tuple[List[str], List[str], List[str]]:
    """interventions: List[Tuple[str,str]] or List[str]. Returns (all_texts, for_de, for_ie)."""
    try:
        from pipeline.realtime_intervention_generator import INTV_STRONG, INTV_MEDIUM
    except Exception:
        INTV_STRONG, INTV_MEDIUM = "strong", "medium"
    if not interventions:
        return [], [], []
    first = interventions[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        tagged = [(t, l) for t, l in interventions if t]
        all_texts = [t for t, _ in tagged]
        for_de = [t for t, l in tagged if l in (INTV_STRONG, INTV_MEDIUM)]
        for_ie = [t for t, l in tagged if l == INTV_STRONG]
        if not for_ie and for_de:
            for_ie = for_de
        return all_texts, for_de or all_texts, for_ie or all_texts
    texts = [t for t in interventions if t]
    return texts, texts, texts


def compute_hcss_realtime(
    model, image, question, interventions, tokenizer, answer_idx, logits_orig_target,
    hcss_computer, answer_group=None, answer_idx_in_type=None, device='cuda',
    min_interventions=2, max_interventions=8,
    sign_adj_margin: float = 0.05,
    hcss_ie_scale: float = 1.0,
    sign_adj_temp: float = 0.05,
    hcss_norm_tau: float = 0.01,
    hcss_floor: float = 0.02,
):
    """分层干预：DE用strong+medium，IE用strong。弱干预不参与IE。

    Args:
        hcss_ie_scale: 对核心项 IE_mean·(1−DE_mean) 的乘性放大（与 CLI 一致）。
    """
    _, for_de, for_ie = _normalize_interventions(interventions)
    ie_diffs, de_logits, de_probs, ie_probs = [], [], [], []
    ie_de_indices = []

    tgt_idx = answer_idx_in_type if answer_idx_in_type is not None else answer_idx
    if logits_orig_target.dim() == 2:
        orig_target_logit = logits_orig_target[:, tgt_idx].squeeze()
    else:
        orig_target_logit = logits_orig_target.squeeze()

    for_de_set = set(for_ie)
    for idx, q_star in enumerate(for_de):
        if len(de_probs) >= max_interventions:
            break
        try:
            de_res = compute_text_direct_effect(
                model, image, question, q_star, tokenizer, answer_idx,
                logits_orig_target, answer_group, answer_idx_in_type, device
            )
            de_logits.append(de_res["logit"].view(-1)[0])
            de_probs.append(de_res["prob"])
            if q_star in for_de_set:
                ie_res = compute_text_indirect_effect(
                    model, image, question, q_star, tokenizer, answer_idx,
                    logits_orig_target, answer_group, answer_idx_in_type, device
                )
                ie_diffs.append(ie_res["diff"].view(-1)[0])
                ie_probs.append(ie_res["prob"])
                ie_de_indices.append(len(de_probs) - 1)
        except Exception:
            continue

    # 至少 1 个 DE 成功即可计算
    if len(de_probs) < 1:
        return {"text_hcss": torch.zeros(32, device=device), "hcss_scalar": 0.0, "hcss_core": 0.0, "sign_adj": 0.0, "de_ie_var": 0.0, "ie_mean": 0.0, "de_mean": 0.0}

    K = len(de_probs)
    de_logits_tensor = torch.stack(de_logits).unsqueeze(0)   # [1, K]
    orig_target_logit = orig_target_logit.view(1, 1)         # [1, 1]

    probs_orig = F.softmax(logits_orig_target, dim=-1)      # [1, C]
    C = probs_orig.shape[-1]
    tgt_clamped = min(tgt_idx, C - 1) if C > 0 else 0
    orig_p = probs_orig[0, tgt_clamped].item()

    # DE: 所有 strong+medium；IE: 仅 strong
    prob_de_per_k = torch.tensor([abs((de_probs[k][tgt_clamped] if de_probs[k].dim() > 0 else de_probs[k]).item() - orig_p) for k in range(K)], device=device, dtype=torch.float32)
    DE_mean = prob_de_per_k.mean().item()

    if ie_probs and ie_de_indices:
        prob_ie_list = [abs((ie_probs[j][tgt_clamped] if ie_probs[j].dim() > 0 else ie_probs[j]).item() - orig_p) for j in range(len(ie_probs))]
        IE_mean = float(np.mean(prob_ie_list))
        # 仅 IE 的稳定性方差（DE 不再进入 HCSS / token 路径，只保留作 loss 惩罚统计）
        tce_per_k = torch.tensor(prob_ie_list, device=device, dtype=torch.float32)
        Var_i = tce_per_k.var().item() if len(ie_probs) > 1 else 0.0
    else:
        IE_mean = 0.0
        Var_i = 0.0

    # ΔH': 稳健归一化熵差
    probs_primes = torch.stack([de_probs[k].to(device) for k in range(K)], dim=0).unsqueeze(0)  # [1, K, C]
    H_orig = -(probs_orig.clamp(1e-12) * torch.log(probs_orig.clamp(1e-12))).sum(dim=-1)  # [1]
    Hp = -(probs_primes.clamp(1e-12) * torch.log(probs_primes.clamp(1e-12))).sum(dim=-1)  # [1, K]
    DeltaH = (Hp - H_orig.unsqueeze(1)).squeeze(0)  # [K]
    med = DeltaH.median()
    mad = (DeltaH - med).abs().median().clamp(min=1e-9)
    DeltaH_prime = ((DeltaH - med) / (mad + 1e-9)).mean().item()

    # 公式 B: HCSS = SignAdj × (αDE + μIE + βΔH' - θVar)
    # SignAdj: 0/1 二值，加 margin 避免 early training collapse
    # d_mean >= -ε: 保留（允许轻微负值，稳定 early training）
    # d_mean < -ε: 关闭
    d_i = (orig_target_logit - de_logits_tensor).squeeze(0)  # [K]，仅用于返回/诊断，不进入 HCSS core
    d_mean = d_i.mean().item()

    # SignAdj：仅由 IE 的 logit 差方向决定（DE 不参与因果前向门控）
    if ie_diffs:
        d_ie = torch.stack(ie_diffs).mean().item()
        # Soft SignAdj avoids hard 0/1 collapse around margin.
        temp = max(float(sign_adj_temp), 1e-6)
        sign_adj_soft = float(torch.sigmoid(torch.tensor((d_ie + sign_adj_margin) / temp)).item())
    else:
        d_ie = 0.0
        sign_adj_soft = 1.0

    # 新版核心项：HCSS = α·TDE + μ·TIE（保留 sign_adj 方向门）
    # - α 固定 1.0（TDE 基础贡献）
    # - μ 复用 hcss_ie_scale（可用 CLI 放大 TIE 权重）
    alpha_de = 1.0
    mu_ie = max(float(hcss_ie_scale), 1e-6)
    hcss_core = alpha_de * float(DE_mean) + mu_ie * float(IE_mean)
    # Rescale tiny probability-space effects to a usable [0,1) band.
    # x -> x/(x+tau) preserves order and prevents near-zero collapse.
    tau = max(float(hcss_norm_tau), 1e-8)
    hcss_core_scaled = hcss_core / (hcss_core + tau) if hcss_core > 0 else 0.0
    hcss_gate = sign_adj_soft * max(0.0, min(1.0, hcss_core_scaled))
    # Keep a tiny floor when there is valid causal evidence, so offline priors do not degenerate to all zeros.
    evidence = float(DE_mean) + float(IE_mean)
    if evidence > 1e-8:
        overall_hcss = max(float(hcss_floor), hcss_gate)
    else:
        overall_hcss = 0.0

    de_mean_val = DE_mean
    ie_mean_val = IE_mean
    de_ie_var = Var_i
    sign_consistency = float(torch.sign(d_i.mean()).item()) if abs(d_i.mean().item()) > 1e-8 else 0.0

    # Build a token-level proxy HCSS:
    # - content tokens > special/pad tokens
    # - scale by sample-level HCSS
    seq_len = 32
    text_hcss = torch.full((seq_len,), overall_hcss, device=device, dtype=torch.float32)
    try:
        enc = tokenizer(
            question, return_tensors='pt', padding='max_length',
            truncation=True, max_length=seq_len
        )
        input_ids = enc['input_ids'][0].to(device)
        attn_mask = enc['attention_mask'][0].to(device).float()
        special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()
        special_mask = torch.zeros_like(attn_mask)
        for sid in special_ids:
            special_mask = torch.logical_or(special_mask.bool(), input_ids == sid)
        special_mask = special_mask.float()
        content_weight = 1.0 - 0.5 * special_mask  # special tokens get lower confidence
        text_hcss = text_hcss * attn_mask * content_weight
    except Exception:
        pass

    return {
        "text_hcss": text_hcss,
        "hcss_scalar": overall_hcss,
        "hcss_core": hcss_core,
        "sign_adj": sign_adj_soft,
        "d_ie_mean": float(d_ie),
        "de_mean": de_mean_val,
        "ie_mean": ie_mean_val,
        "sign_consistency": sign_consistency,
        "de_ie_var": de_ie_var,
    }

def compute_ccs_realtime(
    model, image, question, interventions, tokenizer, answer_idx, logits_orig_target,
    ccs_computer, answer_group=None, answer_idx_in_type=None,
    mask_ratio_global: float = 0.5, topk_local: int = 5, tau: float = 0.01,
    local_ie_alpha_scale: float = 1.0,
    use_local_ie: bool = True,
    mask_mode: str = "zero", device: str = "cuda",
    max_interventions: int = 5, precomputed_text_de: Optional[float] = None,
):
    """实时计算 CCS。Visual IE 来自 compute_visual_indirect_effect（概率尺度 ~[0,1]）。
    样本级与 patch 级 compute_CCS 第二路为文本 DE（参数名在 CCSComputer 内仍为 text_de 张量）。"""
    # 1. Visual IE (IE_g + alpha*IE_l, normalized)
    vis_res = compute_visual_indirect_effect(
        model, image, question, tokenizer, answer_idx,
        answer_group=answer_group, answer_idx_in_type=answer_idx_in_type,
        mask_ratio_global=mask_ratio_global, topk_local=topk_local, tau=tau,
        local_ie_alpha_scale=local_ie_alpha_scale,
        use_local_ie=use_local_ie, mask_mode=mask_mode, device=device
    )
    
    # 2. Text DE：参与 CCS 前向（与 visual_ie 组成样本级一致性）；同时写入返回 dict 供日志/惩罚
    if precomputed_text_de is not None:
        text_de = float(precomputed_text_de)
    else:
        _, for_de, _ = _normalize_interventions(interventions)
        interventions_for_de = for_de[:max_interventions]

        def _target_prob_from_logits(logits: torch.Tensor, idx: int) -> float:
            probs = F.softmax(logits, dim=-1)
            if probs.dim() == 2:
                return probs[0, idx].item()
            if probs.dim() == 1:
                return probs[idx].item()
            return probs.reshape(-1, probs.shape[-1])[0, idx].item()

        de_diffs = []
        for q_star in interventions_for_de:
            try:
                res = compute_text_direct_effect(
                    model, image, question, q_star, tokenizer, answer_idx,
                    logits_orig_target, answer_group, answer_idx_in_type, device
                )
                tgt_idx = answer_idx_in_type if answer_idx_in_type is not None else answer_idx
                orig_p = _target_prob_from_logits(logits_orig_target, tgt_idx)
                de_p = res["prob"][tgt_idx].item()
                de_diffs.append(abs(de_p - orig_p))
            except: pass
        text_de = float(np.mean(de_diffs)) if de_diffs else 0.0

    # 3. 样本级 CCS：visual_ie vs text_de（DE）
    ccs_scalar_res = ccs_computer.compute_CCS(
        torch.tensor([vis_res['visual_ie']], device=device),
        torch.tensor([text_de], device=device)
    )
    ccs_scalar = ccs_scalar_res['CCS'].item()

    # 4. Compute patch-level CCS only for mask attribution.
    visual_ie_patches = vis_res.get("visual_ie_patches")
    patch_scores = vis_res.get("patch_scores")
    
    if visual_ie_patches is not None and patch_scores is not None:
        # Patch level is used for selecting visual patches, not gate sign.
        ccs_res = ccs_computer.compute_CCS_patch_level(
             visual_ie_patches=visual_ie_patches.unsqueeze(0),
             text_de=torch.tensor([text_de], device=device).unsqueeze(0),
             patch_scores=patch_scores.unsqueeze(0)
        )
        vg = vis_res.get("visual_ie_g", 0.0)
        vl = vis_res.get("visual_ie_l", 0.0)
        vg = 0.0 if not np.isfinite(float(vg)) else float(vg)
        vl = 0.0 if not np.isfinite(float(vl)) else float(vl)
        return {
            "ccs": ccs_scalar,
            "ccs_patches": ccs_res['CCS_patches'].squeeze(),
            "visual_ie": vis_res['visual_ie'],
            "visual_ie_g": vg,
            "visual_ie_l": vl,
            "text_de": text_de
        }
    else:
        # No patch attribution available; return scalar gate signal only.
        vg = vis_res.get("visual_ie_g", 0.0)
        vl = vis_res.get("visual_ie_l", 0.0)
        vg = 0.0 if not np.isfinite(float(vg)) else float(vg)
        vl = 0.0 if not np.isfinite(float(vl)) else float(vl)
        return {
            "ccs": ccs_scalar,
            "visual_ie": vis_res['visual_ie'],
            "visual_ie_g": vg,
            "visual_ie_l": vl,
            "text_de": text_de
        }

def compute_text_de_from_interventions(model, image, question, interventions, tokenizer, answer_idx, logits_orig_target, answer_group, answer_idx_in_type, device, min_interventions=2):
    """Helper for CCS usage"""
    def _target_prob_from_logits(logits: torch.Tensor, idx: int) -> float:
        probs = F.softmax(logits, dim=-1)
        if probs.dim() == 2:
            return probs[0, idx].item()
        return probs[idx].item()

    de_diffs = []
    for q_star in interventions:
        try:
            res = compute_text_direct_effect(
                model, image, question, q_star, tokenizer, answer_idx,
                logits_orig_target, answer_group, answer_idx_in_type, device
            )
            idx = answer_idx_in_type if answer_idx_in_type is not None else answer_idx
            
            orig_p = _target_prob_from_logits(logits_orig_target, idx)
            
            de_p = res["prob"][idx].item()
            de_diffs.append(abs(de_p - orig_p))
        except: pass
    
    if len(de_diffs) < min_interventions: return 0.0
    return float(np.mean(de_diffs))
