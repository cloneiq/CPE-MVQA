# -*- coding: utf-8 -*-
"""
因果训练 schedule: λ_hcss, λ_ccs, mask_ratio 等按 epoch 调度
"""
from typing import Optional


def get_hcss_lam(epoch: int) -> float:
    """HCSS 文本因果权重: Epoch<=5 为 0.02, 6–15 提升至 0.04, 16+ 略降至 0.032 (配合 causal_start_epoch=3)"""
    if epoch <= 5:
        return 0.02
    if epoch <= 15:
        return 0.04
    return 0.032  # 后期略降防过拟合


def get_ccs_lam(epoch: int) -> float:
    """CCS 视觉因果权重: 保持低权重 0.0015，避免视觉主导 (Epoch 3+ 启用，causal_start_epoch=3)"""
    if epoch <= 2:
        return 0.0  # Epoch 1-2 无 causal，不生效
    return 0.0015


def get_mask_ratio(epoch: int) -> float:
    """mask_ratio 动态增加: Epoch 3-5:0.15, 6-10:0.20, 11+:0.25"""
    if epoch <= 5:
        return 0.15
    if epoch <= 10:
        return 0.20
    return 0.25


def get_bias_weight(sign_adj_ratio: float, base_bias_weight: float = 0.5) -> float:
    """Bias 分支控制: sign_adj_ratio>0.8 时 bias_weight*=0.7, 目标 0.6~0.75"""
    w = base_bias_weight
    if sign_adj_ratio > 0.8:
        w *= 0.7
    return w


def should_apply_causal_dropout(epoch: int, causal_dropout_start: int = 20) -> bool:
    """epoch>=20 时启用 causal_dropout"""
    return epoch >= causal_dropout_start
