from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------
# 工具函数
# --------------------------
def median_mad(x: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算中位数和绝对中位差（MAD）"""
    if dim is None:
        median = x.median()
        mad = (x - median).abs().median()
    else:
        median = x.median(dim=dim).values
        mad = (x - median.unsqueeze(dim)).abs().median(dim=dim).values
    mad = mad.clamp(min=eps)  # 避免除以零
    return median, mad


def compute_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """计算概率分布的熵"""
    p = probs.clamp(min=eps)  # 避免log(0)
    H = -(p * torch.log(p)).sum(dim=-1)
    return H


def bootstrap_ci(values: torch.Tensor, func=torch.mean, B: int = 200, ci: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """通过bootstrap方法计算置信区间"""
    values = values.detach().cpu()
    N = values.shape[0]
    est = func(values, dim=0) if callable(func) else values.mean(dim=0)
    stats = []
    for _ in range(B):
        idx = torch.randint(0, N, (N,))  # 有放回采样
        sample = values[idx]
        stats.append(func(sample, dim=0))
    stats = torch.stack(stats)
    lower = stats.quantile((1 - ci) / 2, dim=0)
    upper = stats.quantile((1 + ci) / 2, dim=0)
    return est, (lower, upper)


def estimate_de_from_logits(logit_orig: torch.Tensor, logit_primes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从logits估计DE（分布偏移）的均值、方差和原始差异"""
    if logit_orig.dim() == 1:
        logit_orig = logit_orig.unsqueeze(-1)
    if logit_primes.dim() == 2 and logit_orig.dim() == 2:
        logit_primes = logit_primes.unsqueeze(-1)
    if logit_primes.dim() == 3 and logit_orig.dim() == 2:
        diffs = logit_orig.unsqueeze(1) - logit_primes  # 原始差异（每个干预的DE值）
    else:
        raise ValueError(f"不匹配的形状: logit_orig {logit_orig.shape}, logit_primes {logit_primes.shape}")
    de_mean = diffs.mean(dim=1)  # DE均值（沿干预样本维度求平均）
    de_var = diffs.var(dim=1, unbiased=False)  # DE方差（沿干预样本维度）
    return de_mean, de_var, diffs


def estimate_de_from_probs_kl(probs_orig: torch.Tensor, probs_primes: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用 KL(P_orig || P_intervened) 估计 DE（分布差异）。
    即使 argmax 不变，只要置信度变化，DE 就不为 0，使 HCSS 更敏感。
    probs_orig: [B, C] 或 [1, C]
    probs_primes: [B, K, C] 或 [1, K, C]，K 为干预数
    """
    if probs_orig.dim() == 2 and probs_primes.dim() == 3:
        B, K, C = probs_primes.shape
        p_orig = probs_orig.clamp(min=eps)
        p_int = probs_primes.clamp(min=eps)
        # KL(P_orig || P_int) = sum(P_orig * (log(P_orig) - log(P_int)))
        log_p_orig = torch.log(p_orig + eps)
        log_p_int = torch.log(p_int + eps)
        kl_per_k = (p_orig.unsqueeze(1) * (log_p_orig.unsqueeze(1) - log_p_int)).sum(dim=-1)  # [B, K]
        diffs = kl_per_k.unsqueeze(-1)  # [B, K, 1] 与 estimate_de_from_logits 的 diffs 形状一致
    else:
        raise ValueError(f"不匹配的形状: probs_orig {probs_orig.shape}, probs_primes {probs_primes.shape}")
    de_mean = diffs.mean(dim=1)
    de_var = diffs.var(dim=1, unbiased=False)
    return de_mean, de_var, diffs


def estimate_ie_from_logits(logit_orig: torch.Tensor, logit_masked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从logits估计IE（干预效果）的均值、方差和原始差异（复用DE的估计逻辑）"""
    return estimate_de_from_logits(logit_orig, logit_masked)


# --------------------------
# GateNetwork（用于输出可学习参数）
# --------------------------
class GateNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 5)  # 输出α, μ, β, θ, γ五个参数
        )
        self.out_scale = out_scale
        # 初始化偏置为0.5，使初始参数接近1（通过softplus后）
        with torch.no_grad():
            self.net[-1].bias.fill_(0.5)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """输入特征，输出五个可学习参数（经softplus激活）"""
        is_3d = (feats.dim() == 3)
        if is_3d:
            B, Lu, D = feats.shape
            x = feats.reshape(B * Lu, D)
        else:
            x = feats
        out = self.net(x)
        out = F.softplus(out) * self.out_scale  # 确保输出为正
        if is_3d:
            out = out.view(B, Lu, 5)
        return out


# --------------------------
# HCSSComputer（核心计算类，修复DE_IE_var）
# --------------------------
class HCSSComputer:
    def __init__(self, gate: Optional[GateNetwork] = None, eps: float = 1e-6):
        self.gate = gate  # 门控网络（用于生成可学习参数）
        self.eps = eps  # 数值稳定性参数

    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        """将logits转换为概率分布"""
        return F.softmax(logits, dim=-1)

    def compute_deltaH_prime(self, probs: torch.Tensor, probs_primes: torch.Tensor) -> torch.Tensor:
        """
        计算稳健归一化的熵差 ΔH'。
        H(p) = -Σ p_k log(p_k), H(p_i*) = -Σ p_k* log(p_k*)
        ΔH_i = H(p_i*) - H(p)，ΔH>0 表示干预后不确定性上升
        ΔH' = (ΔH - median(ΔH)) / (MAD(ΔH) + ε)
        """
        H = compute_entropy(probs)  # [B] 原始分布的熵
        B, K, C = probs_primes.shape
        Hp = compute_entropy(probs_primes.view(-1, C)).view(B, K)  # [B, K] 每个干预的熵
        DeltaH = Hp - H.unsqueeze(1)  # [B, K] 每个干预的熵差
        med, mad = median_mad(DeltaH, dim=1)  # 沿干预维度求 median/MAD
        DeltaH_prime = (DeltaH - med.unsqueeze(1)) / (mad.unsqueeze(1) + self.eps)
        return DeltaH_prime

    def compute_signadj(self, logit_orig: torch.Tensor, logit_prime: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        方向性校正项 SignAdj。
        d_i = logit(P_y) - logit(P_y*), SignAdj_i = 1 + γ_i * tanh(d_i)
        """
        d = logit_orig - logit_prime
        return 1.0 + gamma * torch.tanh(d)

    def compute_hcss(self,
                     logit_orig_target: torch.Tensor,
                     logit_primes_target: torch.Tensor,
                     probs_orig: Optional[torch.Tensor] = None,
                     probs_primes: Optional[torch.Tensor] = None,
                     probs_ie_primes: Optional[torch.Tensor] = None,
                     unit_feats: Optional[torch.Tensor] = None,
                     pre_de: Optional[torch.Tensor] = None,
                     pre_de_var: Optional[torch.Tensor] = None,
                     pre_ie: Optional[torch.Tensor] = None,
                     pre_ie_var: Optional[torch.Tensor] = None,
                     pre_ie_diffs: Optional[torch.Tensor] = None,
                     use_bootstrap: bool = False,
                     ) -> Dict[str, torch.Tensor]:
        """
        层级因果稳定度分数 HCSS（单模态文本）：
        HCSS_i = α_i*DE_i + μ_i*IE_i + β_i*ΔH'_i - θ_i*Var_i*SignAdj_i
        - ΔH': 稳健归一化熵差（median/MAD）
        - Var_i: 多次干预下 TCE(DE+IE) 的方差（稳定性惩罚）
        - SignAdj_i: 方向性校正 1+γ*tanh(logit_orig - logit_prime)
        - α,μ,β,θ,γ: 由 GateNetwork 输出（无 gate 时用 1）
        """
        # DE：KL(P_orig || P_de) 或 logit 差
        if probs_orig is not None and probs_primes is not None:
            DE_mean, DE_var, diffs = estimate_de_from_probs_kl(probs_orig, probs_primes)
        else:
            DE_mean, DE_var, diffs = estimate_de_from_logits(logit_orig_target, logit_primes_target)
        
        # IE：优先用 KL(P_orig || P_ie) 与 DE 同尺度；否则用 pre_ie（logit 差）
        if probs_orig is not None and probs_ie_primes is not None:
            IE_mean, IE_var, ie_diffs = estimate_de_from_probs_kl(probs_orig, probs_ie_primes)
        elif pre_ie is not None:
            IE_mean = pre_ie
            ie_diffs = pre_ie_diffs if pre_ie_diffs is not None else diffs.clone()
            IE_var = pre_ie_var if pre_ie_var is not None else ie_diffs.var(dim=1, unbiased=False)
            if ie_diffs.dim() == 2:
                ie_diffs = ie_diffs.unsqueeze(-1)
        else:
            IE_mean, IE_var, ie_diffs = DE_mean.clone(), DE_var.clone(), diffs.clone()

        # Var_i: 同一特征多次干预下 TCE(DE+IE) 的方差（稳定性惩罚）
        sum_de_ie = diffs + ie_diffs
        if sum_de_ie.dim() == 3:
            sum_de_ie = sum_de_ie.squeeze(-1)
        sum_de_ie_per_intervention = sum_de_ie
        Var_i = sum_de_ie.var(dim=1, unbiased=False)  # [B]

        # ΔH': 稳健归一化熵差
        if probs_orig is not None and probs_primes is not None:
            DeltaH_prime = self.compute_deltaH_prime(probs_orig, probs_primes)  # [B, K]
            DeltaH_prime_mean = DeltaH_prime.mean(dim=1)  # [B] 样本级
        else:
            DeltaH_prime = torch.zeros(logit_orig_target.size(0), logit_primes_target.size(1),
                                       device=logit_orig_target.device)
            DeltaH_prime_mean = DeltaH_prime.mean(dim=1)

        # SignAdj: d_i = logit(P_y) - logit(P_y*), SignAdj_i = 1 + γ_i * tanh(d_i)
        logit_orig_exp = logit_orig_target.unsqueeze(1)  # [B, 1]
        gamma = 0.5 * torch.ones_like(DE_mean, device=logit_orig_target.device)
        if self.gate is not None and unit_feats is not None:
            gate_out = self.gate(unit_feats)
            alpha_i, mu_i, beta_i, theta_i, gamma = gate_out[..., 0], gate_out[..., 1], gate_out[..., 2], gate_out[..., 3], gate_out[..., 4]
        else:
            alpha_i = mu_i = beta_i = theta_i = torch.ones_like(DE_mean, device=logit_orig_target.device)
        SignAdj = self.compute_signadj(logit_orig_exp, logit_primes_target, gamma.unsqueeze(1))  # [B, K]
        SignAdj_mean = SignAdj.mean(dim=1)  # [B]

        # ===== [重构] 极简版 HCSS (去除熵与方差) =====
        DE_term = 1.0 * torch.abs(DE_mean)
        IE_term = 1.0 * torch.abs(IE_mean)
        HCSS = DE_term + IE_term
        DeltaH_term = torch.zeros_like(DE_term)
        Var_term = torch.zeros_like(DE_term)
        # 哑变量，防止下方 return 报错
        if "diffs" in locals():
            DeltaH_prime = torch.zeros_like(diffs)
            SignAdj = torch.ones_like(diffs)
        else:
            DeltaH_prime = torch.zeros_like(DE_mean)
            SignAdj = torch.ones_like(DE_mean)
        # =========================================

        # 若启用bootstrap，计算HCSS的置信区间（与主路径同尺度：KL）
        if use_bootstrap:
            K = logit_primes_target.shape[1]
            Bbatch = logit_orig_target.shape[0]
            boot_stats = []
            for b in range(Bbatch):
                per_k_vals = []
                for k in range(K):
                    if probs_orig is not None and probs_primes is not None and probs_ie_primes is not None:
                        _, _, de_d = estimate_de_from_probs_kl(probs_orig[b:b+1], probs_primes[b:b+1, k:k+1])
                        _, _, ie_d = estimate_de_from_probs_kl(probs_orig[b:b+1], probs_ie_primes[b:b+1, k:k+1])
                        hcss_k = (de_d.abs() + ie_d.abs()).squeeze()
                    else:
                        de_k = (logit_orig_target[b] - logit_primes_target[b, k])
                        ie_k = de_k.clone()
                        hcss_k = torch.abs(de_k) + torch.abs(ie_k)
                    per_k_vals.append(hcss_k.unsqueeze(0))

                # 对当前样本的所有扰动结果做bootstrap
                per_k_vals = torch.cat(per_k_vals, dim=0)
                est, (l, u) = bootstrap_ci(per_k_vals, func=lambda x, dim=0: x.mean(dim=0), B=200, ci=0.95)
                boot_stats.append((est, l, u))

            # 汇总bootstrap结果
            ests = torch.stack([s[0] for s in boot_stats], dim=0)
            lowers = torch.stack([s[1] for s in boot_stats], dim=0)
            uppers = torch.stack([s[2] for s in boot_stats], dim=0)
            results = {
                'DE_mean': DE_mean, 'DE_var': DE_var,
                'IE_mean': IE_mean, 'IE_var': IE_var,
                'DeltaH_prime': DeltaH_prime, 'SignAdj': SignAdj,
                'DE_IE_sum': sum_de_ie_per_intervention.mean(dim=1),
                'DE_IE_var': Var_i,
                'HCSS': HCSS,
                'HCSS_boot_mean': ests, 'HCSS_boot_lower': lowers, 'HCSS_boot_upper': uppers,
                'diffs': diffs
            }
        else:
            results = {
                'DE_mean': DE_mean, 'DE_var': DE_var,
                'IE_mean': IE_mean, 'IE_var': IE_var,
                'DeltaH_prime': DeltaH_prime, 'SignAdj': SignAdj,
                'DE_IE_sum': sum_de_ie_per_intervention.mean(dim=1),
                'DE_IE_var': Var_i,
                'HCSS': HCSS,
                'diffs': diffs
            }
        return results


# --------------------------
# CCSComputer（跨模态贡献计算）
# --------------------------
class CCSComputer:
    """
    核心逻辑：CCS = 视觉IE contribution - 文本DE contribution
    兼容新版 VQA 模型 (Logits 输出)
    text_de_scale: 文本DE量纲补偿，因 vis_ie 通常比 text_de 大 10~20 倍
    """

    def __init__(self, eps: float = 1e-8, text_de_scale: float = 1.0):
        self.eps = eps  # 避免除零
        self.text_de_scale = text_de_scale  # 平衡 vis_ie 与 text_de 的典型量纲差异

    @staticmethod
    def compute_effect_from_logits(logits_orig: torch.Tensor, logits_intervened: torch.Tensor, 
                                   mode: str = "prob_diff", target_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从两组 logits 计算因果效应 (IE 或 DE)
        
        Args:
            logits_orig: 原始 Logits [B, NumClasses]
            logits_intervened: 干预后 Logits [B, NumClasses]
            mode: 计算模式 
                  - "prob_diff": 概率差 (Prob_orig - Prob_int)
                  - "logit_diff": Logit 差 (Logit_orig - Logit_int)
                  - "kl_div": KL 散度
            target_idx: 目标类别索引 [B]. 如果为 None，计算这整个分布的差异或最大差异。
            
        Returns:
            effect: [B] 效应值
        """
        if mode == "prob_diff":
            probs_orig = F.softmax(logits_orig, dim=-1)
            probs_int = F.softmax(logits_intervened, dim=-1)
            if target_idx is not None:
                # 取目标类别的概率差
                p_orig = probs_orig.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                p_int = probs_int.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                effect = (p_orig - p_int).abs() # 通常取绝对变化量作为“效应”大小
            else:
                # 如果没有指定目标，取 L1 距离平均或其他度量
                effect = (probs_orig - probs_int).abs().mean(dim=-1)
                
        elif mode == "logit_diff":
            if target_idx is not None:
                l_orig = logits_orig.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                l_int = logits_intervened.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                effect = (l_orig - l_int).abs()
            else:
                effect = (logits_orig - logits_intervened).abs().mean(dim=-1)
                
        elif mode == "kl_div":
             log_probs_orig = F.log_softmax(logits_orig, dim=-1)
             probs_int = F.softmax(logits_intervened, dim=-1)
             # KL(Int || Orig) measuring surprise? Or KL(Orig || Int)?
             # Usually effect is distance. 
             effect = F.kl_div(log_probs_orig, probs_int, reduction='none').sum(dim=-1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return effect

    def aggregate_modal_effect(self, effect: torch.Tensor) -> torch.Tensor:
        """聚合模态效应（默认直接展平）"""
        return effect.squeeze()

    def compute_CCS(self, visual_ie: torch.Tensor, text_de: torch.Tensor, normalize_text_de: bool = True) -> Dict[str, torch.Tensor]:
        """
        计算CCS指标及各模态贡献比例
        
        Args:
            visual_ie: 视觉间接效应（概率尺度，0-1）
            text_de: 文本直接效应（logit尺度，可能很大）
            normalize_text_de: 是否将text_de标准化到0-1范围（解决量纲不一致问题）
        
        Returns:
            CCS相关指标字典
        """
        # 确保效应非负（贡献不能为负）
        visual_ie = torch.clamp(visual_ie, min=0.0)
        text_de_raw = torch.clamp(text_de, min=0.0)
        
        # 量纲补偿：vis_ie 典型 0.05~0.2，text_de 典型 0.002~0.02，用 scale 平衡
        text_de_raw = text_de_raw * self.text_de_scale
        
        # 🔧 修复量纲不一致问题：将text_de标准化到0-1范围
        if normalize_text_de:
            # 智能判断：如果text_de已经在0-1范围内（可能是已经归一化的），直接使用
            # 否则，假设是logit尺度，使用sigmoid映射到(0,1)
            text_de_max = text_de_raw.max().item() if text_de_raw.numel() > 0 else 1.0
            if text_de_max <= 1.0:
                # 已经在0-1范围内，可能是已经归一化的，直接使用
                text_de = text_de_raw
            else:
                # 稳定压缩：避免标量场景下 max-normalize 后几乎恒等于 sigmoid(10) 导致 text_de≈1
                # f(x)=x/(1+x) 单调、范围(0,1)、对大值不过度饱和
                text_de = text_de_raw / (1.0 + text_de_raw)
        else:
            text_de = text_de_raw
        
        # 总效应（避免除零）
        total_effect = visual_ie + text_de + self.eps
        
        # 计算贡献比例与CCS
        visual_contribution = visual_ie / total_effect
        text_contribution = text_de / total_effect
        ccs_raw = visual_contribution - text_contribution
        # CCS 稳定化: tanh(0.7*x) 压缩到 [-0.7, 0.7]，避免极化 std↑
        CCS = torch.tanh(0.7 * ccs_raw)

        return {
            'visual_ie': visual_ie,
            'text_de': text_de,
            'total_effect': total_effect,
            'visual_contribution': visual_contribution,
            'text_contribution': text_contribution,
            'CCS': CCS
        }

    def compute_CCS_patch_level(
        self,
        visual_ie_patches: torch.Tensor,  # [batch, num_patches] 或 [num_patches]
        text_de: torch.Tensor,  # [batch] 或标量
        patch_scores: Optional[torch.Tensor] = None,  # [batch, num_patches] - patch重要性分数
        normalize_text_de: bool = True,
        aggregation: str = "topk"  # "mean", "topk", "max", "weighted"
    ) -> Dict[str, torch.Tensor]:
        """
        计算patch级别的CCS（关键修复：支持MedVQA微小病灶检测）
        
        Args:
            visual_ie_patches: 每个patch的visual_ie [batch, num_patches] 或 [num_patches]
            text_de: 文本直接效应 [batch] 或标量
            patch_scores: patch重要性分数（Grad-CAM分数）[batch, num_patches]，用于加权聚合
            normalize_text_de: 是否将text_de标准化到0-1范围
            aggregation: 聚合方式
                - "mean": 所有patch平均（可能被背景稀释）
                - "topk": 高贡献patch平均（推荐）
                - "max": 最高贡献patch（最激进）
                - "weighted": 基于patch_scores加权（最合理）
        
        Returns:
            {
                'CCS_patches': [batch, num_patches],  # 每个patch的CCS
                'CCS': [batch] 或标量,  # 样本级别的CCS（聚合后）
                'CCS_mean': [batch],  # 所有patch平均
                'CCS_topk': [batch],  # 高贡献patch平均
                'CCS_max': [batch],  # 最高贡献patch
                'visual_ie_patches': [batch, num_patches],
                'text_de': [batch, num_patches]
            }
        """
        # 确保维度正确
        if visual_ie_patches.dim() == 1:
            visual_ie_patches = visual_ie_patches.unsqueeze(0)  # [1, num_patches]
        
        batch_size, num_patches = visual_ie_patches.shape
        
        if text_de.dim() == 0:
            text_de = text_de.unsqueeze(0).unsqueeze(0)  # [1, 1]
        elif text_de.dim() == 1:
            if text_de.shape[0] == batch_size:
                text_de = text_de.unsqueeze(1)  # [batch, 1]
            else:
                text_de = text_de.unsqueeze(0)  # [1, 1]
        
        # 扩展text_de到patch维度，并应用量纲补偿
        text_de_expanded = text_de.expand(-1, num_patches) * self.text_de_scale  # [batch, num_patches]
        
        # 标准化text_de（如果需要）
        if normalize_text_de:
            text_de_max = text_de_expanded.max().item()
            if text_de_max > 1.0:
                scale_factor = 1.0 / (text_de_max + self.eps)
                text_de_expanded = torch.sigmoid(text_de_expanded * scale_factor * 10)
        
        # 确保效应非负
        visual_ie_patches = torch.clamp(visual_ie_patches, min=0.0)
        text_de_expanded = torch.clamp(text_de_expanded, min=0.0)
        
        # 计算每个patch的CCS
        total_effect = visual_ie_patches + text_de_expanded + self.eps
        visual_contribution = visual_ie_patches / total_effect
        text_contribution = text_de_expanded / total_effect
        ccs_raw = visual_contribution - text_contribution
        ccs_patches = torch.tanh(0.7 * ccs_raw)  # 稳定化: [-0.7, 0.7]
        
        # 样本级别的CCS聚合
        # 方法1：简单平均（可能被背景patch稀释）
        ccs_mean = ccs_patches.mean(dim=1)  # [batch]
        
        # 方法2：使用高visual_ie的patches（病灶候选区）
        # 只考虑visual_ie > 阈值的patches（避免背景稀释）
        visual_ie_threshold = visual_ie_patches.mean(dim=1, keepdim=True)  # [batch, 1]
        high_ie_mask = visual_ie_patches > visual_ie_threshold  # [batch, num_patches]
        if high_ie_mask.any():
            ccs_topk = (ccs_patches * high_ie_mask.float()).sum(dim=1) / high_ie_mask.float().sum(dim=1).clamp(min=1)  # [batch]
        else:
            ccs_topk = ccs_mean
        
        # 方法3：最大patch的CCS（病灶最可能的位置）
        ccs_max = ccs_patches.max(dim=1)[0]  # [batch]
        
        # 方法4：基于patch_scores加权（最合理，如果提供了patch_scores）
        if patch_scores is not None:
            if patch_scores.dim() == 1:
                patch_scores = patch_scores.unsqueeze(0)  # [1, num_patches]
            
            # 归一化patch_scores作为权重
            patch_weights = F.softmax(patch_scores, dim=1)  # [batch, num_patches]
            ccs_weighted = (ccs_patches * patch_weights).sum(dim=1)  # [batch]
        else:
            ccs_weighted = ccs_mean
        
        # 根据聚合方式选择最终的CCS
        if aggregation == "mean":
            ccs_final = ccs_mean
        elif aggregation == "topk":
            ccs_final = ccs_topk
        elif aggregation == "max":
            ccs_final = ccs_max
        elif aggregation == "weighted":
            ccs_final = ccs_weighted
        else:
            ccs_final = ccs_topk  # 默认使用topk
        
        # 如果是单样本，返回标量（向后兼容）
        if batch_size == 1:
            ccs_final_scalar = ccs_final[0]
        else:
            ccs_final_scalar = ccs_final
        
        return {
            'CCS_patches': ccs_patches,  # [batch, num_patches]
            'CCS': ccs_final_scalar,  # [batch] 或标量（向后兼容）
            'CCS_mean': ccs_mean,  # [batch]
            'CCS_topk': ccs_topk,  # [batch]
            'CCS_max': ccs_max,  # [batch]
            'CCS_weighted': ccs_weighted if patch_scores is not None else ccs_mean,  # [batch]
            'visual_ie_patches': visual_ie_patches,  # [batch, num_patches]
            'text_de': text_de_expanded,  # [batch, num_patches]
            'visual_contribution_patches': visual_contribution,  # [batch, num_patches]
            'text_contribution_patches': text_contribution  # [batch, num_patches]
        }

    def bootstrap_ccs(self, visual_ie_list: torch.Tensor, text_de_list: torch.Tensor, Bboot: int = 200, ci: float = 0.95) -> Dict[str, Any]:
        """通过bootstrap计算CCS的置信区间"""
        N = visual_ie_list.shape[0]
        if N != text_de_list.shape[0]:
            raise ValueError("视觉IE和文本DE的样本数必须一致")
        
        # 基础结果
        base_result = self.compute_CCS(visual_ie_list, text_de_list)
        base_ccs = base_result['CCS']
        
        # Bootstrap采样
        ccs_samples = []
        for _ in range(Bboot):
            idx = torch.randint(0, N, (N,))  # 有放回采样
            sampled_visual_ie = visual_ie_list[idx]
            sampled_text_de = text_de_list[idx]
            sampled_result = self.compute_CCS(sampled_visual_ie, sampled_text_de)
            ccs_samples.append(sampled_result['CCS'].mean().item())
        
        # 计算置信区间
        ccs_samples = np.array(ccs_samples)
        lower = np.quantile(ccs_samples, (1 - ci) / 2)
        upper = np.quantile(ccs_samples, (1 + ci) / 2)

        return {
            'base_ccs_mean': base_ccs.mean().item(),
            'base_visual_contribution_mean': base_result['visual_contribution'].mean().item(),
            'base_text_contribution_mean': base_result['text_contribution'].mean().item(),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'per_sample_ccs': base_ccs.cpu().numpy()
        }


# --------------------------
# 工具函数（应用HCSS结果）
# --------------------------
def apply_hcss_to_attention_logits(attn_logits: torch.Tensor, hcss: torch.Tensor, mode: str = 'add', scale: float = 1.0) -> torch.Tensor:
    """将HCSS结果应用于注意力logits（调整注意力权重）"""
    out = attn_logits.clone()
    if hcss.dim() == 2:
        B, Lq = hcss.shape
        if out.dim() == 4:  # 多头注意力形状: [B, num_heads, Lq, Lk]
            if hcss.shape[1] == out.shape[2]:  # HCSS长度匹配查询序列长度
                bias = hcss.unsqueeze(1).unsqueeze(-1)
                bias = bias.expand(-1, out.shape[1], -1, out.shape[3])
            elif hcss.shape[1] == out.shape[3]:  # HCSS长度匹配键序列长度
                bias = hcss.unsqueeze(1).unsqueeze(2)
                bias = bias.expand(-1, out.shape[1], out.shape[2], -1)
            else:
                raise ValueError("HCSS长度与注意力维度不匹配")
        else:  # 非多头注意力形状: [B, Lq, Lk]
            if hcss.shape[1] == out.shape[1]:
                bias = hcss.unsqueeze(-1)
            elif hcss.shape[1] == out.shape[2]:
                bias = hcss.unsqueeze(1)
            else:
                raise ValueError("HCSS长度与注意力维度不匹配")
    else:
        raise ValueError("HCSS必须是2D张量 [B, L]")

    # 应用HCSS调整（加性或乘性）
    if mode == 'add':
        out = out + scale * bias
    elif mode == 'scale':
        out = out * (1.0 + scale * bias)
    else:
        raise ValueError("模式必须是 'add' 或 'scale'")

    return out


def soft_prune_embeddings(embeddings: torch.Tensor, hcss: torch.Tensor, method: str = 'dropout', scale: float = 1.0) -> torch.Tensor:
    """基于HCSS对嵌入进行软修剪（保留重要特征）"""
    hcss_scaled = hcss * scale  # 缩放HCSS权重
    if method == 'dropout':
        keep_prob = torch.sigmoid(hcss_scaled)  # 保留概率（HCSS越大保留概率越高）
        return embeddings * keep_prob.unsqueeze(-1)
    elif method == 'threshold':
        thr = hcss_scaled.median(dim=1, keepdim=True).values  # 以中位数为阈值
        mask = (hcss_scaled >= thr).unsqueeze(-1).float()  # 高于阈值的保留
        return embeddings * mask
    else:
        raise ValueError("未知方法，支持 'dropout' 或 'threshold'")


if __name__ == "__main__":
    # 测试HCSS计算（验证DE_IE_var非零）
    torch.manual_seed(42)
    B, L, C, K = 2, 10, 5, 3  # 批次、序列长度、类别数、干预样本数
    logit_orig = torch.randn(B, L, C)  # 原始logits [2,10,5]
    logit_primes = torch.randn(B, K, L, C)  # 3个干预样本的logits [2,3,10,5]
    probs_orig = F.softmax(logit_orig, dim=-1)
    probs_primes = F.softmax(logit_primes, dim=-1)
    unit_feats = torch.randn(B, L, 32)  # 特征 [2,10,32]

    # 初始化门控网络和HCSS计算器
    gate = GateNetwork(in_dim=32)
    hcss_comp = HCSSComputer(gate=gate)
    
    # 计算IE结果（获取ie_diffs）
    ie_res = hcss_comp.compute_hcss(
        logit_orig_target=logit_orig,
        logit_primes_target=logit_primes,
        probs_orig=probs_orig,
        probs_primes=probs_primes,
        unit_feats=unit_feats,
        use_bootstrap=False
    )
    
    # 计算最终HCSS（传递pre_ie_diffs）
    final_res = hcss_comp.compute_hcss(
        logit_orig_target=logit_orig,
        logit_primes_target=logit_primes,
        probs_orig=probs_orig,
        probs_primes=probs_primes,
        unit_feats=unit_feats,
        pre_ie=ie_res['IE_mean'],
        pre_ie_var=ie_res['IE_var'],
        pre_ie_diffs=ie_res['diffs'],  # 传递IE的每个干预差异
        use_bootstrap=False
    )
    
    print("HCSS测试结果（修复后）：")
    print(f"DE_IE_var形状: {final_res['DE_IE_var'].shape}")  # 应输出 [2,10,5]
    print(f"DE_IE_var是否非零: {not torch.allclose(final_res['DE_IE_var'], torch.zeros_like(final_res['DE_IE_var']))}")  # 应输出True