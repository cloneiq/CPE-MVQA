import matplotlib
matplotlib.use("Agg")
import os
import warnings
# Suppress transformers FutureWarning about deprecated device argument (v5)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
# Fix libgomp "Invalid value for OMP_NUM_THREADS" (must set before importing torch/numpy)
v = os.environ.get('OMP_NUM_THREADS', '')
if not v or not str(v).strip().isdigit() or int(str(v).strip()) <= 0:
    os.environ['OMP_NUM_THREADS'] = '1'

import json
import torch
import itertools
import numpy as np
import argparse
import logging
import torch.nn.functional as F
import torchvision
import torch.nn as nn

torchvision.disable_beta_transforms_warning()
from tqdm import tqdm
import pickle

import inspect
from train import train_epoch, validate
from build_causal_cache import rebuild_signal_cache, partial_update_signal_cache
from utils.dataloader import VQADataLoader
from utils.pseudo_mask import PseudoOrganMaskGenerator
from models.vqa_module import CausalVQAModel
from models.m3ae import M3AE
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from transformers import get_cosine_schedule_with_warmup
import math
from torch.optim.lr_scheduler import LambdaLR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VQADataLoader')

# Environment variable settings
for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if env_var in os.environ:
        os.environ.pop(env_var)

# Configure logging (include logger name for train module output)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable blocking for CUDA operation


def _anneal_params_continuous(visual_ie: float, ccs_var: float,
                              base_alpha: float = 2.0, k1: float = 8.0, T1: float = 0.22,
                              base_inv: float = 0.01) -> tuple:
    """
    连续退火：gate 强度、invariance 强度 = f(visual_ie, CCS分布)
    - alpha = base_alpha * sigmoid(k1*(visual_ie - T1))：vi 高 → gate 强
    - inv = base_inv * (1 - var(CCS))：exp 衰减太快，改用线性
    - 滞回：ccs_var>0.25 降 inv，ccs_var<0.05 升 inv，防抖动
    """
    import math
    sig = 1.0 / (1.0 + math.exp(-k1 * (visual_ie - T1)))
    gate_alpha = base_alpha * sig
    ccs_var_cap = min(max(ccs_var, 0.0), 1.0)  # [0,1] 防负值
    inv_lam = base_inv * (1.0 - ccs_var_cap)
    # 滞回：防抖动、自适应
    if ccs_var > 0.25:
        inv_lam *= 0.5
    elif ccs_var < 0.05:
        inv_lam *= 1.2
    return float(gate_alpha), float(inv_lam)


def parse_args():
    parser = argparse.ArgumentParser(description='training')

    # Data-related parameters
    parser.add_argument('--data_dir', type=str, default='data_med', help='Root directory of data')
    parser.add_argument('--image_dir', type=str, default='data_med/images', help='Image directory')
    parser.add_argument('--strong_augment', type=int, default=0, help='1=RandomResizedCrop, RandomRotation±15°, ColorJitter, RandomHorizontalFlip (小batch推荐)')
    parser.add_argument('--train_json', type=str, default='data_med/train_typed.jsonl', help='Training data JSON')
    parser.add_argument('--val_json', type=str, default='data_med/val_typed.jsonl', help='Validation data JSON')
    parser.add_argument('--test_json', type=str, default='data_med/test_typed.jsonl', help='Test data JSON')

    # Model-related parameters
    parser.add_argument('--vocab', type=str, default='roberta', help='Vocabulary')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--max_length', type=int, default=32, help='Maximum sequence length')
   
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension')
    parser.add_argument('--num_top_layer', type=int, default=6, help='attention layer')
    parser.add_argument('--input_image_embed_size', type=int, default=768, help='Visual feature dimension')
    parser.add_argument('--input_text_embed_size', type=int, default=768, help='Question feature dimension')
    parser.add_argument('--finetune', type=bool, default=False, help='Whether to finetune the pretrained model')
    parser.add_argument('--load_path', type=str, default='pretrained_weights/m3ae.ckpt',
                        help='Pretrained weights path')
    parser.add_argument('--roberta_path', type=str, default='pretrain/roberta-base',
                        help='Local path to roberta-base model directory')
    parser.add_argument('--embeddings_dir', type=str, default='data_med/embeddings_all', help='Embeddings directory')

    # Training-related parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--val_num_workers', type=int, default=0, help='Val/test workers (0=单进程防OOM)')
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Number of warmup epochs (推荐2)')
    parser.add_argument('--warmup_ratio', type=float, default=-1, help='Warmup比例(0~1, >0时覆盖warmup_epochs: warmup_epochs=int(epochs*ratio))')
    parser.add_argument('--warmup_type', type=str, default='sigmoid', help='Warmup type')
    parser.add_argument('--use_causal_schedule', type=int, default=1, help='1=使用λ_hcss/λ_ccs/mask_ratio按epoch调度; 0=固定值')
    parser.add_argument('--use_light_causal_weights', type=int, default=0, help='1=轻量损失权重(λ_hcss=λ_ccs=0.003)，主要靠特征门控; 0=使用schedule或固定值')
    parser.add_argument('--causal_lam', type=float, default=0.02, help='λ_hcss 基准(use_causal_schedule=0时使用)')
    parser.add_argument('--hcss_alpha', type=float, default=1.0, help='HCSS公式 DE项权重 α (预留)')
    parser.add_argument('--hcss_mu', type=float, default=1.0, help='HCSS公式 IE项权重 μ (预留)')
    parser.add_argument('--hcss_stage1_epochs', type=int, default=3, help='Stage1 duration (use_causal_schedule=0时生效)')
    parser.add_argument('--ccs_lam', type=float, default=0.015, help='λ_ccs 基准(use_causal_schedule=0时使用)')
    parser.add_argument('--ccs_alpha', type=float, default=4.0, help='v_mask=sigmoid(alpha*CCS), CCS=-0.5->0.1, 0->0.5, 0.5->0.9')
    parser.add_argument('--hcss_topk_ratio', type=float, default=0.30, help='Top causal tokens=30%%, token ranking后guided mask')
    parser.add_argument('--v_causal_topk_ratio', type=float, default=0.4, help='Top-k ratio for visual CCS: keep top k%% patch groups as causal (0.3~0.5)')
    parser.add_argument('--lr', type=float, default=5e-6, help='Base LR (visual/text encoder)')
    parser.add_argument('--lr_classifier', type=float, default=1e-5, help='Classifier (vqa_head) LR')
    parser.add_argument('--lr_bias', type=float, default=2.5e-6, help='Bias/cross-modal branch LR ')
    parser.add_argument('--min_lr', type=float, default=5e-7, help='Cosine decay min LR')
    parser.add_argument('--lr_decay_epoch', type=int, default=0, help='One-time LR decay epoch (0=disabled, 用cosine)')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='LR multiplier after lr_decay_epoch')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=3, help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping epochs')
    parser.add_argument('--val_collapse_threshold', type=float, default=0.0, help='Stop immediately if val drops below best - threshold (0=disabled, 5=stop if drop>=5%%)')
    parser.add_argument('--label_smoothing', type=float, default=0.2, help='Label smoothing (0~0.2, e.g. 0.1 to reduce overfitting)')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='Dropout in vqa_head (0~0.3, e.g. 0.1 to reduce overfitting)')
    parser.add_argument('--backbone_dropout', type=float, default=0.0, help='Dropout on backbone outputs (vision/text after proj, 0.1~0.15 to reduce overfitting)')
    parser.add_argument('--merge_val_train', action='store_true', help='Merge val data into train for final training')
    parser.add_argument('--eval_test_freq', type=int, default=0,
                        help='merge 且无 val 时: 每 N 个 epoch 在 test 上跑一遍 **仅写日志/历史**，不用于选 best、不触发 early stop；0=不测 test')
    parser.add_argument('--seed', type=int, default=105, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=5, help='Logging interval')
    parser.add_argument('--log_diagnostic_interval', type=int, default=5, help='分布/CEM/CCS诊断每N次log打印(5=每5次log打印1次)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (leave blank for auto selection)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'infer'], help='Run mode')
    parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path (for evaluation or inference)')
    parser.add_argument('--rebuild_vocab', action='store_true', help='Rebuild vocabulary')
    parser.add_argument('--save_dir', type=str, default='checkpoints-3', help='Model save directory')
   
    # Visualization parameters
    parser.add_argument('--visualize_every', type=int, default=5, help='Visualize every N epochs')
    parser.add_argument('--val_freq', type=int, default=2, help='Validation frequency (every N epochs, 2=每2个epoch验证一次)')
    parser.add_argument('--val_subset_ratio', type=float, default=1.0, help='训练时验证集采样比例(0.1=10%%快速监控, 1.0=100%%完整验证)')
    parser.add_argument('--val_batch_size', type=int, default=-1, help='验证 batch size (-1=与 train 相同, 512 等可加速验证)')
    
    # Causal switch
    parser.add_argument('--use_causal', type=int, default=1, help='Whether to enable causal reasoning (1: Enable, 0: Disable)')
    parser.add_argument(
        '--intervention_path',
        type=str,
        default='',
        help='Optional JSONL intervention bank; if empty/missing, interventions are built at cache time via rules + frozen model (no separate file required).',
    )
    parser.add_argument('--causal_start_epoch', type=int, default=1, help='Epoch to start causal reasoning (Stage1: E1-3 HCSS only, Stage2: E4+ HCSS+CCS)')
    parser.add_argument('--causal_batch_ratio', type=float, default=1.0, help='[Legacy/no-op] MoE/ top-k 已替代二次截断')
    parser.add_argument('--causal_max_interventions', type=int, default=3, help='Max interventions per sample for TRAIN (2-3 faster, 5 better quality)')
    parser.add_argument('--causal_max_interventions_val_test', type=int, default=2, help='Val/Test: fewer interventions to reduce inference cost ~50%% (train:4 val:2)')
    parser.add_argument('--causal_ratio', type=float, default=0.15,
                        help='[Legacy] use_moe_router=0 时作 router_topk_ratio 回退；MoE 下请优先用 --router_topk_ratio')
    parser.add_argument('--min_causal_ratio', type=float, default=0.08,
                        help='[Legacy] 当前 top-k 由 router_topk_ratio 直接定长，不参与 MoE')
    parser.add_argument('--use_moe_router', type=int, default=0,
                        help='1=融合特征 MLP 打分 top-k，稀疏 CEM+干预仅 top-k（推荐）')
    parser.add_argument('--router_topk_ratio', type=float, default=0.15,
                        help='每 batch 进入因果专家(干预+CEM) 的比例 0.1~0.2（与 MoE/随机 top-k 共用）')
    parser.add_argument('--lambda_router_entropy', type=float, default=0.002,
                        help='warmup 后：loss -= λ·H(sigmoid(score))；≤0.01；warmup 内关闭')
    parser.add_argument('--lambda_router_calib', type=float, default=0.03,
                        help='loss += λ·KL(Bern(p)||Bern(r_topk))；p=sigmoid(router score)；宜小')
    parser.add_argument('--moe_warmup_epochs', type=int, default=5,
                        help='MoE Stage1：仅 CE + router 校准，CEM scale=0、不传 CCS intra')
    parser.add_argument('--moe_post_warm_cem_scale', type=float, default=0.03,
                        help='Stage2 起始 CEM 混合（略低减轻压 text），随后在 moe_cem_ramp_epochs 内升到 1.0')
    parser.add_argument('--moe_cem_ramp_epochs', type=int, default=5,
                        help='Stage2 CEM 从 post_warm_cem_scale 爬升到 1.0 的 epoch 数')
    parser.add_argument('--causal_router_hidden', type=int, default=-1,
                        help='MoE MLP 隐层；-1=auto hidden_dim//4')
    parser.add_argument('--cem_align_schedule', type=int, default=0,
                        help='0=L_cem_align 全 epoch 常数权重；1=分阶段线性升温（弱化 schedule）')
    parser.add_argument('--cem_align_start_epoch', type=int, default=10)
    parser.add_argument('--cem_align_full_epoch', type=int, default=20)
    parser.add_argument('--causal_semantic_filter', type=float, default=0.45, help='>0=enable RoBERTa filter; 0.45=医学paraphrase宽松(0.6会100%%no-quality), 0=skip')
    parser.add_argument('--min_quality_interventions', type=int, default=1, help='Minimum interventions (1=宽松, 2=严格; 语义过滤开时建议1)')
    parser.add_argument('--min_entity_overlap', type=float, default=0.40, help='Minimum entity overlap for intervention filtering')
    parser.add_argument('--sim_low', type=float, default=0.55, help='Lower semantic similarity bound for interventions')
    parser.add_argument('--sim_high', type=float, default=0.90, help='Upper semantic similarity bound for interventions')
    parser.add_argument('--sim_low_strong', type=float, default=0.25, help='Relaxed sim for strong interventions (IE/CCS)')
    parser.add_argument('--overlap_min_strong', type=float, default=0.02, help='Min entity overlap for strong interventions')
    parser.add_argument('--ccs_negative_weight', type=float, default=0.0, help='CCS<0 sample loss weight: 0=ignore, 0.5=half, 1.0=no change')
    parser.add_argument('--ccs_negative_weight_min', type=float, default=0.3, help='Floor for CCS<0 weight to avoid double suppression (SignAdj=0+CCS<0)')
    parser.add_argument('--ccs_text_de_scale', type=float, default=1.0, help='Scale text_de (1=no scale; 15~20 if need balance)')
    parser.add_argument('--ccs_mask_ratio', type=float, default=0.22, help='mask_ratio_target (use_causal_schedule=0时固定; =1时按epoch调度)')
    parser.add_argument('--schedule_mask_ratio', type=float, default=-1, help='Override ccs_mask_ratio (>=0时生效)')
    parser.add_argument('--mask_schedule', type=str, default='step', choices=['step', 'cosine'], help='mask_ratio调度: step=阶梯, cosine=余弦')
    parser.add_argument('--ccs_topk_local', type=int, default=5, help='Local IE top-k patches (IE_l)')
    parser.add_argument('--ccs_tau', type=float, default=0.01, help='Local IE stable gating threshold')
    parser.add_argument('--local_ie_alpha', type=float, default=1.0, help='Local IE weight scale: alpha *= this (1.2~1.5 for more local)')
    parser.add_argument('--ccs_use_local_ie', type=int, default=1, help='1=enable Local IE (3 forwards), 0=IE_g only (2 forwards, faster)')
    parser.add_argument('--ccs_target', type=float, default=0.2, help='Target CCS value; samples with CCS far from target get reduced weight')
    parser.add_argument('--ccs_penalty_lambda', type=float, default=0.08, help='Penalty scale when CCS deviates from ccs_target; 0=no penalty')
    parser.add_argument('--sign_adj_margin', type=float, default=0.05, help='SignAdj margin')
    parser.add_argument('--sign_adj_temp', type=float, default=0.05, help='Soft SignAdj temperature (smaller=sharper gate)')
    parser.add_argument('--hcss_norm_tau', type=float, default=0.01, help='HCSS core rescale tau: x/(x+tau)')
    parser.add_argument('--hcss_floor', type=float, default=0.02, help='Minimum HCSS when IE/DE evidence exists')
    parser.add_argument('--offline_interaction_alpha_scale', type=float, default=0.3,
                        help='Offline causal residual max scale (recommended <=0.3)')
    parser.add_argument('--bias_weight', type=float, default=0.12, help='Bias分支权重(初期); 若设bias_weight_ramp_epoch则epoch>=时用bias_weight_late')
    parser.add_argument('--bias_weight_late', type=float, default=-1, help='Epoch>=bias_weight_ramp_epoch时的bias_weight (-1=不用分阶段)')
    parser.add_argument('--bias_weight_ramp_epoch', type=int, default=-1, help='从该epoch起使用bias_weight_late (-1=禁用分阶段)')
    parser.add_argument('--causal_dropout', type=float, default=0.0, help='>0 时按 epoch 随机关闭 q/v mask（论文主线建议 0；legacy 实验可开）')
    parser.add_argument('--causal_dropout_start_epoch', type=int, default=20, help='causal_dropout>0 时起始 epoch')
    parser.add_argument('--vis_boost_lambda', type=float, default=0.02, help='L_vis=-λ*vis_ie 视觉causal boost，强制用视觉')
    parser.add_argument('--text_de_penalty_lambda', type=float, default=0.002, help='loss+=λ·mean(text_de)；弱惩罚防 shortcut，默认低于 0.004')
    parser.add_argument('--visual_ie_auto_anneal', type=int, default=0, help='1=auto-adjust ccs/invariance by visual_ie (LOW=0.10, HIGH=0.22); 0=固定 train_config（论文主线默认 0）')
    parser.add_argument('--use_feature_gate', type=int, default=1, help='1=gate主导 gated_feats=feats*gate (推荐); 0=hard mask')
    parser.add_argument('--use_causal_gate_in_val', type=int, default=0,
                        help='0=验证不做实时干预（不跑 compute_causal_masks，与「仅权重」评测一致；推荐）；'
                             '1=验证也走 HCSS+CCS+干预（Train=Val 结构，耗时长、与关闭干预的 test 可能不一致）')
    parser.add_argument('--gate_alpha', type=float, default=1.0, help='Feature gate: alpha for ccs_patches, 初期1.0')
    parser.add_argument('--gate_beta', type=float, default=0.8, help='Feature gate: beta for hcss_patches, 初期0.8')
    parser.add_argument('--causal_mask_causal_parts', type=int, default=1, help='If 1, mask top CCS patches (causal parts) to detect change; 0=keep causal parts, mask rest (old)')
    parser.add_argument('--abn_loss_weight', type=float, default=2.0, help='Loss weight for abnormality (推荐2.0, 不超过2.5)')
    parser.add_argument('--abn_loss_boost', type=float, default=1.0, help='Multiplier for abn_loss_weight (1.0=稳定推荐)')
    parser.add_argument('--organ_loss_weight', type=float, default=1.2, help='Loss weight for organ (推荐1.2)')
    parser.add_argument('--abn_hcss_target', type=float, default=0.18, help='abnormal HCSS软区间中心，0.12~0.24')
    parser.add_argument('--abn_hcss_margin', type=float, default=0.06, help='abnormal HCSS软区间半宽')
    parser.add_argument('--abn_ratio_target', type=float, default=1.1, help='abnormal 视觉主导 R=CCS/HCSS>=1.1')
    parser.add_argument('--abn_lam', type=float, default=0.03, help='λ_abn: abnormal HCSS软区间+ratio软惩罚')
    parser.add_argument('--abn_oversample_ratio', type=float, default=1.0, help='Oversample abnormality in train (3.0=3x more often per epoch, key for 70%%)')
    parser.add_argument('--open_oversample_ratio', type=float, default=1.0, help='Oversample OPEN questions (VQA-RAD/SLAKE: 2.0~3.0 when Open acc is low)')
    parser.add_argument('--open_loss_weight', type=float, default=2.5, help='Loss weight for OPEN questions (VQA-RAD: 2.0 when Open acc is 0)')
    parser.add_argument('--use_open_embedding_matching', type=int, default=0, help='1=Open 用 embedding matching 替代 BCE，解决 OOV/同义词')
    parser.add_argument('--answer_embeddings_path', type=str, default='', help='answer_embeddings.pt 路径，默认 data_dir/answer_embeddings.pt')
    parser.add_argument('--open_embedding_tau', type=float, default=0.07, help='Embedding matching 温度 (0.05~0.1)')
    parser.add_argument('--open_embedding_loss_weight', type=float, default=1.5,
                        help='Open CE loss weight when use_open_embedding_matching (CE梯度强，建议1.0~1.5，低于open_loss_weight)')
    parser.add_argument('--open_embedding_topk_soft', type=int, default=5,
                        help='Top-k soft target: 同义词 lung≈pulmonary 共享概率 (1=硬标签, 5=推荐)')
    parser.add_argument('--open_embedding_soft_temp', type=float, default=0.07,
                        help='Soft target 温度 (0.05~0.1)，按语义距离加权，避免 lung≈brain 权相同')
    parser.add_argument('--open_embedding_align_lam', type=float, default=0.2,
                        help='Alignment loss: pred_emb 与 gt_emb 对齐，λ=0.1~0.3')
    parser.add_argument('--open_embedding_hybrid_weight', type=float, default=0.5,
                        help='Open 混合损失: (1-w)*BCE + w*CE(open_logits)，0.5=各半')
    parser.add_argument('--use_ce_for_answer', type=int, default=1,
                        help='1=主答案用加权 CE(answer_indices)（与 VQA-Med 单 canonical 索引一致）；0=对 targets 做 BCE 多标签；validate 的 L_ans 与此对齐')
    parser.add_argument('--use_vqa_rad_concept', type=int, default=0,
                        help='1=VQA-RAD concept: Open用concept, Closed用question-aware concept(因果先验proxy DE), misc目标15-25%%')
    parser.add_argument('--use_slake_concept', type=int, default=0,
                        help='1=SLAKE concept: 四层语义单元, 仅Open局部refine, 与use_vqa_rad_concept互斥')
    parser.add_argument('--stage2', type=int, default=0,
                        help='1=Stage2 Open Concept：加载 Stage1 ckpt，L_ans+λ_concept·L_concept+可选 L_open_refine+sym-KL')
    parser.add_argument('--use_concept_head', type=int, default=0,
                        help='1=启用 concept head（concept 主线实验）；0=纯 VQA 主线')
    parser.add_argument('--lambda_concept', type=float, default=0.0,
                        help='L_concept 权重；默认 0；concept 实验时再设 0.2~0.5')
    parser.add_argument('--repr_gate_alpha', type=float, default=0.2,
                        help='表示层视觉: F_img *= (1 + α·patch_soft)，VQA-RAD 可 0.2，MEDVQA 可 0.05')
    parser.add_argument('--repr_gate_beta', type=float, default=0.2,
                        help='表示层文本: F_txt *= (1 + β·token_soft)')
    parser.add_argument('--use_offline_causal', type=int, default=1,
                        help='1=使用离线因果先验（cache）；；MoE 由 --use_moe_router 控制')
    parser.add_argument('--causal_cache_path', type=str, default='',
                        help='离线因果先验 JSON（兼容旧版：未单独指定 train 时，train/val/test 共用此文件）')
    parser.add_argument('--causal_cache_path_train', type=str, default='',
                        help='训练集专用 cache（如 data_RAD/train_cache.json）。若设置则与 test 分离；val 默认同目录 val_cache.json')
    parser.add_argument('--causal_cache_path_val', type=str, default='',
                        help='验证集专用 cache；为空且已设 train 专用路径时默认为同目录 val_cache.json')
    parser.add_argument('--causal_cache_path_test', type=str, default='',
                        help='测试集专用 cache；test.py 应传此路径；为空且已设 train 专用路径时默认为同目录 test_cache.json')
    parser.add_argument('--causal_bias_suppress_beta', type=float, default=0.3,
                        help='BiasSuppressor: T\'=T*(1-β*bias_mask)')
    parser.add_argument('--causal_hcss_attn_scale', type=float, default=2.0,
                        help='CausalTextEncoder: HCSS attention bias scale')
    parser.add_argument('--lambda_causal_cons', type=float, default=0.01,
                        help='L_cons = KL(F_s || F_c) 权重')
    parser.add_argument('--lambda_causal_bias', type=float, default=0.01,
                        help='L_bias = bias * CE(logits,y) 权重')
    parser.add_argument('--lambda_causal_pos', type=float, default=0.01,
                        help='L_pos: positive representation alignment weight')
    parser.add_argument('--lambda_causal_neg', type=float, default=0.01,
                        help='L_neg: negative debias cosine constraint weight')
    parser.add_argument('--causal_pos_tau', type=float, default=0.1,
                        help='Positive sample threshold on offline HCSS')
    parser.add_argument('--use_do_controller', type=int, default=1,
                        help='1=use representation-level DO controller as the only causal module; 0=disable DO control')
    parser.add_argument('--lambda_do_consistency', type=float, default=0.1,
                        help='DO consistency: masked KL(p_base||p_do) on ccs<0 samples')
    parser.add_argument('--lambda_do_pref', type=float, default=0.05,
                        help='Modality preference: -vis_ie * log p_do(y)')
    parser.add_argument('--do_loss_beta', type=float, default=0.2,
                        help='DO CE reweight strength: weight = 1 + beta * do_gate (0 disables)')
    parser.add_argument(
        '--causal_alpha',
        type=float,
        default=0.5,
        help='因果融合系数 α∈[0,1]：表征 F_mix=LN(α·F_do+(1-α)·F_clean)；post_fusion 时对 logits 同权混合',
    )
    parser.add_argument(
        '--dynamic_causal_alpha',
        type=int,
        default=1,
        help='1=每样本 α_i=α_base·(1−CCS_i) 再 clamp[0,1] 并乘 valid_mask；0=全局标量 causal_alpha',
    )
    parser.add_argument('--lambda_do_anchor', type=float, default=0.0,
                        help='可选：表征锚定 MSE(F_mix,F_clean) 在 CCS≥0 上；默认 0（融合由 --causal_alpha 显式控制）')
    parser.add_argument('--do_mode', type=str, default='hard_mask',
                        choices=['hard_mask'],
                        help='DO implementation mode (hard_mask: feature-level hard intervention).')
    parser.add_argument('--do_ccs_margin', type=float, default=0.1,
                        help='Hard DO trigger margin: apply DO when ccs < -margin')
    parser.add_argument(
        '--do_keep_base',
        type=float,
        default=0.85,
        help='DO soft-mask floor in compute_do_mask (try 0.4~0.55 for stronger text suppression when CCS<0; higher=closer to identity).',
    )
    parser.add_argument(
        '--do_intervention_point',
        type=str,
        default='pre_fusion',
        choices=['pre_fusion', 'post_fusion'],
        help='pre_fusion: do 在融合前（文本 q_mask_pre + 视觉 patch 缩放）；post_fusion: 融合后 CLS 乘性干预（旧式）',
    )
    parser.add_argument(
        '--lambda_offline_bank_align',
        type=float,
        default=0.01,
        help='MSE(F_clean, fusion_bank)：需 cache 含 fusion_bank（重建 cache 时 --cache_store_fusion_repr=1）',
    )
    parser.add_argument(
        '--bank_align_warmup_epochs',
        type=int,
        default=15,
        help='前 N 个 epoch 将 lambda_offline_bank_align 置 0，稳定后再打开字典对齐，减轻早期死锁/噪声',
    )
    parser.add_argument(
        '--cache_store_fusion_repr',
        type=int,
        default=1,
        help='1=重建/增量更新 cache 时写入 fusion_bank（冻结模型无 do 的融合向量）',
    )
    parser.add_argument(
        '--skip_causal_cache_rebuild',
        type=int,
        default=0,
        help='1=若 train/val/test 对应 json 已存在则跳过训练开始时的全量 rebuild（沿用已有 cache）；缺失的路径仍会重建',
    )
    parser.add_argument('--do_gate_tau', type=float, default=1.2,
                        help='Soft DO gate temperature on standardized CCS: sigmoid(-z_ccs/tau)')
    parser.add_argument('--do_gate_bias', type=float, default=0.2,
                        help='Soft DO gate bias on standardized CCS: sigmoid(-(z_ccs-bias)/tau)')
    parser.add_argument('--do_logit_tau', type=float, default=1.0,
                        help='Decision-space DO residual multiplier: logits=base+gate*(tau*delta)')
    parser.add_argument('--do_delta_scale', type=float, default=1.0,
                        help='Delta-logit norm constraint: delta=tanh(W_do(...))*scale')
    parser.add_argument('--do_residual_clamp_k', type=float, default=1.5,
                        help='Residual clamp: logits shift per class is clamped to [-k, k] after gate')
    parser.add_argument('--signal_update_interval', type=int, default=5,
                        help='Rebuild offline causal signal cache every K epochs (0=disable periodic rebuild)')
    parser.add_argument('--signal_update_ratio', type=float, default=0.25,
                        help='Partial refresh ratio for signal cache updates (e.g. 0.25)')
    parser.add_argument('--signal_ema_momentum', type=float, default=0.9,
                        help='EMA momentum used inside signal builder during cache rebuild')
    parser.add_argument('--fusion_dropout_prob', type=float, default=-1.0,
                        help='fusion(BertCrossLayer) dropout。<0 且 use_causal=1 时默认 0.05；≥0 强制该值')
    parser.add_argument('--causal_soft_alpha', type=float, default=-1.0,
                        help='若>=0 则覆盖 --repr_gate_alpha（表示层视觉软门）')
    parser.add_argument('--causal_soft_beta', type=float, default=-1.0,
                        help='若>=0 则覆盖 --repr_gate_beta')
    parser.add_argument('--use_logits_cem', type=int, default=1,
                        help='1=CEM 仅在 logits（默认）；0=参数兼容，行为同 1')
    parser.add_argument('--cem_gamma', type=float, default=1.0,
                        help='CEM 模态权重 softmax 温度 τ：>1 更均匀、<1 更尖锐；1.0=默认')
    parser.add_argument('--cem_direction_k', type=float, default=2.0,
                        help='CEM direction=tanh(k*(vi-ti)/(vi+ti)) 的 k，建议 1.5~2.5；越大对 IE 差分越敏感')
    parser.add_argument('--use_causal_aligned_cem', type=int, default=1,
                        help='1=对「CEM 后主 logits」与「logits_base」加 SmoothL1 对齐（推荐开）')
    parser.add_argument('--lambda_cem_align', type=float, default=0.002,
                        help='L_cem_align（需 use_causal_aligned_cem=1）；因果子集；与 L_feat_cons 共稳 IE')
    parser.add_argument('--lambda_cem_guard', type=float, default=0.005,
                        help='CEM KL guard: λ·KL(P_out||P_base)，约束最终 logits 勿过度偏离 logits_base；0=关闭')
    parser.add_argument('--lambda_cem_gate_align', type=float, default=0.003,
                        help='CEM gate 弱对齐 MSE；默认 0.003，避免强控 g_t')
    parser.add_argument('--lambda_ie_reg', type=float, default=0.001,
                        help='text_ie batch 方差正则，减轻 IE 漂移')
    parser.add_argument('--lambda_ie_floor', type=float, default=0.002,
                        help='text_ie 下界: loss+=λ·mean(relu(thr−text_ie))，IE<thr 惩罚')
    parser.add_argument('--ie_floor_threshold', type=float, default=0.08,
                        help='与 --lambda_ie_floor 配合；IE 低于该值惩罚，高于不罚')
    parser.add_argument('--lambda_ie_de_coupling', type=float, default=0.002,
                        help='IE-DE 方向项: λ·mean(text_ie·text_de.detach())；DE 不参与该项梯度，避免双双被压')
    parser.add_argument('--lambda_causal_feat_consistency', type=float, default=0.0025,
                        help='L_feat_cons：因果子集 fusion 表征对齐（0=关）')
    parser.add_argument('--use_feature_causal_probe', type=int, default=0,
                        help='1=仅在因果子集随机 probe：token 置 pad、图像随机擦除（强度见 probe_*）')
    parser.add_argument('--probe_text_token_ratio', type=float, default=0.0,
                        help='probe 开时：每个保留 token 以该概率换 pad（因果样本）')
    parser.add_argument('--probe_patch_zero_ratio', type=float, default=0.0,
                        help='probe 开时：因果样本以该概率触发一块随机矩形擦除')
    parser.add_argument('--lambda_open', type=float, default=0.0,
                        help='L_open_refine 权重 (0.10~0.15): Open 样本局部 patch 因果强化，小数据集建议 0.12')
    parser.add_argument('--open_refine_v_topk_ratio', type=float, default=0.2,
                        help='L_open_refine: Open 样本用更严格的 top-k patch 比例 (0.2=20%% 最因果 patch)')
    parser.add_argument('--concept_focus_refine_weight', type=float, default=1.2,
                        help='concept_focus_flag=1(open+non-misc) 时 L_open_refine 样本权重 (1.2~1.5)，普通 open 为 1.0')
    parser.add_argument('--allow_last_resort_interventions', type=int, default=0, help='If 1, use raw interventions when quality filter rejects all (default 0 to avoid unreliable text)')
    parser.add_argument('--relax_sim_low', type=float, default=0.50, help='Relaxed sim lower bound when quality filter fails (0.35=permissive, 0.50=stricter)')
    parser.add_argument('--teacher_ema_decay', type=float, default=0.999, help='EMA decay for Teacher (Mean Teacher): 0.99~0.999, Teacher computes HCSS/CCS to break feedback loop')
    parser.add_argument('--use_teacher', type=int, default=0, help='1=use Teacher-Student (EMA for HCSS/CCS), 0=single model (saves ~50%% GPU memory)')
    parser.add_argument('--difficulty_gate_threshold', type=float, default=0.0, help='困难样本门控: threshold>0 时仅对 cls_loss>threshold 的 batch 做因果 (0=关闭)')
    # Paper experiment: causal training starts from baseline checkpoint (e.g. 65% -> 70%)
    parser.add_argument('--resume_from_meve', type=str, default='',
                        help='Optional MEVE checkpoint loaded BEFORE --resume_from_baseline (for staged transfer).')
    parser.add_argument('--resume_from_baseline', type=str, default='',
                        help='When use_causal=1: load this baseline checkpoint before training, so causal starts from baseline (e.g. 65%%)')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint (alias: 会设置resume_from_baseline)')
    parser.add_argument('--ablation_no_hcss', action='store_true',
                        help='强消融: 结构上移除 HCSS，不参与 loss/gate/CEM/val/test')
    parser.add_argument('--ablation_no_ccs', action='store_true',
                        help='强消融: 结构上移除 CCS，不参与 loss/gate/CEM/val/test')
    parser.add_argument('--ablation_no_open_refine', action='store_true',
                        help='强消融: 结构上移除 Open 局部因果分支，不做 open local causal forward，Open 直走全局分支')
    parser.add_argument('--lambda_counterfactual', type=float, default=0.02,
                        help='反事实监督权重；过大易压主 CE、诱发不用 text，默认 0.02')
    parser.add_argument('--counterfactual_margin', type=float, default=0.1,
                        help='反事实 margin（概率空间，对应 sigmoid(logit_gt)）')
    parser.add_argument('--counterfactual_start_epoch', type=int, default=10,
                        help='≥该 epoch 才加反事实 loss（默认 10，避免过早 CF）')
    parser.add_argument('--cem_gate_align_start_epoch', type=int, default=10,
                        help='≥该 epoch 才加 λ_cem_gate_align（默认与 CF 同轮开启）')
    parser.add_argument('--ie_reg_start_epoch', type=int, default=10,
                        help='≥该 epoch 才加 λ_ie_reg')
    parser.add_argument('--hcss_ie_scale', type=float, default=1.5,
                        help='对 HCSS 核心 IE_mean·(1−DE_mean) 的乘性放大（典型 1.3~2.0）')
    parser.add_argument('--counterfactual_only_closed', type=int, default=1,
                        help='1=仅 closed 样本计入（排除 open）；0=全 batch')

    args = parser.parse_args()
    if getattr(args, 'resume', '') and not (args.resume_from_baseline or '').strip():
        args.resume_from_baseline = args.resume
    return args


def train(args, device):
    """Execute model training process"""
    import torch.nn as nn
    import torch.optim as optim

    if float(getattr(args, 'causal_soft_alpha', -1.0)) >= 0.0:
        args.repr_gate_alpha = float(args.causal_soft_alpha)
    if float(getattr(args, 'causal_soft_beta', -1.0)) >= 0.0:
        args.repr_gate_beta = float(args.causal_soft_beta)

    # use_slake_concept 自动联动: use_concept_head=1, use_vqa_rad_concept=0
    if getattr(args, 'use_slake_concept', 0):
        args.use_concept_head = 1
        args.use_vqa_rad_concept = 0

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    logger.info("===== Start training preparation =====")

    # Configure dataloader parameters
    data_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'embeddings_dir': args.embeddings_dir,
        'train_json': args.train_json,
        'val_json': args.val_json,
        'test_json': args.test_json,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'val_num_workers': args.val_num_workers,
        'image_size': args.image_size,
        'max_length': args.max_length,
        'tokenizer': args.vocab,
        'roberta_path': args.roberta_path,
        'rebuild_vocab': args.rebuild_vocab,
        'merge_val_train': args.merge_val_train,
        'device': str(device),
        'abn_oversample_ratio': args.abn_oversample_ratio,
        'open_oversample_ratio': getattr(args, 'open_oversample_ratio', 1.0),
        'skip_ae_maml': True,
        'val_subset_ratio': getattr(args, 'val_subset_ratio', 1.0),
        'val_batch_size': args.val_batch_size if args.val_batch_size > 0 else args.batch_size,
        'use_slake_concept': bool(getattr(args, 'use_slake_concept', 0)),
        'use_vqa_rad_concept': (bool(getattr(args, 'use_vqa_rad_concept', 0)) or bool(getattr(args, 'stage2', 0)) or bool(getattr(args, 'use_concept_head', 0))) and not getattr(args, 'use_slake_concept', 0),
        'strong_augment': bool(getattr(args, 'strong_augment', 0)),
    }

    # Initialize dataloader
    logger.info("Initializing dataloader...")
    data_loader = VQADataLoader(data_config)
    loaders = data_loader.get_loaders()
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')

    # Get number of answer classes
    answer_vocab = data_loader.get_answer_vocab()
    num_classes = answer_vocab['vocab_size']
    logger.info(f"Number of answer classes: {num_classes}")

    # Offline causal caches: train / val / test (split when --causal_cache_path_train is set).
    train_cache_path = (
        str(getattr(args, "causal_cache_path_train", "") or "").strip()
        or str(getattr(args, "causal_cache_path", "") or "").strip()
    )
    use_split_caches = bool(str(getattr(args, "causal_cache_path_train", "") or "").strip())
    _cache_base = os.path.dirname(train_cache_path) if train_cache_path else "."
    if use_split_caches:
        val_cache_path = (
            str(getattr(args, "causal_cache_path_val", "") or "").strip()
            or os.path.join(_cache_base or ".", "val_cache.json")
        )
        test_cache_path = (
            str(getattr(args, "causal_cache_path_test", "") or "").strip()
            or os.path.join(_cache_base or ".", "test_cache.json")
        )
    else:
        val_cache_path = train_cache_path
        test_cache_path = train_cache_path

    # Model configuration
    answer_emb_path = getattr(args, 'answer_embeddings_path', '').strip()
    if not answer_emb_path and getattr(args, 'use_open_embedding_matching', 0):
        answer_emb_path = os.path.join(args.data_dir, 'answer_embeddings.pt')
    _fdp_arg = float(getattr(args, 'fusion_dropout_prob', -1.0))
    if _fdp_arg >= 0.0:
        _fusion_dropout_prob = _fdp_arg
    elif bool(args.use_causal):
        _fusion_dropout_prob = 0.05
    else:
        _fusion_dropout_prob = None

    model_config = {
        'input_image_embed_size': args.input_image_embed_size,
        'input_text_embed_size': args.input_text_embed_size,
        'num_top_layer': args.num_top_layer,
        'hidden_size': args.hidden_dim,
        'num_hid': num_classes,
        'visual_backbone': 'ViT-B/16',
        'image_size': args.image_size,
        'load_path': args.load_path,
        'roberta_path': args.roberta_path,
        'patch_size': args.patch_size,
        'use_causal': bool(args.use_causal),
        'head_dropout': args.head_dropout,
        'backbone_dropout': getattr(args, 'backbone_dropout', 0.0),
        'fusion_dropout_prob': _fusion_dropout_prob,
        'use_open_embedding_matching': bool(getattr(args, 'use_open_embedding_matching', 0)),
        'answer_embeddings_path': answer_emb_path,
        'open_embedding_tau': getattr(args, 'open_embedding_tau', 0.07),
        'use_open_concept_head': bool(getattr(args, 'stage2', 0)) or bool(getattr(args, 'use_concept_head', 0)) or bool(getattr(args, 'use_slake_concept', 0)),
        'num_concepts': __import__('utils.slake_concept', fromlist=['NUM_CONCEPTS']).NUM_CONCEPTS if getattr(args, 'use_slake_concept', 0) else 5,
        'repr_gate_alpha': getattr(args, 'repr_gate_alpha', 0.2),
        'repr_gate_beta': getattr(args, 'repr_gate_beta', 0.2),
        'use_offline_causal': bool(int(getattr(args, 'use_offline_causal', 0))),
        'causal_bias_suppress_beta': float(getattr(args, 'causal_bias_suppress_beta', 0.3)),
        'causal_hcss_attn_scale': float(getattr(args, 'causal_hcss_attn_scale', 2.0)),
        'cem_gamma': float(getattr(args, 'cem_gamma', 1.0)),
        'cem_direction_k': float(getattr(args, 'cem_direction_k', 2.0)),
        'use_logits_cem': bool(int(getattr(args, 'use_logits_cem', 1))),
        'use_moe_router': bool(int(getattr(args, 'use_moe_router', 0))),
        'offline_interaction_alpha_scale': float(getattr(args, 'offline_interaction_alpha_scale', 0.3)),
        'causal_router_hidden': int(getattr(args, 'causal_router_hidden', -1)),
        'use_do_controller': bool(args.use_causal) and bool(int(getattr(args, 'use_do_controller', 1))),
        'do_mode': str(getattr(args, 'do_mode', 'hard_mask')),
        'do_ccs_margin': float(getattr(args, 'do_ccs_margin', 0.05)),
        'do_keep_base': float(getattr(args, 'do_keep_base', 0.55)),
        'do_intervention_point': str(getattr(args, 'do_intervention_point', 'pre_fusion')),
        'causal_alpha': float(max(0.0, min(1.0, getattr(args, 'causal_alpha', 0.5)))),
        'dynamic_causal_alpha': int(getattr(args, 'dynamic_causal_alpha', 1)),
        'do_gate_tau': float(getattr(args, 'do_gate_tau', 1.2)),
        'do_gate_bias': float(getattr(args, 'do_gate_bias', 0.2)),
        'do_logit_tau': float(getattr(args, 'do_logit_tau', 1.0)),
        'do_delta_scale': float(getattr(args, 'do_delta_scale', 1.0)),
        'do_residual_clamp_k': float(getattr(args, 'do_residual_clamp_k', 1.5)),
    }

    hcss_stage1 = getattr(args, 'hcss_stage1_epochs', 3)
    causal_start = args.causal_start_epoch
    stage1_end = causal_start + hcss_stage1 - 1
    logger.info("=" * 60)
    if getattr(args, 'use_concept_head', 0) and args.use_causal:
        no_open_ref = getattr(args, 'ablation_no_open_refine', False)
        if no_open_ref:
            logger.info("  TRAINING MODE: 统一框架（L_ans + λ_cem_align + λ_concept）[ablation: w/o L_open_refine]")
            logger.info(f"  - logits CEM 主路径 | L_concept（可选）| Open 直走全局（无局部 refine）")
        else:
            logger.info("  TRAINING MODE: 统一框架（L_ans + λ_cem_align + λ_concept + 可选 λ_open·L_open_refine）")
            logger.info(f"  - L_concept + L_open_refine（可选）")
        logger.info(
            f"  - λ_concept={getattr(args,'lambda_concept',0.0)} "
            f"λ_open={getattr(args,'lambda_open',0.0) if not no_open_ref else 0}")
    elif getattr(args, 'use_concept_head', 0) and not getattr(args, 'stage2', 0):
        logger.info("  TRAINING MODE: Baseline + Concept Head（L_ans + λ_concept*L_concept，无 causal）")
        logger.info(f"  - backbone + answer head + concept head 同时训练")
    elif getattr(args, 'stage2', 0):
        logger.info("  TRAINING MODE: Stage 2 Open Concept 强化")
        logger.info(f"  - L_total = L_ans + λ_concept*L_concept + 可选 L_open_refine（logits CEM 主路径）")
        logger.info(
            f"  - λ_concept={getattr(args,'lambda_concept',0.0)}")
        if not args.resume_from_baseline:
            logger.warning("  Stage 2 需要 --resume_from_baseline 加载 Stage 1 checkpoint!")
    elif args.use_causal:
        if model_config.get("use_do_controller", True):
            logger.info("  TRAINING MODE: CAUSAL (offline bank -> fusion 前/后 do + CE + 可选 MoE + 离线字典 MSE)")
            logger.info(
                "  - do 标量: m_T/m_V 由 CCS/HCSS/text_de/vis_ie 推出；默认 pre_fusion：q_mask_pre + 视觉 patch 缩放后进入融合"
            )
            logger.info(
                f"  - causal_alpha={float(getattr(args, 'causal_alpha', 0.5)):.3f}：F_blend=α·F_do+(1-α)·F_clean；"
                f"MoE 时在 F_blend 与专家塔间路由；L_bank 权重={float(getattr(args, 'lambda_offline_bank_align', 0.01))}"
            )
        else:
            logger.info("  TRAINING MODE: BASELINE (DO controller disabled)")
    else:
        logger.info("  TRAINING MODE: BASELINE (No Causal Intervention)")
        logger.info("  - Causal intervention modules: DISABLED")
        logger.info("  - Standard BCE training only")
    logger.info("=" * 60)

    train_config = {
        "warmup_epochs": args.warmup_epochs,
        "warmup_type": args.warmup_type,
        "use_causal": bool(args.use_causal),
        "label_smoothing": args.label_smoothing,
        "causal_start_epoch": args.causal_start_epoch,
        "causal_batch_ratio": args.causal_batch_ratio,
        "causal_max_interventions": args.causal_max_interventions,
        "causal_max_interventions_val_test": getattr(args, 'causal_max_interventions_val_test', 2),
        "causal_ratio": float(getattr(args, 'causal_ratio', 1.0)),
        "use_offline_causal": bool(int(getattr(args, 'use_offline_causal', 0))),
        "causal_cache_path": train_cache_path,
        "causal_cache_path_val": val_cache_path,
        "lambda_causal_cons": float(getattr(args, 'lambda_causal_cons', 0.01)),
        "lambda_causal_bias": float(getattr(args, 'lambda_causal_bias', 0.01)),
        "lambda_causal_pos": float(getattr(args, 'lambda_causal_pos', 0.01)),
        "lambda_causal_neg": float(getattr(args, 'lambda_causal_neg', 0.01)),
        "causal_pos_tau": float(getattr(args, 'causal_pos_tau', 0.1)),
        "min_causal_ratio": float(getattr(args, 'min_causal_ratio', 0.08)),
        "use_moe_router": bool(int(getattr(args, 'use_moe_router', 0))),
        "router_topk_ratio": float(getattr(args, 'router_topk_ratio', 0.15)),
        "lambda_router_entropy": float(getattr(args, 'lambda_router_entropy', 0.002)),
        "lambda_router_calib": float(getattr(args, 'lambda_router_calib', 0.03)),
        "moe_post_warm_cem_scale": float(getattr(args, 'moe_post_warm_cem_scale', 0.06)),
        "moe_cem_ramp_epochs": int(getattr(args, 'moe_cem_ramp_epochs', 5)),
        "moe_warmup_epochs": int(getattr(args, 'moe_warmup_epochs', 5)),
        "cem_align_schedule": bool(int(getattr(args, 'cem_align_schedule', 0))),
        "cem_align_start_epoch": int(getattr(args, 'cem_align_start_epoch', 10)),
        "cem_align_full_epoch": int(getattr(args, 'cem_align_full_epoch', 20)),
        "difficulty_gate_threshold": float(getattr(args, 'difficulty_gate_threshold', 0.0)),
        "causal_semantic_filter": float(args.causal_semantic_filter),
        "roberta_path": args.roberta_path,
        "min_quality_interventions": args.min_quality_interventions,
        "min_entity_overlap": args.min_entity_overlap,
        "sim_low": args.sim_low,
        "sim_high": args.sim_high,
        "sim_low_strong": args.sim_low_strong,
        "overlap_min_strong": args.overlap_min_strong,
        "causal_mask_causal_parts": bool(args.causal_mask_causal_parts),
        "abn_loss_weight": min(float(args.abn_loss_weight) * float(args.abn_loss_boost), 2.5),  # 不超过2.5防不稳定
        "open_loss_weight": getattr(args, 'open_loss_weight', 2.0),  # OPEN 问题更难，VQA-RAD 时提高
        "open_embedding_loss_weight": getattr(args, 'open_embedding_loss_weight', 1.5),  # CE梯度强，embedding matching时用较低权重
        "open_embedding_topk_soft": getattr(args, 'open_embedding_topk_soft', 5),  # Top-k soft target 同义词
        "open_embedding_soft_temp": getattr(args, 'open_embedding_soft_temp', 0.07),  # soft target 温度
        "open_embedding_align_lam": getattr(args, 'open_embedding_align_lam', 0.2),  # alignment loss
        "open_embedding_hybrid_weight": getattr(args, 'open_embedding_hybrid_weight', 0.5),  # 混合 BCE+CE
        "use_ce_for_answer": bool(getattr(args, 'use_ce_for_answer', 1)),
        "use_slake_concept": bool(getattr(args, 'use_slake_concept', 0)),
        "lambda_concept": getattr(args, 'lambda_concept', 0.0),  # concept 实验；默认 0
        "use_causal_aligned_cem": bool(int(getattr(args, 'use_causal_aligned_cem', 1))),
        "lambda_cem_align": float(getattr(args, 'lambda_cem_align', 0.002)),
        "lambda_cem_guard": float(getattr(args, 'lambda_cem_guard', 0.005)),
        "lambda_cem_gate_align": float(getattr(args, 'lambda_cem_gate_align', 0.003)),
        "lambda_ie_reg": float(getattr(args, 'lambda_ie_reg', 0.001)),
        "lambda_ie_floor": float(getattr(args, 'lambda_ie_floor', 0.002)),
        "ie_floor_threshold": float(getattr(args, 'ie_floor_threshold', 0.08)),
        "lambda_ie_de_coupling": float(getattr(args, 'lambda_ie_de_coupling', 0.002)),
        "lambda_causal_feat_consistency": float(getattr(args, 'lambda_causal_feat_consistency', 0.0025)),
        "use_feature_causal_probe": bool(int(getattr(args, 'use_feature_causal_probe', 0))),
        "probe_text_token_ratio": float(getattr(args, 'probe_text_token_ratio', 0.0)),
        "probe_patch_zero_ratio": float(getattr(args, 'probe_patch_zero_ratio', 0.0)),
        "lambda_open": getattr(args, 'lambda_open', 0.0),  # L_open_refine: Open 样本局部 patch 强化
        "open_refine_v_topk_ratio": getattr(args, 'open_refine_v_topk_ratio', 0.2),
        "concept_focus_refine_weight": getattr(args, 'concept_focus_refine_weight', 1.3),
        "organ_loss_weight": getattr(args, 'organ_loss_weight', 1.2),
        "ablation_no_open_refine": getattr(args, 'ablation_no_open_refine', False),
        "abn_hcss_target": args.abn_hcss_target,
        "abn_hcss_margin": args.abn_hcss_margin,
        "abn_ratio_target": args.abn_ratio_target,
        "abn_lam": args.abn_lam,
        "allow_last_resort_interventions": bool(args.allow_last_resort_interventions),
        "relax_sim_low": args.relax_sim_low,
        "teacher_ema_decay": args.teacher_ema_decay,
        "causal_lam": args.causal_lam,
        "lambda_hcss": getattr(args, 'lambda_hcss', -1),
        "lambda_ccs": getattr(args, 'lambda_ccs', -1),
        "hcss_alpha": getattr(args, 'hcss_alpha', 1.0),
        "hcss_mu": getattr(args, 'hcss_mu', 1.0),
        "hcss_stage1_epochs": getattr(args, 'hcss_stage1_epochs', 3),
        "ccs_lam": getattr(args, 'ccs_lam', 0.01),
        "mask_schedule": getattr(args, 'mask_schedule', 'step'),
        "epochs": args.epochs,
        "ccs_alpha": getattr(args, 'ccs_alpha', 4.0),
        "hcss_topk_ratio": args.hcss_topk_ratio,
        "v_causal_topk_ratio": args.v_causal_topk_ratio,
        "ccs_negative_weight": args.ccs_negative_weight,
        "ccs_negative_weight_min": args.ccs_negative_weight_min,
        "ccs_text_de_scale": args.ccs_text_de_scale,
        "ccs_mask_ratio": args.schedule_mask_ratio if getattr(args, 'schedule_mask_ratio', -1) >= 0 else args.ccs_mask_ratio,
        "ccs_topk_local": args.ccs_topk_local,
        "ccs_tau": args.ccs_tau,
        "local_ie_alpha": args.local_ie_alpha,
        "ccs_use_local_ie": bool(args.ccs_use_local_ie),
        "ccs_target": args.ccs_target,
        "ccs_penalty_lambda": args.ccs_penalty_lambda,
        "sign_adj_margin": args.sign_adj_margin,
        "sign_adj_temp": args.sign_adj_temp,
        "hcss_norm_tau": args.hcss_norm_tau,
        "hcss_floor": args.hcss_floor,
        "invariance_lambda": 0.0,
        "invariance_margin": 0.25,
        "invariance_ccs_threshold": 0.05,
        "energy_lambda": 0.0,
        "vis_boost_lambda": args.vis_boost_lambda,
        "text_de_penalty_lambda": args.text_de_penalty_lambda,
        "lambda_bias": 0.0,
        "lambda_bias_dynamic": False,
        "ablation_no_hcss": getattr(args, 'ablation_no_hcss', False),
        "ablation_no_ccs": getattr(args, 'ablation_no_ccs', False),
        "invariance_use_pure_vision": True,
        "visual_ie_auto_anneal": bool(args.visual_ie_auto_anneal),
        "visual_ie_anneal_state": None,  # 滞回状态: recovery | normal | stable
        "use_feature_gate": bool(args.use_feature_gate),
        "gate_alpha": args.gate_alpha,
        "gate_beta": args.gate_beta,
        "log_diagnostic_interval": args.log_diagnostic_interval,
        "use_causal_schedule": bool(getattr(args, 'use_causal_schedule', 1)),
        "use_light_causal_weights": bool(getattr(args, 'use_light_causal_weights', 0)),
        "bias_weight": getattr(args, 'bias_weight', 0.5),
        "causal_dropout": getattr(args, 'causal_dropout', 0.0),
        "causal_dropout_start_epoch": getattr(args, 'causal_dropout_start_epoch', 20),
        "lambda_counterfactual": float(getattr(args, 'lambda_counterfactual', 0.0)),
        "counterfactual_margin": float(getattr(args, 'counterfactual_margin', 0.5)),
        "counterfactual_start_epoch": int(getattr(args, 'counterfactual_start_epoch', 10)),
        "counterfactual_only_closed": bool(int(getattr(args, 'counterfactual_only_closed', 1))),
        "cem_gate_align_start_epoch": int(getattr(args, 'cem_gate_align_start_epoch', 10)),
        "ie_reg_start_epoch": int(getattr(args, 'ie_reg_start_epoch', 10)),
        "hcss_ie_scale": float(getattr(args, 'hcss_ie_scale', 1.5)),
        "use_do_controller": bool(model_config.get("use_do_controller", True)),
        "lambda_do_consistency": float(getattr(args, 'lambda_do_consistency', 0.1)),
        "lambda_do_pref": float(getattr(args, 'lambda_do_pref', 0.05)),
        "do_loss_beta": float(getattr(args, 'do_loss_beta', 0.2)),
        "lambda_do_anchor": float(getattr(args, 'lambda_do_anchor', 0.0)),
        "lambda_offline_bank_align": float(getattr(args, "lambda_offline_bank_align", 0.01)),
        "fusion_bank_dim": int(getattr(args, "hidden_dim", 768)) * 2,
        "do_intervention_point": str(getattr(args, "do_intervention_point", "pre_fusion")),
        "causal_alpha": float(max(0.0, min(1.0, getattr(args, "causal_alpha", 0.5)))),
        "dynamic_causal_alpha": int(getattr(args, "dynamic_causal_alpha", 0)),
    }
    if bool(int(getattr(args, 'use_offline_causal', 0))):
        model_config["use_logits_cem"] = False
        model_config["use_do_controller"] = bool(args.use_causal) and bool(int(getattr(args, 'use_do_controller', 1)))
        train_config["use_do_controller"] = bool(model_config.get("use_do_controller", True))
        logger.info(
            "[OfflineCausal] Cached priors | logits CEM off | MoE=%s | do_point=%s"
            % (
                bool(int(getattr(args, "use_moe_router", 0))),
                str(getattr(args, "do_intervention_point", "pre_fusion")),
            )
        )
    if args.use_causal:
        if bool(model_config.get("use_do_controller", True)):
            logger.info(
                f"  [Causal] DO controller ON | offline_priors={bool(int(getattr(args, 'use_offline_causal', 0)))}"
                f" | fusion_dropout={_fusion_dropout_prob}"
            )
        else:
            logger.info("  [Causal] DO controller OFF (baseline forward)")

    # Initialize model
    logger.info("Initializing model...")
    model = CausalVQAModel(model_config)
    # model = M3AE(model_config)
    model = model.to(device)
    # Print number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # [Transfer] Optional staged checkpoint loading:
    # 1) --resume_from_meve (if provided), then
    # 2) --resume_from_baseline (if provided, overrides same-name params from MEVE).
    is_stage2 = bool(getattr(args, 'stage2', 0))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve_ckpt_path(path_raw: str) -> str:
        path = (path_raw or "").strip()
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        for base in [script_dir, os.getcwd()]:
            candidate = os.path.normpath(os.path.join(base, path))
            if os.path.exists(candidate):
                return candidate
        return path

    def _load_transfer_ckpt(ckpt_path: str, tag: str) -> bool:
        """Returns True iff loaded state_dict contained any ``moe_expert_*`` keys."""
        if not ckpt_path:
            return False
        if not os.path.exists(ckpt_path):
            logger.warning(
                f"{tag} checkpoint NOT FOUND: {ckpt_path} | script_dir={script_dir} cwd={os.getcwd()}."
            )
            return False
        logger.info("=" * 60)
        logger.info(f"  Loading {tag} checkpoint: {ckpt_path}")
        logger.info("  Transfer/fine-tune: backbone from checkpoint, vqa_head re-init if vocab differs")
        logger.info("=" * 60)
        try:
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            state_dict = dict(state_dict)
            had_moe_expert = any(str(k).startswith("moe_expert_") for k in state_dict.keys())
            # Strip "module." prefix (DataParallel) for single-GPU loading
            if any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                had_moe_expert = any(str(k).startswith("moe_expert_") for k in state_dict.keys())
            # Remap vqa_head keys: checkpoint may have dropout (vqa_head.4) while current may use vqa_head.3
            head_wk = 'vqa_head.4.weight' if args.head_dropout > 0 else 'vqa_head.3.weight'
            head_bk = 'vqa_head.4.bias' if args.head_dropout > 0 else 'vqa_head.3.bias'
            if args.head_dropout > 0 and 'vqa_head.3.weight' in state_dict:
                state_dict['vqa_head.4.weight'] = state_dict.pop('vqa_head.3.weight')
                if 'vqa_head.3.bias' in state_dict:
                    state_dict['vqa_head.4.bias'] = state_dict.pop('vqa_head.3.bias')
            elif args.head_dropout == 0 and 'vqa_head.4.weight' in state_dict:
                state_dict['vqa_head.3.weight'] = state_dict.pop('vqa_head.4.weight')
                if 'vqa_head.4.bias' in state_dict:
                    state_dict['vqa_head.3.bias'] = state_dict.pop('vqa_head.4.bias')
                logger.info("  Remapped: vqa_head.4 -> vqa_head.3 (checkpoint had dropout, run has none)")

            # Handle vocab size mismatch
            ckpt_num_classes = num_classes
            if head_wk in state_dict:
                ckpt_num_classes = state_dict[head_wk].shape[0]
            if ckpt_num_classes != num_classes:
                logger.info(f"  Vocab mismatch: ckpt={ckpt_num_classes} vs model={num_classes}, skipping vqa_head (transfer learning)")
                state_dict.pop(head_wk, None)
                state_dict.pop(head_bk, None)

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"  Loaded {tag}: {len(state_dict)} keys | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
            if missing:
                logger.warning(f"  Missing keys (may cause random init): {missing}")
            if unexpected:
                logger.warning(f"  Unexpected keys (ignored): {unexpected}")
            vqa_missing = [k for k in missing if 'vqa_head' in k]
            if vqa_missing and ckpt_num_classes == num_classes:
                logger.error("  CRITICAL: vqa_head not fully loaded! Check checkpoint/model config match.")
            return had_moe_expert
        except Exception as e:
            logger.error(f"Failed to load {tag} checkpoint: {e}")
            raise

    meve_path = _resolve_ckpt_path(getattr(args, 'resume_from_meve', ''))
    baseline_path = _resolve_ckpt_path(args.resume_from_baseline or "")
    moe_expert_in_any_ckpt = False
    if meve_path:
        moe_expert_in_any_ckpt = _load_transfer_ckpt(meve_path, "MEVE") or moe_expert_in_any_ckpt
    if baseline_path:
        moe_expert_in_any_ckpt = _load_transfer_ckpt(baseline_path, "baseline") or moe_expert_in_any_ckpt
    if bool(int(getattr(args, "use_moe_router", 0))) and (not moe_expert_in_any_ckpt):
        if hasattr(model, "_warm_start_moe_expert"):
            model._warm_start_moe_expert()
            logger.info(
                "[MoE] No moe_expert_* keys in loaded checkpoint(s); expert tower warm-started in-process."
            )

    # Teacher model for causal: computes HCSS/CCS (breaks feedback loop)
    teacher_model = None
    if args.use_causal and bool(getattr(args, "use_do_controller", 1)):
        import copy
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        logger.info("Frozen Teacher M0 initialized for DO signal generation.")
    elif args.use_causal and args.use_teacher:
        import copy
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        logger.info(f"Teacher (EMA) initialized for causal HCSS/CCS computation (decay={args.teacher_ema_decay})")
    elif args.use_causal:
        logger.info("Single-model mode: using Student for HCSS/CCS (no Teacher, saves GPU memory)")

    # Define loss function and optimizer
    structure_mask_generator = PseudoOrganMaskGenerator(image_size=(args.image_size, args.image_size),
                                                        patch_size=args.patch_size)
    criterion = F.binary_cross_entropy_with_logits

    # ========== Optimizer: 分模块学习率 (visual 5e-6, text 5e-6, classifier 1e-5, bias 2.5e-6) ==========
    vision_param_ids = {id(p) for p in model.vision_encoder.parameters()}
    text_param_ids = {id(p) for p in model.language_encoder.parameters()}
    head_param_ids = {id(p) for p in model.vqa_head.parameters()}
    concept_head = getattr(model, 'open_concept_head', None)
    concept_head_param_ids = {id(p) for p in concept_head.parameters()} if concept_head is not None else set()
    vision_params = [p for p in model.parameters() if id(p) in vision_param_ids]
    text_params = [p for p in model.parameters() if id(p) in text_param_ids]
    head_params = [p for p in model.parameters() if id(p) in head_param_ids]
    concept_head_params = [p for p in model.parameters() if id(p) in concept_head_param_ids]
    bias_params = [p for p in model.parameters() if id(p) not in vision_param_ids and id(p) not in text_param_ids and id(p) not in head_param_ids and id(p) not in concept_head_param_ids]
    is_stage2 = bool(getattr(args, 'stage2', 0))
    lr_vis = args.lr * (1.0 / 4.0 if is_stage2 else 1.0)  # Stage 2: backbone 1/4 lr
    lr_text = args.lr * (1.0 / 4.0 if is_stage2 else 1.0)
    lr_cls = getattr(args, 'lr_classifier', 1e-5)
    use_concept_head = bool(getattr(args, 'use_concept_head', 0))
    lr_concept = getattr(args, 'lr_classifier', 1e-5) if (is_stage2 or use_concept_head) else 0.0
    lr_bias = getattr(args, 'lr_bias', 2.5e-6) * (1.0 / 3.0 if is_stage2 else 1.0)
    min_lr = getattr(args, 'min_lr', 5e-7)
    param_groups = [
        {"params": vision_params, "lr": lr_vis},
        {"params": text_params, "lr": lr_text},
        {"params": head_params, "lr": lr_cls},
        {"params": bias_params, "lr": lr_bias},
    ]
    if concept_head_params:
        param_groups.append({"params": concept_head_params, "lr": lr_concept or lr_cls})
    lr_log = f"visual={lr_vis:.2e} text={lr_text:.2e} classifier={lr_cls:.2e} bias={lr_bias:.2e}"
    if concept_head_params:
        lr_log += f" concept_head={lr_concept or lr_cls:.2e}"
    logger.info(f"Optimizer: AdamW | {lr_log} | wd={args.weight_decay}")
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    warmup_ratio = getattr(args, 'warmup_ratio', -1)
    warmup_epochs = max(1, int(args.epochs * warmup_ratio)) if warmup_ratio > 0 else getattr(args, 'warmup_epochs', 2)
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs - warmup_epochs, eta_min=min_lr)

    def lr_lambda(current_epoch):
        # LambdaLR starts with last_epoch=-1, so (current_epoch+1) would be 0 -> LR=0. Use max(0, current_epoch)+1.
        if current_epoch < warmup_epochs:
            return (max(0, current_epoch) + 1) / warmup_epochs
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    logger.info(f"LR schedule: {warmup_epochs}-epoch linear warmup → cosine annealing")

    # epoch = 35 if args.resume else 1
    epoch = 1
    best_val_score = 0
    early_stop_count = 0
    bank_align_warmup = max(0, int(getattr(args, "bank_align_warmup_epochs", 5)))
    original_bank_align = float(train_config.get("lambda_offline_bank_align", 0.02))
    bank_align_logged = False
    
    # AMP (mixed precision) for memory efficiency (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    logger.info("Mixed precision training (AMP) enabled" if scaler else "AMP disabled (CPU mode)")
    
    # Initialize loss record list
    training_history = []

    # Check if validation set is available
    has_val_data = val_loader is not None and len(val_loader) > 0
    has_test_data = test_loader is not None and len(test_loader) > 0
    eval_test_freq = getattr(args, 'eval_test_freq', 0)
    use_test_for_eval = (not has_val_data) and (eval_test_freq > 0) and has_test_data
    if not has_val_data:
        if use_test_for_eval:
            logger.info(
                f"merge_val_train: 每 {eval_test_freq} epoch 在 **test** 上评测一次 → **仅训练监控**（写入日志与 history），"
                f"**不**用 test 分数选 best、**不**据此 early stop；每轮保存 checkpoint_epoch_N.pth，请自行选模或默认用最后一轮。"
            )
        else:
            logger.warning(
                "No validation set: 训练中不跑 val/test；每轮保存 checkpoint_epoch_N.pth，结束时保存 best_model=最后一轮权重。"
            )

    # Load intervention bank if provided (optional; offline DO cache uses runtime + lexical fallbacks when absent)
    intervention_bank = None
    if args.intervention_path and os.path.exists(args.intervention_path):
        logger.info(f"Loading intervention bank from {args.intervention_path}...")
        try:
            intervention_bank = {}
            with open(args.intervention_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        q = item.get('original_question')
                        if q:
                            # Filter kept interventions or take all if 'kept' not present
                            kept_invs = [inv['text'] for inv in item.get('interventions', []) if inv.get('kept', True)]
                            if kept_invs:
                                intervention_bank[q] = kept_invs
                                intervention_bank[q.strip().lower()] = kept_invs
                    except:
                        continue
            logger.info(f"Loaded {len(intervention_bank)} questions from intervention bank.")
        except Exception as e:
            logger.error(f"Failed to load intervention bank: {e}")
    elif getattr(args, "intervention_path", ""):
        logger.warning(
            "intervention_path is set but file not found; continuing without JSONL bank (runtime + lexical interventions)."
        )
    else:
        logger.info("No intervention_path: using runtime + lexical interventions only (no pre-generated JSONL).")

    # Training loop
    if train_loader is None or len(train_loader) == 0:
        logger.error("No training data available (train_loader is None or empty). Check train_json and data_dir.")
        return model
    logger.info("Start training...")
    if args.use_causal and bool(getattr(args, "use_do_controller", 1)):
        if not train_cache_path:
            logger.error("DO mode requires --causal_cache_path_train or --causal_cache_path.")
            raise ValueError("Missing causal cache path")
        logger.info(
            f"Initializing offline signal caches | train={train_cache_path} | val={val_cache_path} | test={test_cache_path}"
        )
        skip_rebuild = bool(int(getattr(args, "skip_causal_cache_rebuild", 0)))
        if skip_rebuild and os.path.isfile(train_cache_path):
            logger.info(f"Skip train cache rebuild (--skip_causal_cache_rebuild=1): {train_cache_path}")
        else:
            rebuild_signal_cache(
                model_frozen=teacher_model if teacher_model is not None else model,
                data_loader=train_loader,
                tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
                intervention_bank=intervention_bank,
                device=device,
                output_path=train_cache_path,
                logger=logger,
                ema_momentum=float(getattr(args, "signal_ema_momentum", 0.9)),
                ccs_text_de_scale=float(getattr(args, "ccs_text_de_scale", 1.0)),
                store_fusion_bank=bool(int(getattr(args, "cache_store_fusion_repr", 1))),
            )
        if val_loader is not None and len(val_loader) > 0 and val_cache_path != train_cache_path:
            if skip_rebuild and os.path.isfile(val_cache_path):
                logger.info(f"Skip val cache rebuild: {val_cache_path}")
            else:
                logger.info(f"Building validation-only causal cache -> {val_cache_path}")
                rebuild_signal_cache(
                    model_frozen=teacher_model if teacher_model is not None else model,
                    data_loader=val_loader,
                    tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
                    intervention_bank=intervention_bank,
                    device=device,
                    output_path=val_cache_path,
                    logger=logger,
                    ema_momentum=float(getattr(args, "signal_ema_momentum", 0.9)),
                    ccs_text_de_scale=float(getattr(args, "ccs_text_de_scale", 1.0)),
                    store_fusion_bank=bool(int(getattr(args, "cache_store_fusion_repr", 1))),
                )
        if test_loader is not None and len(test_loader) > 0 and test_cache_path != train_cache_path:
            if (
                test_cache_path == val_cache_path
                and val_loader is not None
                and len(val_loader) > 0
                and val_cache_path != train_cache_path
            ):
                raise ValueError(
                    "When building separate val and test caches, --causal_cache_path_test must differ "
                    "from val cache path (defaults: val_cache.json vs test_cache.json)."
                )
            if skip_rebuild and os.path.isfile(test_cache_path):
                logger.info(f"Skip test cache rebuild: {test_cache_path}")
            else:
                logger.info(f"Building test-only causal cache -> {test_cache_path}")
                rebuild_signal_cache(
                    model_frozen=teacher_model if teacher_model is not None else model,
                    data_loader=test_loader,
                    tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
                    intervention_bank=intervention_bank,
                    device=device,
                    output_path=test_cache_path,
                    logger=logger,
                    ema_momentum=float(getattr(args, "signal_ema_momentum", 0.9)),
                    ccs_text_de_scale=float(getattr(args, "ccs_text_de_scale", 1.0)),
                    store_fusion_bank=bool(int(getattr(args, "cache_store_fusion_repr", 1))),
                )

    while epoch <= args.epochs:
        model.train()
        current_scheduler = warmup_scheduler if epoch <= warmup_epochs else cosine_scheduler

        if bank_align_warmup > 0 and epoch < bank_align_warmup:
            train_config["lambda_offline_bank_align"] = 0.0
        else:
            train_config["lambda_offline_bank_align"] = original_bank_align
            if (not bank_align_logged) and bank_align_warmup > 0 and epoch == bank_align_warmup:
                logger.info(
                    f"Epoch {epoch}: lambda_offline_bank_align enabled -> {original_bank_align}"
                )
                bank_align_logged = True

        if args.use_causal and bool(getattr(args, "use_do_controller", 1)):
            interval = int(getattr(args, "signal_update_interval", 5))
            ratio = float(getattr(args, "signal_update_ratio", 0.25))
            need_partial_update = (interval > 0) and (epoch > 1) and (epoch % interval == 0)
            if need_partial_update:
                logger.info(
                    f"Partial signal-cache refresh at epoch {epoch}: interval={interval}, ratio={ratio:.2f}"
                )
                partial_update_signal_cache(
                    model_frozen=teacher_model if teacher_model is not None else model,
                    data_loader=train_loader,
                    tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
                    intervention_bank=intervention_bank,
                    device=device,
                    output_path=train_cache_path,
                    update_ratio=ratio,
                    logger=logger,
                    ema_momentum=float(getattr(args, "signal_ema_momentum", 0.9)),
                    ccs_text_de_scale=float(getattr(args, "ccs_text_de_scale", 1.0)),
                    store_fusion_bank=bool(int(getattr(args, "cache_store_fusion_repr", 1))),
                )

        # Bias 分阶段: epoch>=ramp 时用 bias_weight_late
        ramp_ep = getattr(args, 'bias_weight_ramp_epoch', -1)
        bw_late = getattr(args, 'bias_weight_late', -1)
        if ramp_ep >= 0 and bw_late >= 0 and epoch >= ramp_ep:
            train_config["bias_weight"] = bw_late
        else:
            train_config["bias_weight"] = args.bias_weight

        # Train one epoch
        train_loss, train_acc, loss_info = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=current_scheduler,
            structure_mask_generator=structure_mask_generator,
            device=device,
            epoch=epoch,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            config=train_config,
            tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
            intervention_bank=intervention_bank,
            scaler=scaler,
            teacher_model=teacher_model,
        )
        
        # Visual IE 自动退火：滞回控制 + 弱 invariance 避免开关震荡
        # Skip when invariance disabled (invariance_lambda=0, Stage1/2 mode)
        if train_config.get("visual_ie_auto_anneal") and args.use_causal:
            vi = loss_info.get("avg_visual_ie", 0.0)
            prev = train_config.get("visual_ie_anneal_state") or "normal"
            LOW, HIGH, HYST = 0.10, 0.22, 0.02
            enter_recovery, leave_recovery = vi < (LOW - HYST), vi > (LOW + HYST)
            enter_stable, leave_stable = vi > (HIGH + HYST), vi < (HIGH - HYST)
            if enter_recovery or (prev == "recovery" and not leave_recovery):
                state = "recovery"
                train_config["ccs_mask_ratio"] = 0.7
                train_config["ccs_topk_local"] = 10
                train_config["ccs_tau"] = 0.03
                train_config["invariance_lambda"] = 0.005  # 弱约束，避免反复开关
                train_config["ccs_penalty_lambda"] = 0.0
                msg = f"  [visual_ie退火] vi={vi:.4f} → 塌缩恢复(滞回): ccs_mask_ratio=0.7 topk=10 tau=0.03 inv=0.005 pen=0"
            elif enter_stable or (prev == "stable" and not leave_stable):
                state = "stable"
                train_config["ccs_mask_ratio"] = 0.4
                train_config["invariance_lambda"] = 0.015
                train_config["ccs_penalty_lambda"] = 0.08
                msg = f"  [visual_ie退火] vi={vi:.4f} → 稳定收紧(滞回): ccs_mask_ratio=0.4 inv=0.015 pen=0.08"
            else:
                state = "normal"
                train_config["ccs_mask_ratio"] = 0.5
                train_config["ccs_topk_local"] = 6
                train_config["ccs_tau"] = 0.02
                train_config["invariance_lambda"] = 0.01
                train_config["ccs_penalty_lambda"] = 0.05
                msg = f"  [visual_ie退火] vi={vi:.4f} → 主力区间: ccs_mask_ratio=0.5 topk=6 tau=0.02 inv=0.01 pen=0.05"
            train_config["visual_ie_anneal_state"] = state
            logger.info(msg)
            tqdm.write(msg)

        # 特征 gate 连续退火（use_feature_gate=1 时生效，gate主导 + inv=base*exp(-var)，退火平滑）
        # Skip when invariance disabled (invariance_lambda=0, Stage1/2 mode)
        if train_config.get("use_feature_gate") and args.use_causal and train_config.get("invariance_lambda", 0) > 0:
            vi = loss_info.get("avg_visual_ie", 0.0)
            ccs_var = loss_info.get("ccs_var", 0.0)
            gate_alpha, new_inv = _anneal_params_continuous(vi, ccs_var)
            prev_inv = train_config.get("invariance_lambda", new_inv)
            inv_lam = prev_inv * 0.9 + new_inv * 0.1  # 平滑，避免每 epoch 大跳
            train_config["gate_alpha"] = gate_alpha
            train_config["gate_beta"] = train_config.get("gate_beta", 0.8)
            train_config["invariance_lambda"] = inv_lam
            logger.info(f"  [gate连续退火] vi={vi:.4f} ccs_var={ccs_var:.4f} → gate_alpha={gate_alpha:.3f} inv={inv_lam:.4f} (平滑)")

        # Bias 分支控制: 已在 epoch 开始时按 bias_weight_ramp_epoch 分阶段设置

        # Record training loss information（含 Causal Stats / CCS 诊断 / CEM 分布 / Bias 控制）
        epoch_record = {
            'epoch': epoch,
            'total_cls_loss': loss_info['total_cls_loss'],
            'total_factor_loss': loss_info['total_factor_loss'],
            'total_loss': loss_info['total_loss'],
            'train_accuracy': loss_info['accuracy'],
            'enable_causal': loss_info['enable_causal'],
            'learning_rate': loss_info['learning_rate'],
            'guided_on_ratio': loss_info.get('guided_on_ratio', 0.0),
            'effective_mask_ratio': loss_info.get('effective_mask_ratio', 0.0),
            'avg_ccs': loss_info.get('avg_ccs', 0.0),
            'effective_inv_ratio': loss_info.get('effective_inv_ratio', 0.0),
            'ccs_gt_01_ratio': loss_info.get('ccs_gt_01_ratio', 0.0),
            'ccs_gt_02_ratio': loss_info.get('ccs_gt_02_ratio', 0.0),
            'avg_hcss': loss_info.get('avg_hcss', 0.0),
            'avg_visual_ie': loss_info.get('avg_visual_ie', 0.0),
            'avg_text_ie': loss_info.get('avg_text_ie', 0.0),
            'avg_text_de': loss_info.get('avg_text_de', 0.0),
            'sign_adj_ratio': loss_info.get('sign_adj_ratio', 0.0),
            'schedule_mask_ratio': loss_info.get('schedule_mask_ratio', 0.0),
            # Causal Stats
            'causal_guided_on_batches': loss_info.get('causal_guided_on_batches', 0),
            'causal_total_batches': loss_info.get('causal_total_batches', 0),
            'ccs_pos_ratio': loss_info.get('ccs_pos_ratio', 0.0),
            'ccs_neg_ratio': loss_info.get('ccs_neg_ratio', 0.0),
            'avg_visual_ie_g': loss_info.get('avg_visual_ie_g', 0.0),
            'avg_visual_ie_l': loss_info.get('avg_visual_ie_l', 0.0),
            # CCS=0 诊断
            'diag_empty_cnt': loss_info.get('diag_empty_cnt', 0),
            'diag_exception_cnt': loss_info.get('diag_exception_cnt', 0),
            'diag_ccs_patches_none_cnt': loss_info.get('diag_ccs_patches_none_cnt', 0),
            'diag_empty_reasons': loss_info.get('diag_empty_reasons', {}),
            'diag_last_exception': loss_info.get('diag_last_exception', ''),
            # CEM 分布
            'cem_visual_causal_pct': loss_info.get('cem_visual_causal_pct', 0.0),
            'cem_cross_modal_pct': loss_info.get('cem_cross_modal_pct', 0.0),
            'cem_text_causal_pct': loss_info.get('cem_text_causal_pct', 0.0),
            'cem_language_bias_pct': loss_info.get('cem_language_bias_pct', 0.0),
            'cem_visual_bias_pct': loss_info.get('cem_visual_bias_pct', 0.0),
            'cem_neutral_pct': loss_info.get('cem_neutral_pct', 0.0),
            # Bias 控制（Bias 控制块之后的值）
            'bias_weight': train_config.get('bias_weight', 0.5),
        }

        run_validation = (has_val_data and epoch % args.val_freq == 0) or (use_test_for_eval and epoch % eval_test_freq == 0)
        eval_loader = val_loader if has_val_data else (test_loader if use_test_for_eval else None)
        # test 上评测仅监控；只有 val 上的分数才用于选 best / early stop / collapse
        eval_updates_best = run_validation and eval_loader is not None and (not use_test_for_eval)
        if run_validation and eval_loader is not None:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eval_name = "Test" if use_test_for_eval else "Val"
            # 当 use_test_for_eval 且 use_teacher=1 时，用 EMA 权重评估 test，更平滑、泛化更好
            eval_model = teacher_model if (use_test_for_eval and teacher_model is not None) else model
            if use_test_for_eval and teacher_model is not None:
                logger.info("  Using EMA Teacher for test evaluation")
            # 兼容旧版 validate：若签名无 use_causal_gate 等参数则用旧调用
            _sig = inspect.signature(validate)
            _params = set(_sig.parameters)
            if 'use_causal_gate' in _params:
                val_loss, val_acc = validate(
                    model=eval_model,
                    data_loader=eval_loader,
                    criterion=criterion,
                    device=device,
                    use_amp=torch.cuda.is_available(),
                    use_causal_gate=bool(args.use_causal and getattr(args, 'use_causal_gate_in_val', 0)),
                    tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
                    intervention_bank=intervention_bank,
                    config=train_config,
                )
            else:
                logger.warning("validate() 为旧版签名，Val 不使用 causal gate。请同步 train.py 以启用 Val causal gate。")
                val_loss, val_acc = validate(
                    model=eval_model,
                    data_loader=eval_loader,
                    criterion=criterion,
                    device=device,
                    use_amp=torch.cuda.is_available(),
                )
            current_score = val_acc

            # Add validation/test info to record
            epoch_record.update({
                f'{eval_name.lower()}_loss': val_loss,
                f'{eval_name.lower()}_accuracy': val_acc
            })

            logger.info(
                f"Epoch {epoch}: Train loss {train_loss:.4f}, Train accuracy {train_acc:.2f}%, {eval_name} loss {val_loss:.4f}, {eval_name} accuracy {val_acc:.2f}%")
            if use_test_for_eval:
                logger.info(f"  (Test 指标仅作监控，不参与 best 与 early stop)")

            # 仅用 Val 更新 best / early stop；merge+test 监控时跳过
            if eval_updates_best:
                if current_score > best_val_score:
                    logger.info(f"{eval_name} performance improved: {best_val_score:.2f}% -> {current_score:.2f}%. Saving model...")
                    best_val_score = current_score

                    mode_tag = "causal" if args.use_causal else "baseline"
                    best_model_path = os.path.join(args.save_dir, f'best_model_{mode_tag}.pth')
                    torch.save(model.state_dict(), best_model_path)

                    if teacher_model is not None:
                        best_ema_path = os.path.join(args.save_dir, f'best_model_{mode_tag}_ema.pth')
                        torch.save(teacher_model.state_dict(), best_ema_path)
                        logger.info(f"  EMA Teacher saved to {best_ema_path} (use --use_ema 1 in test.py to load)")

                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    logger.info(f"{eval_name} performance did not improve. Early stop count: {early_stop_count}/{args.early_stop}")
        else:
            # Do not save model and continue training for non-validation epochs
            logger.info(f"Epoch {epoch}: Train loss {train_loss:.4f}, Train accuracy {train_acc:.2f}%")
        
        # Add current epoch record to history
        training_history.append(epoch_record)

        # Score 趋势: 最近 N 个 epoch 的 train/val 或 train/test
        trend_n = min(5, len(training_history))
        trend_records = training_history[-trend_n:]
        train_scores = [r['train_accuracy'] for r in trend_records]
        eval_key = 'test_accuracy' if use_test_for_eval else 'val_accuracy'
        eval_scores = [r.get(eval_key, None) for r in trend_records]
        trend_label = "Test" if use_test_for_eval else "Val"
        trend_str = f"Score趋势(近{trend_n}ep): Train " + "->".join(f"{s:.1f}" for s in train_scores)
        if any(v is not None for v in eval_scores):
            trend_str += f" | {trend_label} " + "->".join(f"{v:.1f}" if v is not None else "-" for v in eval_scores)
        logger.info(trend_str)
        tqdm.write(trend_str)

        # Save training history to file
        history_file = os.path.join(args.save_dir, 'our_training_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training history saved to: {history_file}")

        # 每轮保存权重（便于 merge 场景按 epoch 自选模型；不用 test 分数选 best）
        mode_tag_ep = "causal" if args.use_causal else "baseline"
        epoch_ckpt = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(model.state_dict(), epoch_ckpt)
        logger.info(f"Saved epoch checkpoint: {epoch_ckpt}")
        if teacher_model is not None:
            ema_epoch_ckpt = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}_ema.pth')
            torch.save(teacher_model.state_dict(), ema_epoch_ckpt)
            logger.info(f"Saved epoch EMA checkpoint: {ema_epoch_ckpt}")

        # Check early stopping（仅 Val 上「未提升」会计数；纯 test 监控时不因 test 变差而 stop）
        if args.early_stop > 0 and early_stop_count >= args.early_stop:
            logger.info(f"Early stopping triggered, no performance improvement for {args.early_stop} epochs. Stopping training.")
            break

        # Collapse guard: 仅 Val，不用 test
        if eval_updates_best and run_validation and args.val_collapse_threshold > 0 and best_val_score > 0:
            if val_acc < best_val_score - args.val_collapse_threshold:
                logger.warning(f"Val collapse detected: {val_acc:.2f}% < best {best_val_score:.2f}% - {args.val_collapse_threshold}%. Stopping to preserve best checkpoint.")
                break

        # LR decay after epoch N (e.g. epoch 4 后 lr *= 0.5，减缓过拟合)
        if args.lr_decay_epoch > 0 and epoch == args.lr_decay_epoch:
            for g in optimizer.param_groups:
                old_lr = g['lr']
                g['lr'] = old_lr * args.lr_decay_factor
                logger.info(f"LR decay at epoch {epoch}: {old_lr:.2e} -> {g['lr']:.2e} (factor={args.lr_decay_factor})")

        epoch += 1

    # 无 val 时：best_model_* 始终表示「最后一轮」权重，**不是** test 最优（test 仅监控时可开 eval_test_freq）
    if not has_val_data:
        mode_tag = "causal" if args.use_causal else "baseline"
        best_model_path = os.path.join(args.save_dir, f'best_model_{mode_tag}.pth')
        torch.save(model.state_dict(), best_model_path)
        if use_test_for_eval:
            logger.info(
                f"Final model saved to {best_model_path} (last epoch; test 仅监控未用于选 best；"
                f"可选 checkpoint_epoch_*.pth 自行挑模)"
            )
        else:
            logger.info(f"Final model saved to {best_model_path} (no val/test during train, last epoch)")
        if teacher_model is not None:
            best_ema_path = os.path.join(args.save_dir, f'best_model_{mode_tag}_ema.pth')
            torch.save(teacher_model.state_dict(), best_ema_path)
            logger.info(f"EMA Teacher saved to {best_ema_path} (use --use_ema 1 in test.py to load)")

    if best_val_score > 0:
        logger.info(f"Training complete! Best Val accuracy (used for best_model during train): {best_val_score:.2f}%")
    else:
        logger.info("Training complete! No Val-based best (merge 场景请用 checkpoint_epoch_*.pth 或最后一轮 best_model_*.pth)")

    # 训练结束：用 100% val set 做最终评估（训练时若用了 subset）
    val_subset_ratio = getattr(args, 'val_subset_ratio', 1.0)
    if has_val_data and val_subset_ratio < 1.0 and hasattr(data_loader, 'val_dataset_full') and data_loader.val_dataset_full is not None:
        from torch.utils.data import DataLoader
        full_val_loader = DataLoader(
            data_loader.val_dataset_full,
            batch_size=data_loader.val_batch_size,
            shuffle=False,
            num_workers=getattr(args, 'val_num_workers', 0),
            persistent_workers=False,
            collate_fn=data_loader.collate_fn,
            pin_memory=True
        )
        logger.info("Running final validation on full val set (100%)...")
        _sig = inspect.signature(validate)
        _params = set(_sig.parameters)
        if 'use_causal_gate' in _params:
            final_val_loss, final_val_acc = validate(
                model=model,
                data_loader=full_val_loader,
                criterion=criterion,
                device=device,
                use_amp=torch.cuda.is_available(),
                use_causal_gate=bool(args.use_causal and getattr(args, 'use_causal_gate_in_val', 0)),
                tokenizer=data_loader.train_dataset.tokenizer if data_loader.train_dataset else None,
                intervention_bank=intervention_bank,
                config=train_config,
            )
        else:
            final_val_loss, final_val_acc = validate(
                model=model,
                data_loader=full_val_loader,
                criterion=criterion,
                device=device,
                use_amp=torch.cuda.is_available(),
            )
        logger.info(f"Final validation (full val set): Loss {final_val_loss:.4f} | Accuracy {final_val_acc:.2f}%")
        training_history.append({'epoch': 'final_full_val', 'val_loss': final_val_loss, 'val_accuracy': final_val_acc})
    
    # Save final training history
    final_history_file = os.path.join(args.save_dir, 'final_our_training_history.json')
    with open(final_history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Final training history saved to: {final_history_file}")
    
    return model


if __name__ == "__main__":
    import random
    args = parse_args()
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
