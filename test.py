import os
import torch
import argparse
import logging
import json  #
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from models.vqa_module import CausalVQAModel
from models.m3ae import M3AE
from utils.dataloader import VQADataLoader
from train import compute_score_with_logits, prepare_batch_data, filter_kwargs_for_causal_masks, moe_probe_router_batch
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _load_offline_causal_cache(cache_path):
    if not cache_path or (not os.path.exists(cache_path)):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        items = raw["items"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [{"sample_id": k, **(v if isinstance(v, dict) else {})} for k, v in raw.items()]
    else:
        items = []
    out = {}
    for it in items:
        sid = str(it.get("sample_id", "")).strip()
        if sid:
            out[sid] = it
    return out


def _get_cached_signals_for_batch(batch, cache, device):
    qids = batch.get("qid", [])
    bsz = len(qids)
    ccs = torch.ones(bsz, device=device, dtype=torch.float32)
    hcss = torch.ones(bsz, device=device, dtype=torch.float32)
    text_de = torch.zeros(bsz, device=device, dtype=torch.float32)
    vis_ie = torch.zeros(bsz, device=device, dtype=torch.float32)
    valid_mask = torch.zeros(bsz, device=device, dtype=torch.float32)
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
    return {"ccs": ccs, "hcss": hcss, "text_de": text_de, "vis_ie": vis_ie, "valid_mask": valid_mask}

def parse_args():
    parser = argparse.ArgumentParser(description='Medical VQA System Test')

    # Data-related parameters
    parser.add_argument('--data_dir', type=str, default='data_med', help='Root data directory')
    parser.add_argument('--image_dir', type=str, default='data_med/images', help='Image directory')
    parser.add_argument('--test_json', type=str, default='data_med/test_typed.jsonl', help='Test data JSON/JSONL')
    parser.add_argument('--train_json', type=str, default='', help='Train JSON/JSONL (optional; for vocab / parity with train)')
    parser.add_argument('--val_json', type=str, default='', help='Val JSON/JSONL (optional; for vocab / parity with train)')

    parser.add_argument('--load_path', type=str, default='pretrained_weights/m3ae.ckpt', help='Pretrained weights path')
    parser.add_argument('--embeddings_dir', type=str, default='data_med/embeddings_all', help='Embeddings directory')
    parser.add_argument('--roberta_path', type=str, default='pretrain/roberta-base',
                        help='Local path to roberta-base model directory')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='', help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    parser.add_argument('--vocab', type=str, default='roberta', help='Vocabulary')
    parser.add_argument('--image_size', type=int, default=384, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--max_length', type=int, default=32, help='Max sequence length')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden dimension')
    parser.add_argument('--num_top_layer', type=int, default=6, help='attention layer')
    parser.add_argument('--input_image_embed_size', type=int, default=768, help='Visual feature dimension')
    parser.add_argument('--input_text_embed_size', type=int, default=768, help='Question feature dimension')

    parser.add_argument('--seed', type=int, default=105, help='Random seed')
    parser.add_argument('--use_causal', type=int, default=0, help='Model mode flag (0 baseline, 1 causal)')
    parser.add_argument('--use_do_controller', type=int, default=1, help='Use DO controller in test forward')
    parser.add_argument('--do_mode', type=str, default='hard_mask', choices=['hard_mask'], help='DO mode; keep aligned with training')
    parser.add_argument('--do_ccs_margin', type=float, default=0.1, help='Hard DO trigger margin: apply when ccs < -margin')
    parser.add_argument('--do_keep_base', type=float, default=0.85, help='Conservative mask scaling keep_base')
    parser.add_argument(
        '--do_intervention_point',
        type=str,
        default='pre_fusion',
        choices=['pre_fusion', 'post_fusion'],
        help='与训练 main --do_intervention_point 一致',
    )
    parser.add_argument('--causal_alpha', type=float, default=0.5, help='与训练 --causal_alpha 一致，α·F_do+(1-α)·F_clean')
    parser.add_argument('--dynamic_causal_alpha', type=int, default=1, help='与训练 --dynamic_causal_alpha 一致')
    parser.add_argument('--do_gate_tau', type=float, default=1.2, help='Soft DO gate tau on standardized CCS (sigmoid(-z_ccs/tau)); align with training main.py')
    parser.add_argument('--do_gate_bias', type=float, default=0.2, help='Soft DO gate bias on standardized CCS (sigmoid(-(z_ccs-bias)/tau)); align with training')
    parser.add_argument('--do_logit_tau', type=float, default=1.0, help='Decision-space DO residual multiplier; align with training main.py')
    parser.add_argument('--do_delta_scale', type=float, default=1.0, help='Delta-logit norm scale (tanh bounded); align with training main.py')
    parser.add_argument('--do_residual_clamp_k', type=float, default=1.5, help='Residual clamp k for DO logits shift; align with training main.py')
    parser.add_argument('--use_offline_causal', type=int, default=1, help='Use offline cached causal signals in test')
    parser.add_argument('--causal_cache_path', type=str, default='', help='Offline causal cache JSON (legacy: used if causal_cache_path_test empty)')
    parser.add_argument(
        '--causal_cache_path_test',
        type=str,
        default='',
        help='测试集专用 cache（与 train 分离）；为空则回退到 --causal_cache_path',
    )
    parser.add_argument('--use_causal_gate_in_test', type=int, default=0,
                        help='0=测试不做实时干预（默认，与 main 默认 use_causal_gate_in_val=0 一致）；'
                             '1=测试跑 HCSS+CCS+干预（需 intervention_path 等与训练对齐）')
    parser.add_argument('--causal_max_interventions_val_test', type=int, default=2, help='Val/Test干预数(2=推理成本~50%%↓, 与train的4区分)')
    parser.add_argument('--intervention_path', type=str, default='', help='Path to intervention bank (for causal gate)')
    parser.add_argument('--head_dropout', type=float, default=-1, help='Head dropout (default -1: auto-detect from checkpoint)')
    parser.add_argument('--use_ema', type=int, default=0, help='1=load EMA Teacher weights (训练时 use_teacher=1 保存的 _ema.pth)，减少 val/test 差异')
    parser.add_argument('--checkpoint_ema', type=str, default='', help='Explicit path to EMA checkpoint; if empty and use_ema=1, auto-derive from checkpoint (xxx.pth -> xxx_ema.pth)')
    parser.add_argument('--use_open_embedding_matching', type=int, default=0, help='1=Open 用 embedding matching（需与训练一致）')
    parser.add_argument('--answer_embeddings_path', type=str, default='', help='answer_embeddings.pt 路径')
    parser.add_argument('--use_vqa_rad_concept', type=int, default=0, help='1=VQA-RAD concept（需与训练一致）')
    parser.add_argument('--ablation_no_hcss', action='store_true', help='强消融: 与训练时 --ablation_no_hcss 一致')
    parser.add_argument('--ablation_no_ccs', action='store_true', help='强消融: 与训练时 --ablation_no_ccs 一致')

    # 因果评测超参：必须与训练时 main.py 一致，否则 HCSS/CCS/mask 与 validate 不对齐，分数会异常偏低
    parser.add_argument('--use_feature_gate', type=int, default=1, help='1=与训练一致 feature gate（默认同 main）')
    parser.add_argument('--gate_alpha', type=float, default=1.0, help='同 main --gate_alpha')
    parser.add_argument('--gate_beta', type=float, default=0.8, help='同 main --gate_beta')
    parser.add_argument('--hcss_topk_ratio', type=float, default=0.30, help='同 main --hcss_topk_ratio')
    parser.add_argument('--v_causal_topk_ratio', type=float, default=0.4, help='同 main --v_causal_topk_ratio')
    parser.add_argument('--ccs_mask_ratio', type=float, default=0.22, help='同 main --ccs_mask_ratio（训练时 schedule 会改，测试请用手动训练末 epoch 常用值）')
    parser.add_argument('--causal_mask_causal_parts', type=int, default=1, help='同 main --causal_mask_causal_parts')
    parser.add_argument('--causal_max_interventions', type=int, default=3, help='同 main --causal_max_interventions（max_interventions 回退用）')
    parser.add_argument('--min_quality_interventions', type=int, default=1, help='同 main')
    parser.add_argument('--min_entity_overlap', type=float, default=0.40, help='同 main')
    parser.add_argument('--sim_low', type=float, default=0.55, help='同 main')
    parser.add_argument('--sim_high', type=float, default=0.90, help='同 main')
    parser.add_argument('--sim_low_strong', type=float, default=0.25, help='同 main')
    parser.add_argument('--overlap_min_strong', type=float, default=0.02, help='同 main')
    parser.add_argument('--relax_sim_low', type=float, default=0.50, help='同 main')
    parser.add_argument('--relax_min_entity_overlap', type=float, default=0.15, help='同 main')
    parser.add_argument('--allow_last_resort_interventions', type=int, default=0, help='同 main')
    parser.add_argument('--ccs_topk_local', type=int, default=5, help='同 main')
    parser.add_argument('--ccs_tau', type=float, default=0.01, help='同 main')
    parser.add_argument('--local_ie_alpha', type=float, default=1.0, help='同 main')
    parser.add_argument('--ccs_use_local_ie', type=int, default=1, help='同 main')
    parser.add_argument('--ccs_target', type=float, default=0.2, help='同 main')
    parser.add_argument('--ccs_penalty_lambda', type=float, default=0.08, help='同 main')
    parser.add_argument('--hcss_ie_scale', type=float, default=1.5,
                        help='同 main --hcss_ie_scale：HCSS core 中 IE_mean 乘性放大（实时 HCSS 与训练对齐）')
    parser.add_argument('--ccs_text_de_scale', type=float, default=1.0, help='同 main（旧 test 写死 15 会导致与训练不一致）')
    # 与训练 vqa_module 静态字段一致（checkpoint 权重相同，但前向尺度/CEM 行为依赖这些；缺省与 main 默认一致）
    parser.add_argument('--repr_gate_alpha', type=float, default=0.2, help='同 main --repr_gate_alpha')
    parser.add_argument('--repr_gate_beta', type=float, default=0.2, help='同 main --repr_gate_beta')
    parser.add_argument('--causal_soft_alpha', type=float, default=-1.0, help='>=0 时覆盖 repr_gate_alpha（同 main）')
    parser.add_argument('--causal_soft_beta', type=float, default=-1.0, help='>=0 时覆盖 repr_gate_beta')
    parser.add_argument('--cem_gamma', type=float, default=1.0, help='CEM softmax 温度，同 main --cem_gamma')
    parser.add_argument('--cem_direction_k', type=float, default=2.0, help='同 main --cem_direction_k（CEM direction 斜率）')
    parser.add_argument('--use_logits_cem', type=int, default=1, help='1=logits 层 CEM（默认）；0=关闭，同 main')
    parser.add_argument('--use_moe_router', type=int, default=0, help='1=与训练一致 MoE router + 稀疏因果子集')
    parser.add_argument('--router_topk_ratio', type=float, default=0.15, help='同 main')
    parser.add_argument('--causal_router_hidden', type=int, default=-1, help='同 main')
    parser.add_argument('--backbone_dropout', type=float, default=0.0, help='同 main；需与训练时结构一致')
    parser.add_argument('--fusion_dropout_prob', type=float, default=-1.0, help='同 main --fusion_dropout_prob（BertCross 层，须与训练一致）')

    return parser.parse_args()


def build_causal_eval_config(args):
    """与 train.validate(..., config=train_config) 同源字段；测试时请传与训练相同的因果相关参数。"""
    if not (bool(getattr(args, 'use_causal', 0)) and bool(getattr(args, 'use_causal_gate_in_test', 0))):
        return {}
    return {
        'use_feature_gate': bool(getattr(args, 'use_feature_gate', 1)),
        'gate_alpha': float(getattr(args, 'gate_alpha', 1.0)),
        'gate_beta': float(getattr(args, 'gate_beta', 0.8)),
        'ccs_mask_ratio': float(getattr(args, 'ccs_mask_ratio', 0.22)),
        'hcss_topk_ratio': float(getattr(args, 'hcss_topk_ratio', 0.30)),
        'v_causal_topk_ratio': float(getattr(args, 'v_causal_topk_ratio', 0.4)),
        'causal_mask_causal_parts': bool(getattr(args, 'causal_mask_causal_parts', 1)),
        'min_quality_interventions': int(getattr(args, 'min_quality_interventions', 1)),
        'min_entity_overlap': float(getattr(args, 'min_entity_overlap', 0.40)),
        'sim_low': float(getattr(args, 'sim_low', 0.55)),
        'sim_high': float(getattr(args, 'sim_high', 0.90)),
        'sim_low_strong': float(getattr(args, 'sim_low_strong', 0.25)),
        'overlap_min_strong': float(getattr(args, 'overlap_min_strong', 0.02)),
        'relax_sim_low': float(getattr(args, 'relax_sim_low', 0.50)),
        'relax_min_entity_overlap': float(getattr(args, 'relax_min_entity_overlap', 0.15)),
        'allow_last_resort_interventions': bool(getattr(args, 'allow_last_resort_interventions', 0)),
        'causal_max_interventions': int(getattr(args, 'causal_max_interventions', 3)),
        'causal_max_interventions_val_test': int(getattr(args, 'causal_max_interventions_val_test', 2)),
        'ccs_topk_local': int(getattr(args, 'ccs_topk_local', 5)),
        'ccs_tau': float(getattr(args, 'ccs_tau', 0.01)),
        'local_ie_alpha': float(getattr(args, 'local_ie_alpha', 1.0)),
        'ccs_use_local_ie': bool(getattr(args, 'ccs_use_local_ie', 1)),
        'ccs_target': float(getattr(args, 'ccs_target', 0.2)),
        'ccs_penalty_lambda': float(getattr(args, 'ccs_penalty_lambda', 0.08)),
        'ccs_text_de_scale': float(getattr(args, 'ccs_text_de_scale', 1.0)),
        'hcss_ie_scale': float(getattr(args, 'hcss_ie_scale', 1.5)),
        'ablation_no_hcss': bool(getattr(args, 'ablation_no_hcss', False)),
        'ablation_no_ccs': bool(getattr(args, 'ablation_no_ccs', False)),
        'use_moe_router': bool(int(getattr(args, 'use_moe_router', 0))),
        'router_topk_ratio': float(getattr(args, 'router_topk_ratio', 0.15)),
        'causal_ratio': float(getattr(args, 'causal_ratio', 0.15)),
    }


def test_accuracy(args):
    """Evaluate model accuracy on the test set"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if float(getattr(args, 'causal_soft_alpha', -1.0)) >= 0.0:
        args.repr_gate_alpha = float(args.causal_soft_alpha)
    if float(getattr(args, 'causal_soft_beta', -1.0)) >= 0.0:
        args.repr_gate_beta = float(args.causal_soft_beta)

    # Load checkpoint (support both raw state_dict and wrapped {'state_dict': ...})
    # use_ema=1: 加载 EMA Teacher 权重，减少 val/test 差异（训练时 use_teacher=1 会保存 _ema.pth）
    load_ema = bool(getattr(args, 'use_ema', 0))
    checkpoint_ema_path = getattr(args, 'checkpoint_ema', '').strip()

    if load_ema:
        # 优先用显式路径，否则从 checkpoint 推导 (xxx.pth -> xxx_ema.pth)
        if checkpoint_ema_path and os.path.exists(checkpoint_ema_path):
            ema_path = checkpoint_ema_path
        else:
            base, ext = os.path.splitext(args.checkpoint)
            ema_path = base + '_ema' + ext
        if os.path.exists(ema_path):
            logger.info(f"Loading EMA Teacher checkpoint: {ema_path}")
            ema_ckpt = torch.load(ema_path, map_location=device)
            checkpoint = ema_ckpt.get('state_dict', ema_ckpt) if isinstance(ema_ckpt, dict) else ema_ckpt
            logger.info("  Using EMA weights (smoother, better generalization)")
        else:
            logger.warning(f"EMA checkpoint not found: {ema_path}, falling back to Student checkpoint")
            raw_ckpt = torch.load(args.checkpoint, map_location=device)
            checkpoint = raw_ckpt.get('state_dict', raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
    else:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        raw_ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(raw_ckpt, dict) and 'state_dict' in raw_ckpt:
            checkpoint = raw_ckpt['state_dict']
            logger.info("  Detected wrapped checkpoint format (state_dict key)")
        else:
            checkpoint = raw_ckpt

    # Auto-detect head_dropout from checkpoint (vqa_head.4 = with dropout)
    head_dropout = args.head_dropout
    if head_dropout < 0:
        head_dropout = 0.1 if any(k.startswith('vqa_head.4.') for k in checkpoint.keys()) else 0.0

    # Configure data loader (与训练一致: skip_ae_maml、embeddings_dir 等)
    data_config = {
        'data_dir': args.data_dir,
        'image_dir': args.image_dir,
        'embeddings_dir': args.embeddings_dir,
        'train_json': getattr(args, 'train_json', '') or '',
        'val_json': getattr(args, 'val_json', '') or '',
        'test_json': args.test_json,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'tokenizer': args.vocab,
        'roberta_path': args.roberta_path,
        'image_size': args.image_size,
        'max_length': args.max_length,
        'device': str(device),
        'skip_ae_maml': True,
        'use_vqa_rad_concept': bool(getattr(args, 'use_vqa_rad_concept', 0)),
    }

    # Initialize data loader
    logger.info("Initializing data loader...")
    data_loader = VQADataLoader(data_config)
    loaders = data_loader.get_loaders()
    test_loader = loaders.get('test')

    if test_loader is None or len(test_loader) == 0:
        logger.error("Test data loading failed!")
        return

    # Initialize model
    logger.info("Initializing model...")
    ans_emb_path = getattr(args, 'answer_embeddings_path', '').strip()
    if not ans_emb_path and getattr(args, 'use_open_embedding_matching', 0):
        ans_emb_path = os.path.join(args.data_dir, 'answer_embeddings.pt')
    has_concept_head = any(k.startswith('open_concept_head') for k in checkpoint.keys())
    _moe_ckpt = any(str(k).startswith('router_trunk') for k in checkpoint.keys())
    _moe_arg = bool(int(getattr(args, 'use_moe_router', 0)))
    use_moe_build = _moe_arg or _moe_ckpt
    _fdp_arg = float(getattr(args, 'fusion_dropout_prob', -1.0))
    if _fdp_arg >= 0.0:
        _fusion_dropout_prob = _fdp_arg
    elif bool(args.use_causal):
        _fusion_dropout_prob = 0.05
    else:
        _fusion_dropout_prob = None

    model_config = {
        'hidden_size': args.hidden_size,
        'num_hid': data_loader.get_answer_vocab()['vocab_size'],
        'input_image_embed_size': args.input_image_embed_size,
        'input_text_embed_size': args.input_text_embed_size,
        'num_top_layer': args.num_top_layer,
        'visual_backbone': 'ViT-B/16',
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'load_path': args.load_path,
        'roberta_path': args.roberta_path,
        'use_causal': bool(args.use_causal),
        'use_do_controller': bool(getattr(args, 'use_do_controller', 1)),
        'do_mode': str(getattr(args, 'do_mode', 'hard_mask')),
        'do_ccs_margin': float(getattr(args, 'do_ccs_margin', 0.1)),
        'do_keep_base': float(getattr(args, 'do_keep_base', 0.85)),
        'do_intervention_point': str(getattr(args, 'do_intervention_point', 'pre_fusion')),
        'causal_alpha': float(max(0.0, min(1.0, getattr(args, 'causal_alpha', 0.5)))),
        'dynamic_causal_alpha': int(getattr(args, 'dynamic_causal_alpha', 1)),
        'do_gate_tau': float(getattr(args, 'do_gate_tau', 1.2)),
        'do_gate_bias': float(getattr(args, 'do_gate_bias', 0.2)),
        'do_logit_tau': float(getattr(args, 'do_logit_tau', 1.0)),
        'do_delta_scale': float(getattr(args, 'do_delta_scale', 1.0)),
        'do_residual_clamp_k': float(getattr(args, 'do_residual_clamp_k', 1.5)),
        'head_dropout': head_dropout,
        'backbone_dropout': float(getattr(args, 'backbone_dropout', 0.0)),
        'use_open_embedding_matching': bool(getattr(args, 'use_open_embedding_matching', 0)),
        'answer_embeddings_path': ans_emb_path,
        'use_open_concept_head': has_concept_head or bool(getattr(args, 'use_vqa_rad_concept', 0)),
        'num_concepts': 5,
        'repr_gate_alpha': float(getattr(args, 'repr_gate_alpha', 0.2)),
        'repr_gate_beta': float(getattr(args, 'repr_gate_beta', 0.2)),
        'cem_gamma': float(getattr(args, 'cem_gamma', 1.0)),
        'cem_direction_k': float(getattr(args, 'cem_direction_k', 2.0)),
        'use_logits_cem': bool(int(getattr(args, 'use_logits_cem', 1))),
        'use_moe_router': use_moe_build,
        'causal_router_hidden': int(getattr(args, 'causal_router_hidden', -1)),
        'fusion_dropout_prob': _fusion_dropout_prob,
    }
    model = CausalVQAModel(model_config)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        logger.warning(f"Checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    offline_cache = {}
    use_do_eval = bool(args.use_causal and getattr(args, 'use_do_controller', 1) and getattr(args, 'use_offline_causal', 1))
    if use_do_eval:
        cache_eval = (getattr(args, "causal_cache_path_test", "") or getattr(args, "causal_cache_path", "") or "").strip()
        if not cache_eval:
            raise ValueError("DO test mode requires --causal_cache_path_test or --causal_cache_path")
        offline_cache = _load_offline_causal_cache(cache_eval)
        logger.info(f"DO test mode ON | cache={cache_eval} | items={len(offline_cache)}")

    # Causal gate for test (与 Val 一致，Train=Val=Test 结构)
    use_causal_gate = bool(args.use_causal and getattr(args, 'use_causal_gate_in_test', 0) and (not use_do_eval))
    intervention_bank = None
    if use_causal_gate and getattr(args, 'intervention_path', '') and os.path.exists(args.intervention_path):
        try:
            intervention_bank = {}
            with open(args.intervention_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        q = item.get('original_question')
                        if q:
                            kept_invs = [inv['text'] for inv in item.get('interventions', []) if inv.get('kept', True)]
                            if kept_invs:
                                intervention_bank[q] = kept_invs
                                intervention_bank[q.strip().lower()] = kept_invs
                    except: continue
            logger.info(f"Loaded intervention bank: {len(intervention_bank)} questions")
        except Exception as e:
            logger.warning(f"Intervention bank load failed: {e}")
    config = build_causal_eval_config(args)
    if config:
        logger.info(
            "Causal eval (请与训练 main 参数一致): "
            f"use_feature_gate={config['use_feature_gate']} "
            f"gate_alpha={config['gate_alpha']:.3f} gate_beta={config['gate_beta']:.3f} "
            f"hcss_topk={config['hcss_topk_ratio']:.2f} v_causal_topk={config['v_causal_topk_ratio']:.2f} "
            f"ccs_mask_ratio={config['ccs_mask_ratio']:.3f} hcss_ie_scale={config.get('hcss_ie_scale', 1.5):.3f} "
            f"ccs_text_de_scale={config['ccs_text_de_scale']:.2f} "
            f"max_inv_val_test={config['causal_max_interventions_val_test']}"
        )
    if args.use_causal:
        logger.info(
            "Model repr/CEM (须与训练一致): "
            f"repr_gate_alpha={getattr(args, 'repr_gate_alpha', 0.2):.3f} repr_gate_beta={getattr(args, 'repr_gate_beta', 0.2):.3f} "
            f"cem_gamma={getattr(args, 'cem_gamma', 1.0):.3f} use_logits_cem={int(getattr(args, 'use_logits_cem', 1))}"
        )
    tokenizer = None
    if use_causal_gate and hasattr(data_loader, 'test_dataset') and data_loader.test_dataset is not None:
        tokenizer = getattr(data_loader.test_dataset, 'tokenizer', None)

    logger.info(f"Model mode: {'Causal' if args.use_causal else 'Baseline'} | Causal gate in test: {use_causal_gate}")
    logger.info("Start evaluation...")
    total_correct = 0
    total_samples = 0
    criterion = F.binary_cross_entropy_with_logits
    
    # Initialize prediction results list
    predictions_list = []
    
    # Initialize inference performance statistics
    total_inference_time = 0.0
    total_samples_processed = 0
    batch_times = []
    sample_times = []
    
    # Per-category counters: Modality, Plane, Organ, Abnormality; SLAKE/VQA-RAD: open, closed
    category_names = ['modality', 'plane', 'organ', 'abnormality', 'open', 'closed']
    cat_correct = {c: 0.0 for c in category_names}
    cat_total = {c: 0 for c in category_names}

    with torch.no_grad():
        total_correct = 0.0
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            batch_start_time = time.time()

            images = batch['images'].to(device)
            questions = batch['questions']['input_ids'].to(device)
            attention_mask = batch['questions']['attention_mask'].to(device)
            do_questions = batch['do_questions']['input_ids'].to(device)
            do_attention_mask = batch['do_questions']['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            pattern_embedding = batch.get('pattern_embedding', None)
            entity_embedding = batch.get('entity_embedding', None)
            ae_images = batch.get('ae_images', None)
            maml_images = batch.get('maml_images', None)

            if pattern_embedding is not None:
                pattern_embedding = pattern_embedding.to(device)
            if entity_embedding is not None:
                entity_embedding = entity_embedding.to(device)
            if ae_images is not None:
                ae_images = ae_images.to(device)
            if maml_images is not None:
                maml_images = maml_images.to(device)

            answer_types = batch['answer_types']
            question_texts = batch['question_texts']
            answer_texts = batch['answer_texts']
            batch_size = images.size(0)

            inference_start_time = time.time()

            fwd_gate = {}
            # C-lite：无干预则无 CEM 统计；与 train 一致请开 use_causal_gate_in_test
            if use_causal_gate and tokenizer is not None and config.get("use_feature_gate", False):
                try:
                    from models.causal_modules import HCSSComputer, CCSComputer
                    from pipeline.causal_masks_intervention import compute_causal_masks_from_interventions
                    hcss_computer = HCSSComputer()
                    ccs_computer = CCSComputer(text_de_scale=config.get("ccs_text_de_scale", 15.0))
                    fwd_kwargs = dict(do_questions_ids=do_questions, do_attention_mask=do_attention_mask,
                        ae_images=ae_images, maml_images=maml_images,
                        pattern_embedding=pattern_embedding, entity_embedding=entity_embedding,
                        epoch=1, causal_start_epoch=1, training=False)
                    use_amp_test = bool(torch.cuda.is_available())
                    use_moe_te = bool(int(config.get("use_moe_router", 0))) and getattr(model, "router_trunk", None) is not None
                    router_intra_t = None
                    cm_t = None
                    if use_moe_te:
                        sub_ix, router_intra_t, _, _ = moe_probe_router_batch(
                            model, images, questions, attention_mask, fwd_kwargs, config, use_amp_test
                        )
                        cm_t = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
                        cm_t[torch.tensor(sub_ix, device=images.device, dtype=torch.long)] = True
                    else:
                        sub_ix = list(range(batch_size))
                        cm_t = torch.ones(batch_size, dtype=torch.bool, device=images.device)
                    sub_sz = len(sub_ix)
                    fwd_kw_sub = {}
                    for kk, vv in fwd_kwargs.items():
                        if isinstance(vv, torch.Tensor) and vv.dim() > 0 and vv.size(0) == batch_size:
                            fwd_kw_sub[kk] = vv[sub_ix]
                        else:
                            fwd_kw_sub[kk] = vv
                    _test_cm_kw = filter_kwargs_for_causal_masks(
                        compute_causal_masks_from_interventions,
                        {
                            "pure_encoder": None,
                            "device": str(device),
                            "seq_len": questions.size(1),
                            "num_visual_patches": 576,
                            "hcss_topk_ratio": config.get("hcss_topk_ratio", 0.4),
                            "v_causal_topk_ratio": config.get("v_causal_topk_ratio", 0.4),
                            "causal_mask_causal_parts": config.get("causal_mask_causal_parts", True),
                            "min_quality_interventions": config.get("min_quality_interventions", 1),
                            "min_entity_overlap": config.get("min_entity_overlap", 0.2),
                            "sim_low": config.get("sim_low", 0.45),
                            "sim_high": config.get("sim_high", 0.90),
                            "sim_low_strong": config.get("sim_low_strong", 0.25),
                            "overlap_min_strong": config.get("overlap_min_strong", 0.02),
                            "relax_sim_low": config.get("relax_sim_low", 0.50),
                            "relax_min_entity_overlap": config.get("relax_min_entity_overlap", 0.15),
                            "allow_last_resort_interventions": config.get("allow_last_resort_interventions", False),
                            "max_interventions": config.get("causal_max_interventions_val_test", config.get("causal_max_interventions", 3)),
                            "question_texts": [question_texts[j] for j in sub_ix] if question_texts else None,
                            "image_paths": [batch.get("image_paths")[j] for j in sub_ix] if batch.get("image_paths") is not None else None,
                            "targets": targets[sub_ix],
                            "answer_types": [answer_types[j] for j in sub_ix] if answer_types else None,
                            "fwd_kwargs": fwd_kw_sub,
                            "ccs_mask_ratio": config.get("ccs_mask_ratio", 0.4),
                            "ccs_topk_local": config.get("ccs_topk_local", 5),
                            "ccs_tau": config.get("ccs_tau", 0.01),
                            "local_ie_alpha": config.get("local_ie_alpha", 1.0),
                            "ccs_use_local_ie": config.get("ccs_use_local_ie", True),
                            "ccs_target": config.get("ccs_target", 0.2),
                            "ccs_penalty_lambda": config.get("ccs_penalty_lambda", 0.08),
                            "use_feature_gate": bool(config.get("use_feature_gate", True)),
                            "gate_alpha": config.get("gate_alpha", 1.0),
                            "gate_beta": config.get("gate_beta", 0.8),
                            "ablation_no_hcss": config.get("ablation_no_hcss", False),
                            "ablation_no_ccs": config.get("ablation_no_ccs", False),
                            "hcss_ie_scale": float(config.get("hcss_ie_scale", 1.5)),
                        },
                    )
                    q_causal, v_causal, causal_stats, _ = compute_causal_masks_from_interventions(
                        model, images[sub_ix], questions[sub_ix], attention_mask[sub_ix], tokenizer, intervention_bank,
                        model, hcss_computer, ccs_computer,
                        **_test_cm_kw,
                    )
                    v_gate_t = causal_stats.get("v_gate")
                    hcss_per = causal_stats.get("hcss_per_sample", [])
                    ccs_per = causal_stats.get("ccs_per_sample", [])
                    visual_ie_per = causal_stats.get("visual_ie_per_sample", [])
                    text_ie_per = causal_stats.get("text_ie_per_sample", [])
                    text_de_per = causal_stats.get("text_de_per_sample", [])
                    text_hcss_mask = causal_stats.get("text_hcss_mask")
                    if sub_sz < batch_size:
                        q_full = torch.ones(batch_size, questions.size(1), device=images.device, dtype=torch.float32)
                        v_full = torch.ones(batch_size, 577, device=images.device, dtype=torch.float32)
                        for idx, j in enumerate(sub_ix):
                            q_full[j] = q_causal[idx]
                            v_full[j] = v_causal[idx]
                        q_causal, v_causal = q_full, v_full
                        if v_gate_t is not None:
                            vgf = torch.ones(batch_size, 577, device=images.device, dtype=torch.float32)
                            for idx, j in enumerate(sub_ix):
                                vgf[j] = v_gate_t[idx]
                            v_gate_t = vgf
                        if len(hcss_per) == len(sub_ix) and len(ccs_per) == len(sub_ix):
                            hf, cf = [0.5] * batch_size, [0.5] * batch_size
                            for idx, j in enumerate(sub_ix):
                                hf[j], cf[j] = float(hcss_per[idx]), float(ccs_per[idx])
                            hcss_per, ccs_per = hf, cf
                        if len(visual_ie_per) == len(sub_ix) and len(text_de_per) == len(sub_ix):
                            vi_f = [0.5] * batch_size
                            td_f = [0.5] * batch_size
                            ti_f = [0.5] * batch_size
                            tie_s = text_ie_per if len(text_ie_per) == len(sub_ix) else None
                            for idx, j in enumerate(sub_ix):
                                vi_f[j] = float(visual_ie_per[idx])
                                td_f[j] = float(text_de_per[idx])
                                ti_f[j] = float(tie_s[idx]) if tie_s is not None else 0.5
                            visual_ie_per, text_de_per, text_ie_per = vi_f, td_f, ti_f
                        if text_hcss_mask is not None and text_hcss_mask.size(0) == sub_sz:
                            sl = text_hcss_mask.size(1)
                            mf = torch.ones(batch_size, sl, device=images.device, dtype=text_hcss_mask.dtype)
                            for idx, j in enumerate(sub_ix):
                                mf[j] = text_hcss_mask[idx]
                            text_hcss_mask = mf
                        cp_raw = causal_stats.get("ccs_patches")
                        if cp_raw is not None and cp_raw.size(0) == sub_sz:
                            pdim = cp_raw.size(1)
                            cf = torch.zeros(batch_size, pdim, device=images.device, dtype=cp_raw.dtype)
                            for idx, j in enumerate(sub_ix):
                                cf[j] = cp_raw[idx]
                            causal_stats["ccs_patches"] = cf
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
                        p50_vis = max(_p(visual_ie_per, 50), 0.50)
                        p50_hcss = max(_p(hcss_per, 50), 0.14)
                        p50_ccs = _p(ccs_per, 50)
                        p25_hcss = _p(hcss_per, 25)
                        p75_text = max(_p(text_ie_per, 75), 0.25)
                        fwd_gate = {"hcss_per_sample": hcss_per, "ccs_per_sample": ccs_per,
                                   "visual_ie_per_sample": visual_ie_per, "text_ie_per_sample": text_ie_per,
                                   "p50_vis": p50_vis, "p50_hcss": p50_hcss, "p50_ccs": p50_ccs,
                                   "p25_hcss": p25_hcss, "p75_text": p75_text, "neutral_mask": [False] * batch_size}
                    elif use_modal:
                        fwd_gate = {"hcss_per_sample": hcss_per, "ccs_per_sample": ccs_per,
                                   "p50_ccs": _p(ccs_per, 50), "p25_hcss": _p(hcss_per, 25)}
                    else:
                        v_gate_val = causal_stats.get("v_gate")
                        fwd_gate = {"v_gate": v_gate_val} if (config.get("use_feature_gate", True) and v_gate_val is not None) else {}
                    if text_hcss_mask is not None and text_hcss_mask.size(0) == batch_size:
                        fwd_gate["text_hcss_mask"] = text_hcss_mask
                    if ccs_patches_tensor is not None and ccs_patches_tensor.size(0) == batch_size:
                        fwd_gate["ccs_patches"] = ccs_patches_tensor
                    # 与训练一致：传入 q_mask_pre/v_mask
                    use_fg = config.get("use_feature_gate", True)
                    fwd_gate["q_mask_pre"] = q_causal
                    fwd_gate["v_mask"] = torch.ones(batch_size, 577, device=images.device, dtype=torch.float32) if use_fg else v_causal
                    if cm_t is not None:
                        fwd_gate["causal_path_mask"] = cm_t.float()
                        fwd_gate["cem_path_scale"] = 1.0
                        if router_intra_t is not None:
                            fwd_gate["router_intra_gate"] = router_intra_t
                except Exception as e:
                    logger.warning(f"Test causal gate failed: {e}")

            fwd = dict(do_questions_ids=do_questions, do_attention_mask=do_attention_mask,
                      ae_images=ae_images, maml_images=maml_images,
                      pattern_embedding=pattern_embedding, entity_embedding=entity_embedding,
                      epoch=1, causal_start_epoch=1, training=False,
                      cem_gt_indices=targets.argmax(dim=1).long(),
                      **fwd_gate)
            if use_do_eval:
                cached_signals = _get_cached_signals_for_batch(batch, offline_cache, device)
                fwd.update(causal_signals=cached_signals, apply_do=True)
            out = model(images, questions, attention_mask, **fwd)
            if isinstance(out, tuple) and len(out) >= 2:
                logits = out[0]
                if getattr(model, 'use_open_embedding_matching', False):
                    open_logits = out[1]
                elif getattr(model, 'use_open_concept_head', False):
                    open_logits = None
                else:
                    open_logits = out[1]
            else:
                logits, open_logits = out, None
            use_open_emb = (open_logits is not None and getattr(model, 'use_open_embedding_matching', False))
            if use_open_emb:
                open_mask = torch.tensor([((answer_types[i] if i < len(answer_types) else "") or "").strip().lower() == 'open' for i in range(batch_size)], device=logits.device, dtype=torch.bool)
                score_logits = logits.clone()
                if open_mask.any():
                    score_logits[open_mask] = open_logits[open_mask]
            else:
                score_logits = logits

            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            batch_size = images.size(0)
            total_inference_time += inference_time
            total_samples_processed += batch_size
            batch_times.append(batch_time)
            sample_times.extend([inference_time / batch_size] * batch_size)

            loss = criterion(score_logits, targets)
            total_loss += loss.item()
            batch_scores = compute_score_with_logits(score_logits, targets)

            for i, (score, ans_type) in enumerate(zip(batch_scores, answer_types)):
                s = score.sum().item()
                total_samples += 1
                total_correct += s

                cat_key = ans_type.lower().strip()
                if cat_key in cat_correct:
                    cat_correct[cat_key] += s
                    cat_total[cat_key] += 1

            if (batch_idx + 1) % 10 == 0:
                avg_batch_time = np.mean(batch_times[-10:])
                throughput = batch_size / avg_batch_time
                logger.info(f"Batch {batch_idx + 1}: Avg batch time {avg_batch_time:.4f}s, "
                            f"Throughput {throughput:.2f} samples/s")

    # ========== Results ==========
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    logger.info("=" * 60)
    logger.info("  Evaluation Results")
    logger.info("=" * 60)
    header = f"{'Category':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"
    logger.info(header)
    logger.info("-" * 45)

    # 只打印有样本的类别：SLAKE/VQA-RAD 显示 Open/Closed，MedVQA2019 显示 Modality/Plane/Organ/Abnormality
    for cat in category_names:
        if cat_total[cat] > 0:
            acc = cat_correct[cat] / cat_total[cat] * 100
            logger.info(f"{cat.capitalize():<15} {cat_correct[cat]:>8.1f} {cat_total[cat]:>8d} {acc:>9.2f}%")

    logger.info("-" * 45)
    logger.info(f"{'All':<15} {total_correct:>8.1f} {total_samples:>8d} {overall_accuracy * 100:>9.2f}%")
    logger.info("=" * 60)
    logger.info(f"Average loss: {total_loss / max(1, total_samples):.4f}")

    # Inference performance
    avg_inference_time_per_sample = total_inference_time / total_samples_processed if total_samples_processed > 0 else 0
    throughput_samples_per_sec = total_samples_processed / total_inference_time if total_inference_time > 0 else 0

    logger.info(f"Inference: {avg_inference_time_per_sample:.4f}s/sample | {throughput_samples_per_sec:.1f} samples/s")

    # Save results to JSON
    results = {
        'overall_accuracy': overall_accuracy * 100,
        'per_category': {cat: {
            'correct': cat_correct[cat],
            'total': cat_total[cat],
            'accuracy': cat_correct[cat] / cat_total[cat] * 100 if cat_total[cat] > 0 else 0
        } for cat in category_names},
        'total_samples': total_samples,
    }
    results_path = os.path.join(os.path.dirname(args.checkpoint), 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return overall_accuracy


if __name__ == "__main__":
    args = parse_args()
    test_accuracy(args)