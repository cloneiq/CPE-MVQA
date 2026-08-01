#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时干预生成模块
用于在训练过程中实时生成文本干预和图像干预
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import os
import json
import re
import logging

from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
_empty_diag_log_count = [0]  # list to allow mutation in nested scope
from sklearn.metrics.pairwise import cosine_similarity
try:
    from models.do_question import MedicalQuestionPatternAndEntityExtractor
except Exception:
    MedicalQuestionPatternAndEntityExtractor = None


_STOPWORDS = {
    "what", "which", "is", "are", "was", "were", "do", "does", "did", "the", "a", "an",
    "in", "on", "at", "of", "for", "to", "and", "or", "with", "this", "that", "these",
    "those", "there", "here", "image", "scan", "picture", "show", "shown", "display"
}

# Curated replacement pools for high-quality interventions.
# These are intentionally conservative and type-consistent.
_CURATED_POOLS = {
    "modality": [
        "ct", "mri", "x-ray", "ultrasound", "pet", "ct scan", "mr imaging"
    ],
    "plane": [
        "axial", "coronal", "sagittal", "axial plane", "coronal plane", "sagittal plane"
    ],
    "organ": [
        "liver", "kidney", "spleen", "lung", "heart", "brain", "pancreas", "stomach", "colon"
    ],
    "abnormality": [
        "effusion", "lesion", "mass", "fracture", "nodule", "tumor", "edema", "pneumothorax"
    ],
}

# core_value 为「问题类型词」时（extractor fallback），不做 replacement，改用 paraphrase
# 这些词描述「在问什么」，不是答案候选
_TYPE_FALLBACK_WORDS = frozenset({
    "modality", "plane", "organ", "anatomy", "pathology", "disease", "symptom",
    "treatment", "body part", "organ function", "tissue", "general", "organ system",
    "prevention",
})

# 干预层级：strong=核心实体替换(IE+CCS), medium=masking(DE), weak=paraphrase/synonym(稳定性)
INTV_STRONG = "strong"   # 改变视觉指向，用于 IE/CCS
INTV_MEDIUM = "medium"   # 局部信息删除，用于 DE
INTV_WEAK = "weak"       # 语义保持，不参与 IE

# 按 answer_type 的 mask 词（避免统一用 "unknown structure"）
_MASK_TOKENS_BY_TYPE = {
    "modality": "unknown modality",
    "plane": "unknown plane",
    "organ": "unknown anatomy",
    "abnormality": "unknown finding",
    "open": "unknown finding",
    "closed": "unknown finding",
}

# 语义等价改写库：当 core_value 为 type 词时使用，保证干预与原问题问同一件事
# 来源：scripts/extract_paraphrase_templates.py 从 train_typed.jsonl 提取，已排除含答案槽位的句式
_PARAPHRASE_TEMPLATES = {
    "modality": [
        "what imaging modality was used to take this image",
        "what type of imaging modality is shown",
        "what modality is used to take this image",
        "what type of imaging is this",
        "which imaging modality is shown",
    ],
    "plane": [
        "in what plane is this image taken",
        "which plane is this image taken",
        "in what plane was this image taken",
        "which plane is the image taken",
        "what plane is demonstrated",
        "which plane is the image shown in",
        "what plane is this",
        "which plane is this image in",
        "what plane is seen",
        "what image plane is this",
        "what is the plane",
        "what imaging plane is depicted here",
        "what plane is the image acquired in",
        "what is the plane of the image",
        "in what plane is this image oriented",
    ],
    "organ": [
        "what organ system is shown in the image",
        "what part of the body is being imaged here",
        "what organ system is primarily present in this image",
        "which organ system is imaged",
        "what organ system is being imaged",
        "what part of the body is being imaged",
        "what organ system is pictured here",
        "what organ system is imaged",
        "what organ system is visualized",
        "what is one organ system seen in this image",
        "what organ system is evaluated primarily",
        "what organ is this image of",
        "what is the organ system in this image",
    ],
    "abnormality": [
        "what abnormality is seen in the image",
        "what is the primary abnormality in this image",
        "does this image look normal",
        "is there something wrong in the image",
        "what abnormality is shown",
        "which finding is present",
        "what pathology is visible",
    ],
    # SLAKE + VQA-RAD: 从 scripts/extract_paraphrase_templates.py 提取合并
    "open": [
        "which part of the body does this image belong to?",
        "what modality is used to take this image?",
        "这张图片的成像方式是什么?",
        "图像里包含的区域属于身体哪个部分?",
        "what is the largest organ in the picture?",
        "what diseases are included in the picture?",
        "图片中包含哪些疾病?",
        "where is/are the abnormality located?",
        "异常病变在哪个位置?",
        "这个图像的扫描平面是什么?",
        "what is the scanning plane of this image?",
        "what is the mr weighting in this image?",
        "what is the main organ in the image?",
        "图片中最大的器官是什么?",
        "图片中体积最大的器官是什么?",
        "where is the lesion located?",
        "where is the mass located?",
        "what type of image is this?",
        "where is the abnormality?",
        "where is the mass?",
        "what is the location of the mass?",
        "where is the lesion?",
        "where is the abnormality in this image?",
        "in what plane was this image taken?",
        "what is the plane?",
        "what imaging modality was used?",
        "what organ system is shown in the above image?",
        "where is the pathology located?",
        "what is abnormal in this image?",
        "how was this image taken?",
    ],
    "closed": [
        "does the picture contain liver?",
        "图片中包含肝脏吗?",
        "does the picture contain spleen?",
        "does the picture contain kidney?",
        "does the picture contain heart?",
        "图片中包含肺吗?",
        "does the picture contain lung?",
        "图片中包含肾脏吗?",
        "图片中包含脾脏吗?",
        "图片中包含心脏吗?",
        "is the lung healthy?",
        "肺是异常的吗?",
        "are there abnormalities in this image?",
        "which organ is abnormal, heart or lung?",
        "肝脏是健康的吗?",
        "is the heart enlarged?",
        "is the liver normal?",
        "is the trachea midline?",
        "is this a normal image?",
        "is there a pneumothorax present?",
        "is this a pa film?",
        "is there cardiomegaly?",
        "is a pleural effusion present?",
        "is this image abnormal?",
        "is there a pneumothorax?",
        "are there rib fractures present?",
        "is this image normal?",
        "is the skull fractured?",
        "is the mass calcified?",
        "is there mass effect?",
    ],
    "general": [
        "what is shown in this image",
        "what does this image show",
        "what can be seen in this picture",
        "what is visible in the image",
    ],
}

_EXTRACTOR = None
_EXTRACTOR_READY = False


def _get_medical_extractor():
    global _EXTRACTOR, _EXTRACTOR_READY
    if _EXTRACTOR_READY:
        return _EXTRACTOR
    _EXTRACTOR_READY = True
    if MedicalQuestionPatternAndEntityExtractor is None:
        return None
    try:
        _EXTRACTOR = MedicalQuestionPatternAndEntityExtractor()
    except Exception:
        _EXTRACTOR = None
    return _EXTRACTOR


def _map_entity_type_to_answer_type(entity_type: str, text: str = "") -> str:
    t = (entity_type or "").lower()
    q = (text or "").lower()
    if t == "modality":
        return "modality"
    if t == "anatomy":
        # 'plane' is often phrased with anatomy-like words; keep heuristic.
        if any(k in q for k in ["axial", "coronal", "sagittal", "plane", "view"]):
            return "plane"
        return "organ"
    if t in {"pathology", "symptom"}:
        return "abnormality"
    return ""


def _infer_answer_type_from_question(question: str) -> str:
    q = (question or "").lower()
    extractor = _get_medical_extractor()
    if extractor is not None and q:
        try:
            res = extractor.extract_pattern(q)
            core = res.get("core_entity", {}) if isinstance(res, dict) else {}
            mapped = _map_entity_type_to_answer_type(core.get("type", ""), q)
            if mapped:
                return mapped
        except Exception:
            pass
    if any(k in q for k in ["modality", "ct", "mri", "x-ray", "xray", "ultrasound", "us", "pet"]):
        return "modality"
    if any(k in q for k in ["plane", "axial", "coronal", "sagittal", "view"]):
        return "plane"
    if any(k in q for k in ["organ", "liver", "lung", "heart", "kidney", "brain", "spleen", "pancreas"]):
        return "organ"
    if any(k in q for k in ["abnormal", "lesion", "mass", "tumor", "fracture", "effusion", "edema", "disease"]):
        return "abnormality"
    return ""


def _extract_entities_weak(text: str) -> set:
    text_l = (text or "").lower()
    entities = set()
    extractor = _get_medical_extractor()
    if extractor is not None and text_l:
        try:
            res = extractor.extract_pattern(text_l)
            core = res.get("core_entity", {}) if isinstance(res, dict) else {}
            core_val = (core.get("value", "") or "").strip().lower()
            if core_val:
                entities.update([t for t in core_val.split() if t and t not in _STOPWORDS])
        except Exception:
            pass
    tokens = re.findall(r"[a-z]+", text_l)
    entities.update({t for t in tokens if len(t) >= 3 and t not in _STOPWORDS})
    return entities


def _entity_overlap_ratio(q1: str, q2: str) -> float:
    e1, e2 = _extract_entities_weak(q1), _extract_entities_weak(q2)
    # If either side has no usable entities, do not hard-fail by overlap.
    # Fall back to semantic similarity filtering.
    if len(e1) == 0 or len(e2) == 0:
        return 1.0
    inter = len(e1.intersection(e2))
    denom = max(1, min(len(e1), len(e2)))
    return inter / denom


def _is_answer_slot_style(cand: str) -> bool:
    """判断是否为「答案槽位」形式：is this a X / does this image show X（与 paraphrase 结构不同，能产生 HCSS）"""
    cand_l = (cand or "").strip().lower()
    if not cand_l or len(cand_l.split()) < 4:
        return False
    return cand_l.startswith("is this ") or cand_l.startswith("does this ")


def load_pure_encoder_for_interventions(bert_path: str, device: str = "cuda"):
    """
    加载纯BERT编码器用于文本干预生成（语义相似度筛选）
    
    Args:
        bert_path: BERT模型路径
        device: 设备
    
    Returns:
        pure_encoder: 纯BERT编码器
        tokenizer: 分词器
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        # 只加载编码器部分
        pure_encoder = AutoModel.from_pretrained(bert_path)
        pure_encoder.eval()
        pure_encoder.to(device)
        return pure_encoder, tokenizer
    except Exception as e:
        print(f"⚠️  加载纯BERT编码器失败：{e}")
        return None, None


def get_text_embedding(text: str, tokenizer, pure_encoder, device: str = "cuda") -> np.ndarray:
    """
    获取文本的embedding（用于语义相似度计算）
    
    Args:
        text: 文本
        tokenizer: 分词器
        pure_encoder: 纯BERT编码器
        device: 设备
    
    Returns:
        embedding: 文本embedding
    """
    if pure_encoder is None or tokenizer is None:
        return None
    
    with torch.no_grad():
        inputs = tokenizer(
            text, padding='max_length', truncation=True, max_length=32,
            return_tensors='pt'
        ).to(device)
        outputs = pure_encoder(**inputs)
        # 使用[CLS] token的embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding


def _get_embedding_from_model(text: str, model, tokenizer, device: str = "cuda") -> Optional[np.ndarray]:
    """
    当 pure_encoder 不可用时，用主模型的 language_encoder 获取 embedding（快速语义过滤）
    """
    if model is None or tokenizer is None or not hasattr(model, 'language_encoder'):
        return None
    try:
        dev = next(model.parameters()).device if hasattr(model, 'parameters') else device
        with torch.no_grad():
            enc = tokenizer(
                text, padding='max_length', truncation=True, max_length=32,
                return_tensors='pt'
            )
            inp = {k: v.to(dev) for k, v in enc.items()}
            out = model.language_encoder(**inp)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        return emb
    except Exception:
        return None


def generate_text_interventions_realtime(
    question: str,
    model=None,
    tokenizer=None,
    pure_encoder=None,
    device: str = "cuda",
    max_interventions: int = 8,
    min_similarity: float = 0.02,
    use_simple_method: bool = True
) -> List[Tuple[str, str]]:
    """
    实时生成分层文本干预：strong(实体替换) / medium(masking) / weak(paraphrase/synonym)
    Returns: List[Tuple[text, level]]
    """
    interventions = _generate_structured_interventions(question, max_interventions=max_interventions)

    if len(interventions) < max_interventions:
        remain = max_interventions - len(interventions)
        fallback = _generate_simple_interventions(question, remain)
        seen_texts = {t for t, _ in interventions}
        for fb in fallback:
            if fb and fb.strip().lower() not in seen_texts:
                interventions.append((fb.strip().lower(), INTV_WEAK))
                seen_texts.add(fb.strip().lower())
                if len(interventions) >= max_interventions:
                    break
        interventions = list(dict.fromkeys([(t, l) for t, l in interventions if t]))[:max_interventions]

    if pure_encoder is not None and tokenizer is not None and len(interventions) > 0:
        text_to_level = {t: l for t, l in interventions}
        texts = [t for t, _ in interventions]
        filtered_texts = _filter_by_similarity(question, texts, tokenizer, pure_encoder, device, min_similarity)
        interventions = [(t, text_to_level.get(t, INTV_WEAK)) for t in filtered_texts if t in text_to_level][:max_interventions]

    return interventions[:max_interventions]


def _get_entity_pool_by_answer_type(answer_type: str, extractor) -> List[str]:
    answer_type = (answer_type or "").lower()
    curated = _CURATED_POOLS.get(answer_type, [])
    if extractor is None:
        return curated
    entities = getattr(extractor, "medical_entities", {})
    if answer_type == "modality":
        base = entities.get("modality", [])
        return list(dict.fromkeys([e.lower() for e in curated + base]))
    if answer_type == "plane":
        base = [e.lower() for e in entities.get("modality", []) if "plane" in e.lower()]
        return list(dict.fromkeys(curated + base + ["transverse plane"]))
    if answer_type == "organ":
        return list(dict.fromkeys(curated + [e.lower() for e in entities.get("anatomy", [])]))
    if answer_type == "abnormality":
        return list(dict.fromkeys(curated + [e.lower() for e in entities.get("pathology", [])]))
    # unknown: mixed but still medical
    merged = []
    for k in ["anatomy", "modality", "pathology"]:
        merged.extend([e.lower() for e in entities.get(k, [])])
    return list(dict.fromkeys(merged))


def _replace_entity_in_question(question: str, old_entity: str, new_entity: str) -> str:
    q = (question or "").strip().lower()
    old_entity = (old_entity or "").strip().lower()
    new_entity = (new_entity or "").strip().lower()
    if not q or not new_entity:
        return q
    if old_entity and re.search(r"\b" + re.escape(old_entity) + r"\b", q):
        return re.sub(r"\b" + re.escape(old_entity) + r"\b", new_entity, q, count=1)
    return q


def _generate_structured_interventions(question: str, max_interventions: int = 8) -> List[Tuple[str, str]]:
    """
    分层干预：strong(实体替换)→medium(masking)→weak(paraphrase)
    strong: 改变视觉指向，用于 IE/CCS
    medium: 局部删除，用于 DE
    weak: 语义保持，不参与 IE
    """
    q = (question or "").strip().lower()
    if not q:
        return []

    extractor = _get_medical_extractor()
    if extractor is None:
        return [(t, INTV_WEAK) for t in _generate_simple_interventions(q, max_interventions)]

    try:
        parsed = extractor.extract_pattern(q)
    except Exception:
        return [(t, INTV_WEAK) for t in _generate_simple_interventions(q, max_interventions)]

    syntax_pattern = (parsed.get("syntax_pattern", "") if isinstance(parsed, dict) else "") or ""
    core_entity = (parsed.get("core_entity", {}) if isinstance(parsed, dict) else {}) or {}
    core_type = (core_entity.get("type", "") or "").lower()
    core_value = (core_entity.get("value", "") or "").strip().lower()
    answer_type = _map_entity_type_to_answer_type(core_type, q) or _infer_answer_type_from_question(q)

    pool_full = _get_entity_pool_by_answer_type(answer_type, extractor)
    pool = [p for p in pool_full if p != core_value] if core_value else pool_full

    core_value_is_answer_slot = core_entity.get("is_answer_slot")
    if core_value_is_answer_slot is None:
        core_value_is_answer_slot = (
            core_value in pool_full
            and core_value not in _TYPE_FALLBACK_WORDS
        )

    n_replace = max(2, int(max_interventions * 0.6))  # 优先 strong
    n_mask = max(1, int(max_interventions * 0.2))   # medium
    n_recomp = max(0, max_interventions - n_replace - n_mask)

    results: List[Tuple[str, str]] = []
    seen = {q}

    if core_value_is_answer_slot:
        # === strong: 核心实体替换 (CT→MRI, liver→kidney) ===
        for cand in pool:
            if len(results) >= n_replace:
                break
            new_q = _replace_entity_in_question(q, core_value, cand)
            if new_q and new_q not in seen and new_q != q:
                results.append((new_q, INTV_STRONG))
                seen.add(new_q)

        # === medium: masking ===
        mask_token = _MASK_TOKENS_BY_TYPE.get(answer_type, "unknown finding")
        if n_mask > 0 and len(results) < max_interventions:
            masked = _replace_entity_in_question(q, core_value, mask_token)
            if masked and masked not in seen and masked != q:
                results.append((masked, INTV_MEDIUM))
                seen.add(masked)

        # === weak: recomposition ===
        if n_recomp > 0 and len(results) < max_interventions:
            pattern_prefix = syntax_pattern.strip().lower()
            if pattern_prefix and len(pool) > 0:
                for cand in pool[: max(1, n_recomp * 2)]:
                    rec_q = f"{pattern_prefix} {cand}".strip()
                    if rec_q and rec_q not in seen and rec_q != q and len(rec_q.split()) >= 4:
                        results.append((rec_q, INTV_WEAK))
                        seen.add(rec_q)
                    if len(results) >= max_interventions:
                        break
    else:
        # === strong: 答案槽位形式 ===
        n_slot = min(5, max(2, max_interventions // 2))
        if answer_type in ("modality", "plane", "organ", "abnormality") and len(pool) > 0:
            for cand in pool[:n_slot]:
                if len(results) >= max_interventions:
                    break
                if answer_type == "modality":
                    slot_q = f"is this a {cand}"
                elif answer_type == "plane":
                    slot_q = f"is this {cand}" if "plane" in cand else f"is this in the {cand} plane"
                elif answer_type == "abnormality":
                    slot_q = f"does this image show {cand}"
                else:
                    slot_q = f"is this image of the {cand}"
                slot_q = slot_q.strip().lower()
                if slot_q and slot_q not in seen and slot_q != q and len(slot_q.split()) >= 3:
                    results.append((slot_q, INTV_STRONG))
                    seen.add(slot_q)

        # === weak: paraphrase 模板 ===
        templates = _PARAPHRASE_TEMPLATES.get(answer_type, [])
        for tpl in templates:
            if len(results) >= max_interventions:
                break
            tpl_lower = tpl.strip().lower()
            if tpl_lower and tpl_lower not in seen and tpl_lower != q:
                results.append((tpl_lower, INTV_WEAK))
                seen.add(tpl_lower)

        # === medium: masking ===
        mask_token = _MASK_TOKENS_BY_TYPE.get(answer_type, "unknown finding")
        if n_mask > 0 and len(results) < max_interventions and core_value:
            masked = _replace_entity_in_question(q, core_value, mask_token)
            if masked and masked not in seen and masked != q:
                results.append((masked, INTV_MEDIUM))
                seen.add(masked)

    if len(results) < 2:
        fallback = _generate_simple_interventions(q, max_interventions - len(results))
        for fb in fallback:
            if fb not in seen and fb != q:
                results.append((fb, INTV_WEAK))
                seen.add(fb)
                if len(results) >= max_interventions:
                    break

    return results[:max_interventions]


def _is_safe_synonym_replacement(words: List[str], i: int, word: str, synonym: str) -> bool:
    """检查替换是否会产生语法错误，避免如 'what modality are this' """
    if word == "is" and synonym == "are":
        next_w = words[i + 1].lower() if i + 1 < len(words) else ""
        if next_w in ("this", "that"):
            return False
    if word == "this" and synonym == "the":
        if i == len(words) - 1:
            return False
    if word == "is" and synonym in ("does", "do"):
        if i > 0 and words[i - 1].lower() in ("modality", "plane", "organ", "type"):
            return False
    return True


def _generate_simple_interventions(question: str, max_interventions: int = 8) -> List[str]:
    """
    简单方法：基于规则的词替换生成干预（语法安全）
    """
    interventions = []
    question_lower = question.lower()
    words = question_lower.split()

    synonym_map = {
        "what": ["which", "what type of", "what kind of"],
        "which": ["what", "which type of"],
        "image": ["picture", "scan"],
        "picture": ["image", "scan"],
        "show": ["display", "depict"],
        "contain": ["include", "have"],
        "type": ["kind", "category"],
        "normal": ["healthy", "typical"],
        "abnormal": ["unusual", "atypical"],
        "large": ["big", "enlarged"],
        "small": ["tiny", "reduced"],
        "clear": ["visible", "evident"],
        "visible": ["clear", "evident"],
        "this": ["that"],
        "is": ["are"],
    }

    for i, word in enumerate(words):
        if word not in synonym_map:
            continue
        for synonym in synonym_map[word][:2]:
            if not _is_safe_synonym_replacement(words, i, word, synonym):
                continue
            new_words = words.copy()
            new_words[i] = synonym
            intervention = " ".join(new_words)
            if intervention != question_lower and intervention not in interventions:
                interventions.append(intervention)
                if len(interventions) >= max_interventions:
                    return interventions[:max_interventions]

    if len(interventions) < max_interventions and "the" not in question_lower and len(words) > 3:
        cand = f"the {question_lower}"
        if cand not in interventions:
            interventions.append(cand)

    return interventions[:max_interventions]


def _filter_by_similarity(
    original_question: str,
    candidate_interventions: List[str],
    tokenizer,
    pure_encoder,
    device: str = "cuda",
    min_similarity: float = 0.02
) -> List[str]:
    """
    使用语义相似度筛选干预
    
    Args:
        original_question: 原始问题
        candidate_interventions: 候选干预列表
        tokenizer: 分词器
        pure_encoder: 纯BERT编码器
        device: 设备
        min_similarity: 最小相似度
    
    Returns:
        filtered_interventions: 筛选后的干预列表
    """
    if not candidate_interventions:
        return []
    
    # 获取原始问题的embedding
    orig_emb = get_text_embedding(original_question, tokenizer, pure_encoder, device)
    if orig_emb is None:
        return candidate_interventions
    
    # 计算所有候选干预的embedding和相似度
    similarities = []
    for intervention in candidate_interventions:
        intv_emb = get_text_embedding(intervention, tokenizer, pure_encoder, device)
        if intv_emb is not None:
            sim = cosine_similarity(orig_emb, intv_emb)[0][0]
            similarities.append((intervention, sim))
    
    # 按相似度排序并筛选
    similarities.sort(key=lambda x: x[1], reverse=True)
    filtered = [intv for intv, sim in similarities if sim >= min_similarity]
    
    return filtered if filtered else candidate_interventions[:len(candidate_interventions)//2]


def generate_visual_interventions_realtime(
    model,
    image: torch.Tensor,
    question: str,
    tokenizer,
    device: str = "cuda",
    mask_ratio: float = 0.9,
    mask_mode: str = "zero",
    use_gradcam: bool = True
) -> torch.Tensor:
    """
    实时生成图像干预（mask区域选择）
    
    Args:
        model: MUMC_VQA模型
        image: 图像张量 [1, 3, H, W]
        question: 问题文本
        tokenizer: 分词器
        device: 设备
        mask_ratio: mask比例
        mask_mode: mask模式（"zero"或"noise"）
        use_gradcam: 是否使用Grad-CAM选择mask区域
    
    Returns:
        masked_image_embeds: mask后的图像embeddings
    """
    from interventions.gradcam_intervention import generate_masked_embeddings
    
    if use_gradcam:
        # 使用Grad-CAM选择重要区域进行mask
        # 兼容新版 vqa_module
        vision_encoder = getattr(model, 'vision_encoder', None)
        if vision_encoder is not None:
            masked_image_embeds = generate_masked_embeddings(
                vision_encoder,
                image,
                mask_ratio=mask_ratio,
                mask_mode=mask_mode
            )
        else:
             print("Warning: Vision encoder not found in model, using random mask.")
             masked_image_embeds = generate_masked_embeddings(
                None, # Will trigger fallback if handled, or fail. Assuming helper handles None or we pass dummy
                image,
                mask_ratio=mask_ratio,
                mask_mode=mask_mode
             )
    else:
        # 简单方法：随机mask
        # 注意: generate_masked_embeddings 的第一个参数如果是None，可能需要适配
        # 这里假设它能接受 model 或 vision_encoder
        vision_encoder = getattr(model, 'vision_encoder', None)
        masked_image_embeds = generate_masked_embeddings(
            vision_encoder,
            image,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode
        )
    
    return masked_image_embeds


def _lexical_fallback_interventions_when_short(
    question_norm: str,
    max_interventions: int,
    min_tokens: int,
    max_tokens: int,
) -> List[Tuple[str, str]]:
    """
    When JSONL intervention_bank is missing or this sample has no bank hits,
    structured/realtime generation can still yield an empty list. These light
    rewrites keep token overlap with the original so CCS/HCSS pipeline can run.
    """
    q = (question_norm or "").strip().lower()
    out: List[Tuple[str, str]] = []
    if not q or max_interventions <= 0:
        return out
    words = q.split()
    seeds: List[str] = []
    if len(words) >= 2:
        seeds.append(" ".join(words[:-1]))
    if len(words) >= 3:
        seeds.append(" ".join(words[1:]))
    seeds.append(f"regarding this scan, {q}")
    seeds.append(f"{q} (radiology context)")
    if q.startswith("in this image"):
        seeds.append(f"for this study: {q}")
    else:
        seeds.append(f"in this image, {q}")
    if len(words) >= 4:
        mid = len(words) // 2
        seeds.append(" ".join(words[:mid] + words[mid + 1:]))

    seen = {q}
    for s in seeds:
        s = (s or "").strip().lower()
        if not s or s in seen:
            continue
        tn = len(s.split())
        if tn < min_tokens or tn > max_tokens:
            continue
        out.append((s, INTV_WEAK))
        seen.add(s)
        if len(out) >= max_interventions:
            break
    return out


def get_interventions_for_sample_realtime(
    image_name: str,
    question: str,
    intervention_bank: Optional[Dict[str, List[str]]],
    answer_type: Optional[str] = None,
    model=None,
    tokenizer=None,
    pure_encoder=None,
    device: str = "cuda",
    max_interventions: int = 8,
    use_realtime_fallback: bool = True,
    min_quality_interventions: int = 2,
    min_entity_overlap: float = 0.4,
    sim_low: float = 0.55,
    sim_high: float = 0.90,
    sim_low_strong: float = 0.25,
    overlap_min_strong: float = 0.02,
    min_tokens: int = 1,
    max_tokens: int = 24,
    relax_on_empty: bool = True,
    relax_min_entity_overlap: float = 0.15,
    relax_sim_low: float = 0.50,
    relax_sim_high: float = 0.95,
    allow_last_resort_interventions: bool = False,
    return_metadata: bool = False,
) -> Any:
    """
    获取样本的干预列表（混合策略：预计算+实时生成）
    
    Args:
        image_name: 图像名称
        question: 问题文本
        intervention_bank: 预计算的干预库（可选）
        model: MUMC_VQA模型（用于实时生成）
        tokenizer: 分词器（用于实时生成）
        pure_encoder: 纯BERT编码器（用于实时生成）
        device: 设备
        max_interventions: 最大干预数
        use_realtime_fallback: 是否在干预不足时实时生成
    
    Returns:
        interventions: 干预列表
    """
    interventions_tagged: List[Tuple[str, str]] = []
    question_norm = question.strip().lower() if isinstance(question, str) else ""
    had_bank_hit = False

    # 1. 优先从干预库获取（预计算视为 strong）
    if intervention_bank is not None:
        raw = []
        if question in intervention_bank:
            raw = intervention_bank[question][:max_interventions]
        elif question_norm in intervention_bank:
            raw = intervention_bank[question_norm][:max_interventions]
        elif image_name in intervention_bank:
            raw = intervention_bank[image_name][:max_interventions]
        else:
            image_basename = os.path.basename(image_name)
            if image_basename in intervention_bank:
                raw = intervention_bank[image_basename][:max_interventions]
            else:
                image_base = os.path.splitext(image_basename)[0]
                if image_base in intervention_bank:
                    raw = intervention_bank[image_base][:max_interventions]
        if raw:
            had_bank_hit = True
        interventions_tagged = [(q.strip().lower(), INTV_STRONG) for q in raw if q]

    # 2. 干预不足时实时生成（带层级）
    if len(interventions_tagged) < 2 and use_realtime_fallback and question is not None:
        try:
            realtime = generate_text_interventions_realtime(
                question, model, tokenizer, pure_encoder, device,
                max_interventions=max_interventions - len(interventions_tagged),
                use_simple_method=False
            )
            seen = {t for t, _ in interventions_tagged}
            for t, lvl in realtime:
                if t and t not in seen:
                    interventions_tagged.append((t, lvl))
                    seen.add(t)
            interventions_tagged = list(dict.fromkeys(interventions_tagged))[:max_interventions]
        except Exception:
            pass

    # 2b. 无库 / 当前样本未命中库 / 实时生成仍不足：规则模板补候选（可不提供 intervention_path）
    need_more = max(2, int(min_quality_interventions or 1))
    if len(interventions_tagged) < need_more:
        extra = _lexical_fallback_interventions_when_short(
            question_norm,
            max_interventions=max(0, max_interventions - len(interventions_tagged)),
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        seen = {t for t, _ in interventions_tagged}
        for t, lvl in extra:
            if t and t not in seen:
                interventions_tagged.append((t, lvl))
                seen.add(t)
        interventions_tagged = list(dict.fromkeys(interventions_tagged))[:max_interventions]

    # 3. 基础质量控制：去重 + 长度约束
    deduped: List[Tuple[str, str]] = []
    seen = set()
    for qn, lvl in interventions_tagged:
        qn = (qn or "").strip().lower()
        if not qn or qn in seen:
            continue
        tok_num = len(qn.split())
        if tok_num < min_tokens or tok_num > max_tokens:
            continue
        deduped.append((qn, lvl))
        seen.add(qn)
    interventions_tagged = deduped[:max_interventions]

    diag_reason = ""
    diag_n_tagged = len(interventions_tagged)
    diag_n_passed = 0
    diag_best_sim = None
    diag_first_overlap = None
    diag_first_type_ok = None

    # 4. 分层过滤：strong 放宽 sim_low/overlap，weak 严格
    orig_type = (answer_type or "").strip().lower()
    if not orig_type:
        orig_type = _infer_answer_type_from_question(question_norm)

    eff_sim_low = sim_low
    eff_min_entity_overlap = min_entity_overlap
    eff_relax_sim_low = relax_sim_low
    eff_relax_min_entity_overlap = relax_min_entity_overlap
    if orig_type == "abnormality":
        eff_sim_low = min(sim_low, 0.48)
        eff_min_entity_overlap = min(min_entity_overlap, 0.18)
        eff_relax_sim_low = min(relax_sim_low, 0.42)
        eff_relax_min_entity_overlap = min(relax_min_entity_overlap, 0.15)

    if pure_encoder is not None:
        eff_sim_low = max(0.32, eff_sim_low * 0.88)
        eff_relax_sim_low = max(0.30, eff_relax_sim_low * 0.88)

    eff_sim_high = min(sim_high, 0.85)
    eff_relax_sim_high = min(relax_sim_high, 0.90)
    # 未使用预计算 JSONL 命中时，原问与轻改写 embedding 常 >0.85；放宽上界避免「无库则全空」
    if not had_bank_hit:
        eff_sim_high = min(0.985, eff_sim_high + 0.10)
        eff_relax_sim_high = min(0.99, eff_relax_sim_high + 0.04)

    def _get_thresholds_for_level(level: str):
        if level == INTV_STRONG:
            return max(0.20, sim_low_strong), overlap_min_strong
        if level == INTV_MEDIUM:
            return max(0.28, eff_sim_low * 0.7), max(0.05, eff_min_entity_overlap * 0.5)
        return eff_sim_low, eff_min_entity_overlap

    orig_emb = None
    if pure_encoder is not None and tokenizer is not None and question_norm:
        orig_emb = get_text_embedding(question_norm, tokenizer, pure_encoder, device)
    elif model is not None and tokenizer is not None and question_norm:
        orig_emb = _get_embedding_from_model(question_norm, model, tokenizer, device)

    high_quality: List[Tuple[str, str]] = []
    quality_scores = []
    for cand, level in interventions_tagged:
        cand_type = _infer_answer_type_from_question(cand)
        type_ok = (orig_type == "") or (cand_type == "") or (cand_type == orig_type)
        if not type_ok:
            continue

        overlap = _entity_overlap_ratio(question_norm, cand)
        is_slot = _is_answer_slot_style(cand)
        sim_lo, overlap_lo = _get_thresholds_for_level(level)
        overlap_min = 0.02 if is_slot else overlap_lo
        if overlap < overlap_min:
            continue

        sim_ok = True
        sim_val = 0.7
        if orig_emb is not None and tokenizer is not None:
            cand_emb = None
            if pure_encoder is not None:
                cand_emb = get_text_embedding(cand, tokenizer, pure_encoder, device)
            elif model is not None:
                cand_emb = _get_embedding_from_model(cand, model, tokenizer, device)
            if cand_emb is not None:
                sim_val = float(cosine_similarity(orig_emb, cand_emb)[0][0])
                sim_ok = (sim_val >= sim_lo) and (sim_val <= eff_sim_high)
        if not sim_ok:
            continue

        high_quality.append((cand, level))
        quality_scores.append({"q": cand, "overlap": overlap, "sim": sim_val, "level": level})
        if len(high_quality) >= max_interventions:
            break

    no_causal_update = len(high_quality) < min_quality_interventions
    result_list = high_quality[:max_interventions]
    diag_n_passed = len(high_quality)
    if diag_n_tagged == 0:
        diag_reason = "no_candidates"
    elif no_causal_update:
        diag_reason = "quality_filtered"

    # 5. Relaxed thresholds retry
    if no_causal_update and relax_on_empty and len(interventions_tagged) > 0:
        relaxed: List[Tuple[str, str]] = []
        relaxed_scores = []
        for cand, level in interventions_tagged:
            cand_type = _infer_answer_type_from_question(cand)
            type_ok = (orig_type == "") or (cand_type == "") or (cand_type == orig_type)
            if not type_ok:
                continue
            overlap = _entity_overlap_ratio(question_norm, cand)
            is_slot = _is_answer_slot_style(cand)
            relax_overlap_min = 0.02 if is_slot else max(0.02, eff_relax_min_entity_overlap * 0.5)
            if overlap < relax_overlap_min:
                continue
            sim_ok = True
            sim_val = 0.5
            if orig_emb is not None and tokenizer is not None:
                cand_emb = None
                if pure_encoder is not None:
                    cand_emb = get_text_embedding(cand, tokenizer, pure_encoder, device)
                elif model is not None:
                    cand_emb = _get_embedding_from_model(cand, model, tokenizer, device)
                if cand_emb is not None:
                    sim_val = float(cosine_similarity(orig_emb, cand_emb)[0][0])
                    sim_ok = (sim_val >= eff_relax_sim_low) and (sim_val <= eff_relax_sim_high)
            if not sim_ok:
                continue
            relaxed.append((cand, level))
            relaxed_scores.append({"q": cand, "overlap": overlap, "sim": sim_val, "level": level})
            if len(relaxed) >= max_interventions:
                break
        if len(relaxed) >= min_quality_interventions:
            result_list = relaxed[:max_interventions]
            quality_scores = relaxed_scores[:max_interventions]
            no_causal_update = False
        elif no_causal_update:
            diag_reason = "relax_failed"

    # 6. Last-resort bypass
    if allow_last_resort_interventions and no_causal_update and len(interventions_tagged) > 0:
        result_list = interventions_tagged[:max_interventions]
        no_causal_update = len(result_list) < min_quality_interventions
        if no_causal_update:
            diag_reason = "last_resort_failed"

    if no_causal_update and min_quality_interventions <= 1 and len(interventions_tagged) > 0:
        best_effort = []
        diag_first_cand_overlap, diag_first_cand_sim, diag_first_cand_type_ok = None, None, None
        for cand, level in interventions_tagged:
            overlap = _entity_overlap_ratio(question_norm, cand)
            sim_val = 0.5
            if tokenizer is not None:
                cand_emb = get_text_embedding(cand, tokenizer, pure_encoder, device) if pure_encoder else _get_embedding_from_model(cand, model, tokenizer, device) if model else None
                if cand_emb is not None:
                    sim_val = float(cosine_similarity(orig_emb, cand_emb)[0][0])
            best_effort.append((cand, level, overlap, sim_val))
        if best_effort:
            best_effort.sort(key=lambda x: (x[3], x[2]), reverse=True)
            best_cand, best_lvl, best_overlap, best_sim = best_effort[0]
            best_effort_sim_threshold = 0.20  # 0.25 仍多 best_effort_failed；0.20 医学 paraphrase 可恢复更多
            if best_sim >= best_effort_sim_threshold:
                result_list = [(best_cand, best_lvl)]
                quality_scores = [{"q": best_cand, "overlap": best_overlap, "sim": best_sim, "level": best_lvl}]
                no_causal_update = False
            elif no_causal_update:
                diag_reason = "best_effort_failed"
                diag_best_sim = best_sim  # 用于诊断
        elif no_causal_update:
            diag_reason = "best_effort_failed"
            diag_best_sim = None  # best_effort 为空，无 sim
            diag_first_overlap = diag_first_cand_overlap
            diag_first_type_ok = diag_first_cand_type_ok

    # empty 样本诊断日志已关闭，如需调试可改为 logger.debug
    # if (no_causal_update or not result_list) and diag_reason:
    #     _empty_diag_log_count[0] += 1
    #     ...

    if return_metadata:
        out = {
            "interventions": result_list,
            "no_causal_update": no_causal_update,
            "orig_type": orig_type,
            "quality_count": len(result_list),
            "quality_scores": quality_scores[:max_interventions],
        }
        if no_causal_update or not result_list:
            out["diag_empty_reason"] = diag_reason or "unknown"
            out["diag_n_tagged"] = diag_n_tagged
            out["diag_n_passed"] = diag_n_passed
            if diag_best_sim is not None:
                out["diag_best_sim"] = diag_best_sim
        return out
    return [t for t, _ in result_list]

