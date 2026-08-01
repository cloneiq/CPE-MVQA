#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA-RAD Concept 设计（5 类，Stage 2 Open-Only 强化）
====================================================
- structural_disease: 脑、胸、腹部结构异常（脑肿瘤、hydrocephalus 等）
- bone_lesion: 骨骼相关病变（fracture, dysplasia 等）
- organ_chest: 胸腔/器官异常（心脏扩大、肺病变等）
- functional_metabolic: 功能性、代谢性异常（edema, hepatomegaly 等）
- misc: 长尾、罕见或未归类

Mapping 规则：
- Closed (yes/no): 从 question 映射
- Open answer: 优先 answer 映射，失败则 question fallback
- Misc 目标: 15-25%
"""
import re
from typing import Optional, Tuple

# ============ 5 类 Concept（最终分类表） ============
VQA_RAD_CONCEPTS = [
    "structural_disease",  # 脑、胸、腹部结构异常
    "bone_lesion",        # 骨骼相关病变
    "organ_chest",        # 胸腔/器官异常
    "functional_metabolic",  # 功能性、代谢性异常
    "misc",               # 长尾、未归类
]
CONCEPT_TO_IDX = {c: i for i, c in enumerate(VQA_RAD_CONCEPTS)}
IDX_TO_CONCEPT = {i: c for i, c in enumerate(VQA_RAD_CONCEPTS)}
NUM_CONCEPTS = len(VQA_RAD_CONCEPTS)
MISC_IDX = CONCEPT_TO_IDX["misc"]


def _norm(s: str) -> str:
    if not s:
        return ""
    return " ".join(re.sub(r"\s+", " ", str(s).lower().strip()).split())


# ============ Question -> Concept（Closed yes/no） ============
QUESTION_TO_CONCEPT = [
    # structural_disease
    (["brain tumor", "brain tumour", "intracranial mass", "brain lesion", "hydrocephalus",
      "ventriculomegaly", "hemorrhage", "haemorrhage", "bleeding", "hematoma",
      "infarct", "infarction", "stroke", "anoxic brain", "atrophy", "atrophic",
      "edema", "oedema", "swelling"], "structural_disease"),
    # bone_lesion
    (["fracture", "fractured", "broken", "rib fracture", "skull fracture", "ribs broken",
      "bone lesion", "bony lesion", "bone abnormality", "dysplasia",
      "degenerative", "degeneration", "arthritis"], "bone_lesion"),
    # organ_chest
    (["cardiomegaly", "heart too big", "heart enlarged", "heart large",
      "effusion", "pleural fluid", "fluid", "air fluid level", "pneumothorax", "free air", "air underneath",
      "pneumonia", "pneumonic", "lung infection", "atelectasis", "atelectatic", "lung collapse"], "organ_chest"),
    # functional_metabolic
    (["fatty", "fatty infiltration", "fat infiltration", "hepatomegaly",
      "inflammatory", "inflammation", "stranding", "fat stranding",
      "infection", "abscess", "infected"], "functional_metabolic"),
    # misc (generic)
    (["obstructed", "obstruction", "bowel obstruct",
      "mass", "tumor", "tumour", "nodule", "lesion",
      "abnormal", "abnormality", "wrong", "abnormality present",
      "disease", "diseases included"], "misc"),
]

# ============ Answer -> Concept（Open answer 优先） ============
ANSWER_TO_CONCEPT = [
    # structural_disease
    (["brain tumor", "tumor", "mass", "lesion", "glioma", "meningioma", "hemorrhage", "hematoma", "bleeding",
      "infarct", "infarction", "stroke", "hydrocephalus", "atrophy", "atrophic", "edema", "oedema"], "structural_disease"),
    # bone_lesion
    (["fracture", "fractured", "dysplasia"], "bone_lesion"),
    # organ_chest
    (["cardiomegaly", "heart enlarged", "effusion", "pleural", "fluid", "pericholecystic fluid",
      "pneumonia", "pneumonic", "atelectasis", "pneumothorax"], "organ_chest"),
    # functional_metabolic
    (["fatty", "fatty infiltration", "hepatomegaly", "inflammatory", "stranding",
      "infection", "abscess"], "functional_metabolic"),
    # misc
    (["fat", "cyst", "cancer", "lung cancer", "none"], "misc"),
]


def _map_question_to_concept(question: str) -> Optional[str]:
    qn = _norm(question)
    for keywords, concept in QUESTION_TO_CONCEPT:
        if any(kw in qn for kw in keywords):
            return concept
    return None


def _map_answer_to_concept(answer: str) -> Optional[str]:
    an = _norm(answer)
    for keywords, concept in ANSWER_TO_CONCEPT:
        if any(kw in an for kw in keywords):
            return concept
    return None


def get_concept(
    question: str,
    answer: str,
    answer_type: str,
) -> Tuple[str, int]:
    """
    - Closed: question → concept
    - Open: answer 优先，失败则 question fallback，再失败则 misc
    """
    at = _norm(answer_type)
    ans_norm = _norm(answer)
    is_closed = at == "closed" or (at != "open" and ans_norm in ("yes", "no"))

    if is_closed:
        c = _map_question_to_concept(question)
    else:
        c = _map_answer_to_concept(answer)
        if c is None:
            c = _map_question_to_concept(question)

    if c is None:
        c = "misc"

    idx = CONCEPT_TO_IDX.get(c, MISC_IDX)
    return c, idx
