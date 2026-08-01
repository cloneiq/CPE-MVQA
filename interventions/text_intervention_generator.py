# -*- coding: utf-8 -*-
"""
Text Intervention Generator – 最终稳定版（解决缓存卡住+JSON写入问题）
核心修复：JSON强制刷盘、Pickle分块缓存、4090批量加速
"""
import os
import re
import json
import pickle
import random
import math
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import scispacy

# ------------------------ 路径 & 超参配置（4090优化） ------------------------
BIOBERT_MODEL = "/root/autodl-tmp/MUMC-main/MUMC-main/biobert-base-cased-v1.1"
DATA_PATH = [
    "/root/autodl-tmp/MUMC-main/MUMC-main/data_PathVQA/train_typed.jsonl",
    "/root/autodl-tmp/MUMC-main/MUMC-main/data_PathVQA/valid_typed.jsonl",
    "/root/autodl-tmp/MUMC-main/MUMC-main/data_PathVQA/test_typed.jsonl",
]
WORD_SYNONYM_PATH = "/root/autodl-tmp/MUMC-main/MUMC-main/interventions/prepare/outputs/medical_synonyms_biobert_clean.json"
ENHANCED_TEMPLATE_PATH = "/root/autodl-tmp/MUMC-main/MUMC-main/interventions/prepare/outputs/question_template_dict_enhanced.json"

OUTPUT_DIR = "outputs_v9_stable_4090"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "intervention_v9_med.jsonl")
REPORT_JSON = os.path.join(OUTPUT_DIR, "intervention_v9_medreport.json")

# 4090适配参数（平衡速度与稳定性）
BATCH_SIZE = 256
EMBED_BATCH_SIZE = 2048
N_REPLACE = 2
Y_RECON = 2
M_MASK = 2

# 超参配置
MASK_PROB = 0.3
SIM_THRESHOLDS = {
    "replacement": 0.70,
    "reconstruction": 0.60,
    "mask": 0.65
}
DIVERSITY_THRESHOLD = 0.60
MAX_RETRIES = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------ 模型 & 工具初始化 ------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
nlp_sci = spacy.load("en_core_sci_sm", disable=["parser", "lemmatizer"])

print("[Info] 加载BioBERT模型（4090优化）...")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL, use_fast=True)
bio_model = AutoModel.from_pretrained(BIOBERT_MODEL).to(DEVICE).eval().half()
if hasattr(torch, "compile") and DEVICE.type == "cuda":
    bio_model = torch.compile(bio_model, mode="max-autotune")

# ------------------------ 批量NLP处理工具 ------------------------
def batch_process_texts(texts, nlp_model, batch_size=1024):
    docs = []
    for doc in tqdm(
        nlp_model.pipe(texts, batch_size=batch_size, disable=["parser"]),
        total=len(texts),
        desc=f"批量NLP处理（{nlp_model.meta['name']}）"
    ):
        docs.append(doc)
    return docs

# ------------------------ 预加载资源 ------------------------
RECON_CACHE = "recon_templates_with_type.json"
EMBEDDING_CACHE = "embedding_cache_4090_stable.pkl"  # 改为pkl格式

def load_type_templates():
    if os.path.exists(RECON_CACHE):
        with open(RECON_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    print("[Info] 加载带类型的模板簇...")
    with open(ENHANCED_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        enhanced_templates = json.load(f)
    type_to_templates = defaultdict(list)
    for cluster in enhanced_templates.values():
        typ = cluster["type"]
        type_to_templates[typ].extend(cluster["templates"])
    for typ in type_to_templates:
        unique_tpls = list(set(type_to_templates[typ]))
        filtered = [t.lower() for t in unique_tpls if len(nlp(t)) >=3 and any(tok.pos_=="VERB" for tok in nlp(t))]
        type_to_templates[typ] = filtered
    with open(RECON_CACHE, "w", encoding="utf-8") as f:
        json.dump(type_to_templates, f, ensure_ascii=False, indent=2)
    return type_to_templates

def load_medical_synonyms(path):
    print("[Info] 过滤医学同义词...")
    with open(path, "r", encoding="utf-8") as f:
        raw_synonyms = json.load(f)
    medical_synonyms = {}
    for term, syns in raw_synonyms.items():
        if len(nlp_sci(term).ents) == 0:
            continue
        filtered_syns = [syn.lower() for syn in syns if len(nlp_sci(syn).ents) > 0]
        if filtered_syns:
            medical_synonyms[term.lower()] = list(set(filtered_syns))
    return medical_synonyms

# ------------------------ 缓存加载/保存（核心修复：Pickle分块） ------------------------
embedding_cache = {}

def load_embedding_cache():
    global embedding_cache
    embedding_cache = {}
    # 加载主缓存文件
    if os.path.exists(EMBEDDING_CACHE):
        try:
            with open(EMBEDDING_CACHE, "rb") as f:
                embedding_cache = pickle.load(f)
            print(f"[Info] 加载主缓存（{len(embedding_cache)}条）")
        except Exception as e:
            print(f"[Warning] 主缓存加载失败：{str(e)}")
    
    # 加载分块缓存
    i = 0
    while os.path.exists(f"{EMBEDDING_CACHE}.part{i}"):
        try:
            with open(f"{EMBEDDING_CACHE}.part{i}", "rb") as f:
                chunk = pickle.load(f)
            embedding_cache.update(chunk)
            print(f"[Info] 加载分块{i}（总计{len(embedding_cache)}条）")
        except Exception as e:
            print(f"[Warning] 分块{i}加载失败：{str(e)}")
        i += 1
    
    # 加载临时缓存（如果有）
    if os.path.exists(f"{EMBEDDING_CACHE}.temp"):
        try:
            with open(f"{EMBEDDING_CACHE}.temp", "rb") as f:
                temp_cache = pickle.load(f)
            embedding_cache.update(temp_cache)
            print(f"[Info] 加载临时缓存（{len(temp_cache)}条）")
        except Exception as e:
            print(f"[Warning] 临时缓存加载失败：{str(e)}")
    
    return embedding_cache

def save_embedding_cache():
    global embedding_cache
    if not embedding_cache:
        print("[Info] 无缓存数据需保存")
        return
    
    # 清理旧缓存文件（避免冲突）
    cache_files = [EMBEDDING_CACHE]
    i = 0
    while os.path.exists(f"{EMBEDDING_CACHE}.part{i}"):
        cache_files.append(f"{EMBEDDING_CACHE}.part{i}")
        i += 1
    cache_files.append(f"{EMBEDDING_CACHE}.temp")
    for f in cache_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"[Warning] 删除旧缓存{os.path.basename(f)}失败：{str(e)}")
    
    # 分块保存（减小分块大小，提升速度）
    chunk_size = 10000  # 核心修复：从10万改为1万
    total = len(embedding_cache)
    items = list(embedding_cache.items())
    chunks = [dict(items[i:i+chunk_size]) for i in range(0, total, chunk_size)]
    
    try:
        # 分块保存
        for i, chunk in enumerate(tqdm(chunks, desc="保存缓存分块")):
            chunk_path = f"{EMBEDDING_CACHE}.part{i}"
            with open(chunk_path, "wb") as f:
                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)  # 高效序列化
        
        # 若分块数<=1，合并为主文件（方便后续加载）
        if len(chunks) == 1:
            os.rename(f"{EMBEDDING_CACHE}.part0", EMBEDDING_CACHE)
            print(f"[Info] 缓存保存完成（单文件，{total}条）")
        else:
            print(f"[Info] 缓存保存完成（{len(chunks)}个分块，{total}条）")
    
    except Exception as e:
        # 保存失败时，备份为临时文件
        temp_path = f"{EMBEDDING_CACHE}.temp"
        with open(temp_path, "wb") as f:
            pickle.dump(embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Error] 分块保存失败：{str(e)}")
        print(f"[Info] 缓存已备份至临时文件：{os.path.basename(temp_path)}")

# ------------------------ 批量embedding计算（4090优化） ------------------------
@torch.no_grad()
def embed_batch_final(texts, batch_size=EMBED_BATCH_SIZE):
    global embedding_cache
    new_texts = [t for t in texts if t not in embedding_cache]
    if not new_texts:
        return np.array([embedding_cache[t] for t in texts])
    
    print(f"[Debug] 4090批量计算嵌入：{len(new_texts)}条，批次{batch_size}")
    enc = tokenizer(new_texts, return_tensors="pt", truncation=True, padding=True, max_length=64)
    input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
    attn_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)
    embs = []
    
    for i in range(0, len(new_texts), batch_size):
        batch_ids = input_ids[i:i+batch_size]
        batch_mask = attn_mask[i:i+batch_size]
        with torch.cuda.amp.autocast():
            out = bio_model(batch_ids, attention_mask=batch_mask).last_hidden_state.mean(dim=1)
        embs.append(out.half().cpu())
    
    new_embs = torch.cat(embs).numpy()
    for t, emb in zip(new_texts, new_embs):
        embedding_cache[t] = emb
    # 嵌入计算后不立即保存大缓存，仅在最终统一保存
    return np.array([embedding_cache[t] for t in texts])

# ------------------------ 批量干预生成函数 ------------------------
def get_question_type(q):
    q_lower = q.lower()
    QUESTION_TYPES = {
        "existence": r"is|are|was|were|exist|present",
        "location": r"where|location|site|position",
        "nature": r"what|type|kind|character|nature",
        "treatment": r"treat|treatment|manage|management",
        "cause": r"why|cause|reason",
        "severity": r"how severe|grade|stage|severity"
    }
    for typ, pattern in QUESTION_TYPES.items():
        if re.search(pattern, q_lower):
            return typ
    return "other"

def batch_replacement_intervention(texts, word_synonyms, docs, sci_docs, prob_replace=0.8):
    results = []
    for text, doc, sci_doc in zip(texts, docs, sci_docs):
        tokens = [tok.text for tok in doc]
        medical_ents = {ent.start: ent.end for ent in sci_doc.ents}
        verbs = {tok.i for tok in doc if tok.pos_ == "VERB"}
        core_indices = set()
        for start, end in medical_ents.items():
            core_indices.update(range(start, end))
        core_indices.update(verbs)
        
        new_tokens, rep_pos, rep_pairs = [], [], []
        replaced = False
        for idx, tok in enumerate(tokens):
            if idx in core_indices:
                new_tokens.append(tok)
                continue
            key = tok.lower().strip(".,?")
            if (key in word_synonyms and random.random() < prob_replace) or not replaced:
                cand = random.choice(word_synonyms[key]) if key in word_synonyms else f"{tok}_alt"
                new_tokens.append(cand)
                rep_pos.append(idx)
                rep_pairs.append((tok, cand))
                replaced = True
            else:
                new_tokens.append(tok)
        if not replaced and len(tokens) > 1:
            idx = random.choice([i for i in range(len(tokens)) if i not in core_indices])
            new_tokens[idx] = f"{tokens[idx]}_alt"
            rep_pos.append(idx)
            rep_pairs.append((tokens[idx], new_tokens[idx]))
        results.append((" ".join(new_tokens), rep_pos, rep_pairs))
    return results

def batch_reconstruction_intervention(texts, type_to_templates, docs, sci_docs):
    results = []
    for text, doc, sci_doc in zip(texts, docs, sci_docs):
        q_type = get_question_type(text)
        candidate_tpls = type_to_templates.get(q_type, [])
        if len(candidate_tpls) < 5:
            candidate_tpls += type_to_templates.get("other", [])
        if not candidate_tpls:
            candidate_tpls = [
                f"what is the {text.split()[-1]}?",
                f"describe the {text.split()[-1]}.",
                f"is the {text.split()[-1]} present?"
            ]
        
        numbers = [tok.text for tok in doc if tok.like_num]
        medical_ents = [ent.text.lower() for ent in sci_doc.ents] or [text.split()[-1].lower()]
        template = random.choice(candidate_tpls)
        
        if "num" in template and numbers:
            template = re.sub(r"\bnum\b", random.choice(numbers), template, flags=re.IGNORECASE)
        if medical_ents:
            tpl_ents = [ent.text.lower() for ent in nlp_sci(template).ents]
            if not tpl_ents:
                tpl_nouns = [tok.text.lower() for tok in nlp(template) if tok.pos_ == "NOUN"]
                if tpl_nouns:
                    tpl_ent = random.choice(tpl_nouns)
                    replace_ent = random.choice(medical_ents)
                    template = re.sub(rf"\b{tpl_ent}\b", replace_ent, template, flags=re.IGNORECASE)
            else:
                tpl_ent = random.choice(tpl_ents)
                replace_ent = random.choice(medical_ents)
                template = re.sub(rf"\b{re.escape(tpl_ent)}\b", replace_ent, template, flags=re.IGNORECASE)
        
        results.append((template, (template, q_type)))
    return results

def batch_masking_intervention(texts, docs, sci_docs, mask_prob=0.4):
    results = []
    for text, doc, sci_doc in zip(texts, docs, sci_docs):
        tokens_out, mask_pos = [], []
        core_ents = [ent for ent in sci_doc.ents]
        core_indices = set()
        for ent in core_ents:
            core_indices.update(range(ent.start, ent.end))
        
        min_mask = max(2, int(len(doc) * 0.2))
        for idx, tok in enumerate(doc):
            if idx in core_indices:
                tokens_out.append(tok.text)
                continue
            is_medical = any(ent.start <= idx < ent.end for ent in sci_doc.ents)
            if (is_medical or tok.pos_ in ["ADJ", "NOUN"]) and random.random() < mask_prob:
                tokens_out.append(tokenizer.unk_token)
                mask_pos.append(idx)
            else:
                tokens_out.append(tok.text)
        
        while len(mask_pos) < min_mask:
            non_core_indices = [i for i in range(len(doc)) if i not in core_indices and i not in mask_pos]
            if not non_core_indices:
                break
            idx = random.choice(non_core_indices)
            tokens_out[idx] = tokenizer.unk_token
            mask_pos.append(idx)
        
        cov = len(mask_pos) / max(1, len(doc))
        results.append((" ".join(tokens_out), cov, mask_pos))
    return results

# ------------------------ 批量多样性检查 ------------------------
def batch_is_diverse(new_texts, existing_texts_list, text_to_idx, all_embs, sample_size=3):
    if not new_texts:
        return [True] * len(new_texts)
    
    new_indices = [text_to_idx[t] for t in new_texts if t in text_to_idx]
    new_embs = all_embs[new_indices] if new_indices else np.array([])
    
    diverse_flags = []
    for i, new_text in enumerate(new_texts):
        existing_texts = existing_texts_list[i]
        if not existing_texts:
            diverse_flags.append(True)
            continue
        
        sample_texts = random.sample(existing_texts, min(sample_size, len(existing_texts)))
        sample_indices = [text_to_idx[t] for t in sample_texts if t in text_to_idx]
        if not sample_indices:
            diverse_flags.append(True)
            continue
        
        existing_embs = all_embs[sample_indices]
        sims = cosine_similarity([new_embs[i]], existing_embs).flatten() if len(new_embs) > i else [0.0]
        diverse_flags.append(np.max(sims) < DIVERSITY_THRESHOLD)
    
    return diverse_flags

# ------------------------ 批量生成干预（带重试） ------------------------
def batch_generate_interventions(texts, generator_func, args_list, text_to_idx, all_embs, max_retries=MAX_RETRIES):
    batch_size = len(texts)
    results = [None] * batch_size
    need_retry = list(range(batch_size))
    
    for retry in range(max_retries):
        if not need_retry:
            break
        
        current_texts = [texts[i] for i in need_retry]
        current_args = [args_list[i] for i in need_retry]
        
        if generator_func.__name__ == "batch_replacement_intervention":
            gen_results = generator_func(
                current_texts, 
                current_args[0]["word_synonyms"],
                [a["doc"] for a in current_args],
                [a["sci_doc"] for a in current_args]
            )
        elif generator_func.__name__ == "batch_reconstruction_intervention":
            gen_results = generator_func(
                current_texts,
                current_args[0]["type_to_templates"],
                [a["doc"] for a in current_args],
                [a["sci_doc"] for a in current_args]
            )
        elif generator_func.__name__ == "batch_masking_intervention":
            gen_results = generator_func(
                current_texts,
                [a["doc"] for a in current_args],
                [a["sci_doc"] for a in current_args]
            )
        
        new_texts = [res[0] for res in gen_results]
        existing_texts_list = [current_args[i]["seen_list"] for i in range(len(current_texts))]
        diverse_flags = batch_is_diverse(new_texts, existing_texts_list, text_to_idx, all_embs)
        
        new_need_retry = []
        for i, idx in enumerate(need_retry):
            cand = new_texts[i]
            if cand != texts[idx] and diverse_flags[i]:
                results[idx] = gen_results[i]
            else:
                new_need_retry.append(idx)
        
        need_retry = new_need_retry
        if retry % 3 == 0 and need_retry:
            print(f"[Debug] 批量重试第{retry+1}次，剩余{len(need_retry)}/{batch_size}样本")
    
    # 保底处理
    for idx in need_retry:
        text = texts[idx]
        base_text = text.rstrip('.?!')
        fallback_text = f"{base_text}_modified." if base_text else "modified_text"
        if generator_func.__name__ == "batch_replacement_intervention":
            results[idx] = (fallback_text, [], [])
        elif generator_func.__name__ == "batch_reconstruction_intervention":
            results[idx] = (fallback_text, (None, "other"))
        elif generator_func.__name__ == "batch_masking_intervention":
            results[idx] = (fallback_text, 0.0, [])
    
    return results

# ------------------------ 数据加载 ------------------------
def load_multiple_jsons(paths):
    all_data = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[Warning] 跳过不存在的文件：{p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        print(f"[Info] 从 {os.path.basename(p)} 加载 {len(data)} 条数据")
        all_data.extend(data)
    print(f"[Info] 总数据量：{len(all_data)}")
    return all_data

# ------------------------ 主函数（核心修复：JSON强制刷盘） ------------------------
def generate_interventions(dataset, word_synonyms, type_to_templates):
    print("[Debug] 阶段1：收集所有文本用于预计算嵌入...")
    all_texts = []
    questions = [item["question"] for item in dataset]
    total_questions = len(questions)
    print(f"[Info] 共{total_questions}条原问题，批量预处理NLP结果...")
    
    nlp_docs = batch_process_texts(questions, nlp, batch_size=2048)
    nlp_sci_docs = batch_process_texts(questions, nlp_sci, batch_size=2048)
    
    # 批量收集文本
    for i in tqdm(range(0, total_questions, BATCH_SIZE), desc="批量收集文本"):
        batch_end = min(i + BATCH_SIZE, total_questions)
        batch_questions = questions[i:batch_end]
        batch_docs = nlp_docs[i:batch_end]
        batch_sci_docs = nlp_sci_docs[i:batch_end]
        
        rep_results = batch_replacement_intervention(batch_questions, word_synonyms, batch_docs, batch_sci_docs)
        rec_results = batch_reconstruction_intervention(batch_questions, type_to_templates, batch_docs, batch_sci_docs)
        mask_results = batch_masking_intervention(batch_questions, batch_docs, batch_sci_docs)
        
        for q in batch_questions:
            all_texts.append(q)
        for rep in rep_results:
            all_texts.append(rep[0])
        for rec in rec_results:
            all_texts.append(rec[0])
        for mask in mask_results:
            all_texts.append(mask[0])
    
    # 过滤异常文本
    all_texts = [t for t in all_texts if len(t.strip()) > 0 and re.search(r"[a-zA-Z0-9]", t)]
    print(f"[Debug] 过滤后文本总量：{len(all_texts)}")
    
    # 阶段2：批量计算嵌入
    print("[Info] 阶段2：4090批量计算嵌入...")
    all_embs = embed_batch_final(all_texts, batch_size=EMBED_BATCH_SIZE)
    text_to_idx = {t: i for i, t in enumerate(all_texts)}
    print(f"[Info] 文本-索引映射建立完成（{len(text_to_idx)}条）")
    
    # 阶段3：批量生成干预结果（核心修复：JSON刷盘）
    print("[Info] 阶段3：批量生成干预结果（4090加速）...")
    stats = defaultdict(int)
    buffer = []
    output_file = open(OUTPUT_JSONL, "w", encoding="utf-8")  # 单独打开文件，方便控制刷盘
    
    try:
        for i in tqdm(range(0, total_questions, BATCH_SIZE), desc="批量生成干预"):
            batch_end = min(i + BATCH_SIZE, total_questions)
            batch_data = dataset[i:batch_end]
            batch_questions = questions[i:batch_end]
            batch_docs = nlp_docs[i:batch_end]
            batch_sci_docs = nlp_sci_docs[i:batch_end]
            batch_size = len(batch_questions)
            
            # 准备参数
            rep_args_list = [
                {"word_synonyms": word_synonyms, "doc": batch_docs[j], "sci_doc": batch_sci_docs[j], "seen_list": []}
                for j in range(batch_size)
            ]
            rec_args_list = [
                {"type_to_templates": type_to_templates, "doc": batch_docs[j], "sci_doc": batch_sci_docs[j], "seen_list": []}
                for j in range(batch_size)
            ]
            mask_args_list = [
                {"doc": batch_docs[j], "sci_doc": batch_sci_docs[j], "seen_list": []}
                for j in range(batch_size)
            ]
            
            # 分多次生成干预
            rep_results_list = []
            for _ in range(N_REPLACE):
                current_rep = batch_generate_interventions(
                    batch_questions, batch_replacement_intervention, rep_args_list, text_to_idx, all_embs
                )
                rep_results_list.append(current_rep)
                for j in range(batch_size):
                    rep_args_list[j]["seen_list"].append(current_rep[j][0])
            
            rec_results_list = []
            for _ in range(Y_RECON):
                current_rec = batch_generate_interventions(
                    batch_questions, batch_reconstruction_intervention, rec_args_list, text_to_idx, all_embs
                )
                rec_results_list.append(current_rec)
                for j in range(batch_size):
                    rec_args_list[j]["seen_list"].append(current_rec[j][0])
            
            mask_results_list = []
            for _ in range(M_MASK):
                current_mask = batch_generate_interventions(
                    batch_questions, batch_masking_intervention, mask_args_list, text_to_idx, all_embs
                )
                mask_results_list.append(current_mask)
                for j in range(batch_size):
                    mask_args_list[j]["seen_list"].append(current_mask[j][0])
            
            # 处理并添加到缓冲区
            for j in range(batch_size):
                item = batch_data[j]
                q = batch_questions[j]
                sample_id = item.get("id") or f"{item.get('image_name', '')}_{i+j}"
                
                base_idx = text_to_idx.get(q, -1)
                base_emb = all_embs[base_idx].tolist() if base_idx != -1 else []
                
                sample = {
                    "id": sample_id,
                    "split": item.get("split", ""),
                    "image_name": item.get("image_name", ""),
                    "original_question": q,
                    "answer": item.get("answer", ""),
                    "embedding_original": base_emb,
                    "interventions": [],
                    "meta": {"version": "v9_stable_4090", "batch_size": BATCH_SIZE}
                }
                
                # 替换干预
                for k in range(N_REPLACE):
                    rep_q, rep_pos, rep_pairs = rep_results_list[k][j]
                    rep_emb = all_embs[text_to_idx[rep_q]].tolist() if rep_q in text_to_idx else []
                    sim = float(cosine_similarity([base_emb], [rep_emb])[0][0]) if base_emb and rep_emb else 0.0
                    kept = sim >= SIM_THRESHOLDS["replacement"]
                    sample["interventions"].append({
                        "type": "replacement", "text": rep_q, "pairs": rep_pairs,
                        "sim": round(sim, 4), "kept": kept, "embedding": rep_emb if kept else None
                    })
                    stats["replacement_kept" if kept else "replacement_filtered"] += 1
                
                # 重构干预
                for k in range(Y_RECON):
                    rec_q, (tpl, q_type) = rec_results_list[k][j]
                    rec_emb = all_embs[text_to_idx[rec_q]].tolist() if rec_q in text_to_idx else []
                    sim = float(cosine_similarity([base_emb], [rec_emb])[0][0]) if base_emb and rec_emb else 0.0
                    kept = sim >= SIM_THRESHOLDS["reconstruction"]
                    sample["interventions"].append({
                        "type": "reconstruction", "text": rec_q, "template_used": tpl,
                        "question_type": q_type, "sim": round(sim, 4), "kept": kept, "embedding": rec_emb if kept else None
                    })
                    stats["reconstruction_kept" if kept else "reconstruction_filtered"] += 1
                
                # 掩码干预
                for k in range(M_MASK):
                    mask_q, cov, mask_pos = mask_results_list[k][j]
                    mask_emb = all_embs[text_to_idx[mask_q]].tolist() if mask_q in text_to_idx else []
                    sample["interventions"].append({
                        "type": "mask", "text": mask_q, "mask_positions": mask_pos,
                        "mask_coverage": round(cov, 4), "embedding": mask_emb, "sim": 1.0, "kept": True
                    })
                    stats["mask_generated"] += 1
                
                buffer.append(json.dumps(sample, ensure_ascii=False))
                
                # 批量写入并刷盘（每1000条）
                if len(buffer) >= 1000:
                    output_file.write('\n'.join(buffer) + '\n')
                    output_file.flush()  # 强制刷盘
                    os.fsync(output_file.fileno())
                    buffer.clear()
        
        # 写入剩余数据（核心修复：确保最后数据写入）
        if buffer:
            output_file.write('\n'.join(buffer) + '\n')
            output_file.flush()
            os.fsync(output_file.fileno())
            print(f"[Info] 剩余{len(buffer)}条数据已写入")
        buffer.clear()
        
        print("[Info] JSON干预文件已完整写入！")
    
    except Exception as e:
        print(f"[Error] 生成干预结果时出错：{str(e)}")
        # 异常时仍尝试写入剩余数据
        if buffer:
            with open(f"{OUTPUT_JSONL}.temp", "w", encoding="utf-8") as f:
                f.write('\n'.join(buffer) + '\n')
            print(f"[Info] 异常时剩余数据已备份至：{os.path.basename(f'{OUTPUT_JSONL}.temp')}")
        raise e
    finally:
        output_file.close()  # 确保文件关闭
    
    # 生成报告（同样强制刷盘）
    report = {
        "dataset_total": total_questions,
        "replacement_kept": stats["replacement_kept"],
        "replacement_filtered": stats["replacement_filtered"],
        "reconstruction_kept": stats["reconstruction_kept"],
        "reconstruction_filtered": stats["reconstruction_filtered"],
        "mask_generated": stats["mask_generated"],
        "version": "v9_stable_4090",
        "batch_size": BATCH_SIZE,
        "embed_batch_size": EMBED_BATCH_SIZE,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    
    # 最终保存缓存（优化后不会卡住）
    print("[Info] 开始保存缓存...")
    save_embedding_cache()
    
    print("\n✅ 4090稳定版干预生成完成！")
    for k, v in report.items():
        print(f"{k:<30}: {v}")

# ------------------------ 启动入口 ------------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(16)
    
    # 加载数据和资源
    dataset = load_multiple_jsons(DATA_PATH)
    word_synonyms = load_medical_synonyms(WORD_SYNONYM_PATH)
    type_to_templates = load_type_templates()
    load_embedding_cache()
    
    # 生成干预
    generate_interventions(dataset, word_synonyms, type_to_templates)