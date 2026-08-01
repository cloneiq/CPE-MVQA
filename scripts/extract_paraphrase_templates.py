#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练数据中提取问题句式，生成 _PARAPHRASE_TEMPLATES 的补充建议

用法:
  python scripts/extract_paraphrase_templates.py --train_json data_med/train_typed.jsonl
  python scripts/extract_paraphrase_templates.py --train_json data_med/train_typed.jsonl --output suggestions.txt

输出:
  1. 按 answer_type 分组的唯一问题句式（去重、按频次排序）
  2. 可直接复制到 _PARAPHRASE_TEMPLATES 的 Python 代码建议
"""
import os
import re
import json
import argparse
from collections import defaultdict, Counter


def load_questions(path: str) -> list:
    """加载问题，支持 JSON 和 JSONL"""
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    result = []
    for d in data:
        q = (d.get("question") or d.get("question_text") or "").strip()
        q = re.sub(r"\s+", " ", q)
        answer_type = (
            d.get("answer_type") or d.get("content_type") or d.get("type") or ""
        ).strip().lower()
        if q:
            result.append({"question": q.lower(), "answer_type": answer_type or "general"})
    return result


def infer_answer_type(question: str) -> str:
    """根据问题推断 answer_type（当数据中无该字段时）"""
    q = question.lower()
    # SLAKE/VQA-RAD: yes/no 问题 -> closed
    if q.startswith(("is ", "are ", "does ", "do ", "was ", "were ", "has ", "have ")) or q.startswith("can "):
        if any(k in q for k in ["yes", "no", "?"]):
            return "closed"
    if any(k in q for k in ["modality", "ct", "mri", "x-ray", "ultrasound", "pet", "scan"]):
        return "modality"
    if any(k in q for k in ["plane", "axial", "coronal", "sagittal", "view"]):
        return "plane"
    if any(k in q for k in ["organ", "liver", "lung", "heart", "kidney", "brain", "anatomy"]):
        return "organ"
    if any(k in q for k in ["abnormal", "lesion", "mass", "tumor", "fracture", "disease"]):
        return "abnormality"
    return "general"


def get_pattern_prefix(question: str, n_words: int = 5) -> str:
    """提取问题前 n 个词作为模式前缀"""
    words = re.findall(r"[a-z]+", question.lower())
    return " ".join(words[:n_words]) if words else ""


def main():
    parser = argparse.ArgumentParser(description="从训练数据提取 Paraphrase 模板建议")
    parser.add_argument(
        "--train_json",
        type=str,
        nargs="+",
        default=["data_med/train_typed.jsonl"],
        help="训练数据路径，可多个 (JSON 或 JSONL)，如 data_Slake/train.jsonl data_RAD/train.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出文件路径，不指定则打印到控制台",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="每类最多输出前 K 个问题句式",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="最少出现次数才纳入建议",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = args.train_json if isinstance(args.train_json, list) else [args.train_json]
    items = []
    for p in paths:
        path = p if os.path.isabs(p) else os.path.join(base, p)
        loaded = load_questions(path)
        items.extend(loaded)
        if loaded:
            print(f"加载 {path}: {len(loaded)} 条")

    if not items:
        print("未加载到任何问题，请检查路径")
        return

    with_type = [x for x in items if x["answer_type"]]
    without_type = [x for x in items if not x["answer_type"]]
    for x in without_type:
        x["answer_type"] = infer_answer_type(x["question"])

    by_type = defaultdict(list)
    for x in items:
        by_type[x["answer_type"]].append(x["question"])

    type_order = ["modality", "plane", "organ", "abnormality", "open", "closed", "general"]
    lines = []
    lines.append(f"# 从 {paths} 提取的 Paraphrase 模板建议")
    lines.append(f"# 共 {len(items)} 条问题，{len(by_type)} 个类型")
    lines.append("")

    for atype in type_order:
        if atype not in by_type:
            continue
        qs = by_type[atype]
        counter = Counter(qs)
        unique = sorted(counter.items(), key=lambda x: -x[1]).copy()

        lines.append(f"## {atype} ({len(qs)} 条)")
        lines.append("")

        for q, cnt in unique[: args.top_k]:
            if cnt < args.min_freq:
                continue
            lines.append(f"  # {cnt} 次: {q[:60]}{'...' if len(q) > 60 else ''}")
            lines.append(f'  "{q}",')
        lines.append("")

    lines.append("\n# --- 可复制到 _PARAPHRASE_TEMPLATES 的 Python 代码 ---\n")
    lines.append("_PARAPHRASE_TEMPLATES = {")
    for atype in type_order:
        if atype not in by_type:
            continue
        qs = by_type[atype]
        counter = Counter(qs)
        unique = sorted(counter.items(), key=lambda x: -x[1])
        kept = [q for q, c in unique[: args.top_k] if c >= args.min_freq]
        if not kept:
            continue
        lines.append(f'    "{atype}": [')
        for q in kept:
            lines.append(f'        "{q}",')
        lines.append("    ],")
    lines.append("}")

    out = "\n".join(lines)
    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.output)
        out_path = os.path.normpath(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"已写入: {out_path}")
    else:
        print(out)


if __name__ == "__main__":
    main()
