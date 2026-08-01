"""
generate_embeddings.py
======================
生成 dataloader 所需的 embeddings.npz 和 embedding_index.json 文件。

该脚本完成以下工作:
  1. 读取所有数据集 (train/val/test) 中的问题
  2. 使用 MedicalQuestionPatternAndEntityExtractor 提取每个问题的 syntax_pattern 和 core_entity
  3. 收集所有不重复的 pattern 和 entity_value
  4. 使用 RoBERTa-base 将它们编码为 768 维向量
  5. 保存为 embeddings.npz + embedding_index.json

用法:
    python generate_embeddings.py \
        --data_dir pvqa \
        --output_dir pvqa/embeddings_all \
        --batch_size 64

生成文件:
    <output_dir>/embeddings.npz          -> 包含 pattern_embeddings, entity_value_embeddings
    <output_dir>/embedding_index.json    -> 包含 patterns, entity_values 的索引映射
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

# 将项目根目录加入 sys.path，以便正确导入 models.do_question
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.do_question import MedicalQuestionPatternAndEntityExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate pattern & entity embeddings for VQA dataloader')
    parser.add_argument('--data_dir', type=str, default='data_med',
                        help='数据根目录，包含 train/val/test jsonl 文件')
    parser.add_argument('--train_json', type=str, default=None,
                        help='训练集路径 (默认: <data_dir>/train.json)')
    parser.add_argument('--val_json', type=str, default=None,
                        help='验证集路径 (默认: <data_dir>/val.json)')
    parser.add_argument('--test_json', type=str, default=None,
                        help='测试集路径 (默认: <data_dir>/test.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (默认: <data_dir>/embeddings_all)')
    parser.add_argument('--model_name', type=str, default='pretrain/roberta-base',
                        help='用于编码的预训练模型名称或本地路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='编码时的 batch size')
    parser.add_argument('--max_length', type=int, default=64,
                        help='tokenizer 最大长度')
    parser.add_argument('--device', type=str, default='',
                        help='设备 (留空自动选择 cuda/cpu)')
    return parser.parse_args()


def load_questions_from_file(filepath):
    """从 JSON 或 JSONL 文件中加载所有问题文本"""
    questions = []
    if not os.path.exists(filepath):
        logger.warning(f"文件不存在，跳过: {filepath}")
        return questions

    logger.info(f"加载文件: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.jsonl'):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    q = item.get('question', '')
                    if q:
                        questions.append(q)
                except json.JSONDecodeError:
                    continue
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    q = item.get('question', '')
                    if q:
                        questions.append(q)

    logger.info(f"  -> 加载了 {len(questions)} 条问题")
    return questions


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=64, max_length=64):
    """
    使用 RoBERTa 将一组文本编码为 [CLS] 向量。
    返回 numpy array, shape = (len(texts), hidden_size)
    """
    model.eval()
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch_texts = texts[start: start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的隐藏状态 (index 0)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    args = parse_args()

    # ---------- 设置路径 ----------
    data_dir = args.data_dir
    train_json = args.train_json or os.path.join(data_dir, 'train_typed.jsonl')
    val_json = args.val_json or os.path.join(data_dir, 'val_typed.jsonl')
    test_json = args.test_json or os.path.join(data_dir, 'test_typed.jsonl')
    output_dir = args.output_dir or os.path.join(data_dir, 'embeddings_all')

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # ---------- 1. 加载所有问题 ----------
    logger.info("=" * 50)
    logger.info("第 1 步: 加载数据集中的所有问题")
    logger.info("=" * 50)

    all_questions = []
    for path, name in [(train_json, 'train'), (val_json, 'val'), (test_json, 'test')]:
        qs = load_questions_from_file(path)
        all_questions.extend(qs)

    # 去重（保留所有唯一问题进行提取）
    unique_questions = list(set(all_questions))
    logger.info(f"总问题数: {len(all_questions)}, 去重后: {len(unique_questions)}")

    # ---------- 2. 提取所有 pattern 和 entity ----------
    logger.info("=" * 50)
    logger.info("第 2 步: 使用 MedicalQuestionPatternAndEntityExtractor 提取 pattern 和 entity")
    logger.info("=" * 50)

    extractor = MedicalQuestionPatternAndEntityExtractor()

    patterns_set = set()
    entity_values_set = set()
    extract_errors = 0

    for q in tqdm(unique_questions, desc="Extracting patterns & entities"):
        try:
            result = extractor.extract_pattern(q)
            syntax_pattern = result.get('syntax_pattern', '')
            core_entity = result.get('core_entity', {})
            entity_value = core_entity.get('value', '')

            if syntax_pattern:
                patterns_set.add(syntax_pattern)
            if entity_value:
                entity_values_set.add(entity_value)
        except Exception as e:
            extract_errors += 1
            if extract_errors <= 5:
                logger.warning(f"提取失败 (问题: '{q[:60]}...'): {e}")

    # 排序以保证确定性
    patterns_list = sorted(patterns_set)
    entity_values_list = sorted(entity_values_set)

    logger.info(f"提取完成:")
    logger.info(f"  - 不重复 syntax_pattern 数量: {len(patterns_list)}")
    logger.info(f"  - 不重复 entity_value 数量:   {len(entity_values_list)}")
    if extract_errors > 0:
        logger.warning(f"  - 提取失败数量: {extract_errors}")

    # 打印一些示例
    logger.info(f"  - pattern 示例: {patterns_list[:10]}")
    logger.info(f"  - entity 示例:  {entity_values_list[:10]}")

    # ---------- 3. 加载 RoBERTa 编码器 ----------
    logger.info("=" * 50)
    logger.info("第 3 步: 加载 RoBERTa 模型进行编码")
    logger.info("=" * 50)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    logger.info(f"使用设备: {device}")

    logger.info(f"加载 tokenizer: {args.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    logger.info(f"加载模型: {args.model_name}")
    model = RobertaModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    logger.info(f"模型隐藏维度: {hidden_size}")

    # ---------- 4. 编码 pattern 和 entity ----------
    logger.info("=" * 50)
    logger.info("第 4 步: 编码 patterns 和 entity_values")
    logger.info("=" * 50)

    if len(patterns_list) > 0:
        logger.info(f"编码 {len(patterns_list)} 个 patterns ...")
        pattern_embeddings = encode_texts(
            patterns_list, tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )
    else:
        logger.warning("没有找到任何 pattern，将创建空数组")
        pattern_embeddings = np.zeros((0, hidden_size), dtype=np.float32)

    if len(entity_values_list) > 0:
        logger.info(f"编码 {len(entity_values_list)} 个 entity_values ...")
        entity_value_embeddings = encode_texts(
            entity_values_list, tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )
    else:
        logger.warning("没有找到任何 entity_value，将创建空数组")
        entity_value_embeddings = np.zeros((0, hidden_size), dtype=np.float32)

    logger.info(f"编码结果:")
    logger.info(f"  - pattern_embeddings shape:      {pattern_embeddings.shape}")
    logger.info(f"  - entity_value_embeddings shape:  {entity_value_embeddings.shape}")

    # ---------- 5. 构建索引映射 ----------
    logger.info("=" * 50)
    logger.info("第 5 步: 构建索引映射并保存文件")
    logger.info("=" * 50)

    # 索引: pattern_string -> int, entity_value_string -> int
    patterns_index = {pattern: idx for idx, pattern in enumerate(patterns_list)}
    entity_values_index = {entity: idx for idx, entity in enumerate(entity_values_list)}

    embedding_index = {
        "patterns": patterns_index,
        "entity_values": entity_values_index
    }

    # ---------- 6. 保存文件 ----------
    npz_path = os.path.join(output_dir, 'embeddings.npz')
    json_path = os.path.join(output_dir, 'embedding_index.json')

    np.savez(
        npz_path,
        pattern_embeddings=pattern_embeddings.astype(np.float32),
        entity_value_embeddings=entity_value_embeddings.astype(np.float32)
    )
    logger.info(f"已保存: {npz_path}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_index, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {json_path}")

    # ---------- 7. 验证 ----------
    logger.info("=" * 50)
    logger.info("第 6 步: 验证生成的文件")
    logger.info("=" * 50)

    # 重新加载并检查
    loaded_emb = np.load(npz_path)
    loaded_idx = json.load(open(json_path, 'r', encoding='utf-8'))

    assert 'pattern_embeddings' in loaded_emb, "缺少 pattern_embeddings"
    assert 'entity_value_embeddings' in loaded_emb, "缺少 entity_value_embeddings"
    assert 'patterns' in loaded_idx, "缺少 patterns 索引"
    assert 'entity_values' in loaded_idx, "缺少 entity_values 索引"

    assert loaded_emb['pattern_embeddings'].shape[0] == len(loaded_idx['patterns']), \
        f"pattern 数量不匹配: embeddings={loaded_emb['pattern_embeddings'].shape[0]}, index={len(loaded_idx['patterns'])}"
    assert loaded_emb['entity_value_embeddings'].shape[0] == len(loaded_idx['entity_values']), \
        f"entity 数量不匹配: embeddings={loaded_emb['entity_value_embeddings'].shape[0]}, index={len(loaded_idx['entity_values'])}"

    logger.info("验证通过!")
    logger.info(f"  - pattern_embeddings:      {loaded_emb['pattern_embeddings'].shape}")
    logger.info(f"  - entity_value_embeddings: {loaded_emb['entity_value_embeddings'].shape}")
    logger.info(f"  - patterns 索引数:         {len(loaded_idx['patterns'])}")
    logger.info(f"  - entity_values 索引数:    {len(loaded_idx['entity_values'])}")

    # 打印随机抽样验证
    logger.info("\n抽样检查 (前 5 个 pattern):")
    for p in patterns_list[:5]:
        idx = patterns_index[p]
        vec = pattern_embeddings[idx]
        logger.info(f"  pattern='{p}' -> index={idx}, embedding_norm={np.linalg.norm(vec):.4f}")

    logger.info("\n抽样检查 (前 5 个 entity_value):")
    for e in entity_values_list[:5]:
        idx = entity_values_index[e]
        vec = entity_value_embeddings[idx]
        logger.info(f"  entity='{e}' -> index={idx}, embedding_norm={np.linalg.norm(vec):.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("全部完成! 文件已生成:")
    logger.info(f"  {npz_path}")
    logger.info(f"  {json_path}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
