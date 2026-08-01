"""
generate_answer_embeddings.py
=============================
为 Open 类问题的 embedding matching 生成答案嵌入。

用法:
    python generate_answer_embeddings.py \
        --data_dir pvqa \
        --roberta_path pretrain/roberta-base \
        --output_path pvqa/answer_embeddings.pt

生成文件:
    <output_path>: dict with 'embeddings' [vocab_size, 768], 'idx2answer', 'answer2idx'
"""

import os
import sys
import json
import argparse
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate answer embeddings for Open embedding matching')
    parser.add_argument('--data_dir', type=str, default='data_RAD', help='数据目录，包含 answer_vocab.json')
    parser.add_argument('--vocab_path', type=str, default='', help='answer_vocab.json 路径，默认 data_dir/answer_vocab.json')
    parser.add_argument('--roberta_path', type=str, default='pretrain/roberta-base', help='RoBERTa 路径')
    parser.add_argument('--output_path', type=str, default='', help='输出路径，默认 data_dir/answer_embeddings.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--device', type=str, default='')
    return parser.parse_args()


@torch.no_grad()
def encode_answers(answers, tokenizer, model, device, batch_size=64, max_length=32):
    """使用 RoBERTa [CLS] 编码答案文本"""
    model.eval()
    all_emb = []
    for i in tqdm(range(0, len(answers), batch_size), desc="Encoding answers"):
        batch = answers[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls_emb = out.last_hidden_state[:, 0, :]  # [B, 768]
        all_emb.append(cls_emb)
    return torch.cat(all_emb, dim=0)


def main():
    args = parse_args()
    vocab_path = args.vocab_path or os.path.join(args.data_dir, 'answer_vocab.json')
    output_path = args.output_path or os.path.join(args.data_dir, 'answer_embeddings.pt')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    if not os.path.exists(vocab_path):
        logger.error(f"answer_vocab.json 不存在: {vocab_path}，请先运行训练以生成词表")
        sys.exit(1)

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    idx2answer = vocab['idx2answer']
    # Use answer2idx for vocab_size to match training dataloader
    answer2idx = vocab.get('answer2idx', {})
    vocab_size = len(answer2idx) if answer2idx else len(idx2answer)
    # CRITICAL: embeddings[i] must align with vocab index i (CE target = answer_idx)
    # Must iterate strictly in order 0, 1, 2, ..., vocab_size-1
    answers = []
    for i in range(vocab_size):
        ans = idx2answer.get(str(i), idx2answer.get(i, '<UNK>'))
        answers.append(ans)
    if len(answers) != vocab_size:
        raise ValueError(f"Vocab index mismatch: got {len(answers)} answers for vocab_size {vocab_size}")

    logger.info(f"加载词表: {vocab_path}, vocab_size={vocab_size}")

    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_path)
    model = RobertaModel.from_pretrained(args.roberta_path)
    model = model.to(device)
    model.eval()

    embeddings = encode_answers(answers, tokenizer, model, device, args.batch_size, args.max_length)
    embeddings = F.normalize(embeddings.float(), p=2, dim=-1)  # L2 归一化，便于 cosine

    save_dict = {
        'embeddings': embeddings.cpu(),
        'idx2answer': idx2answer,
        'answer2idx': vocab.get('answer2idx', {}),
        'vocab_size': vocab_size,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(save_dict, output_path)
    logger.info(f"已保存: {output_path}, shape={embeddings.shape}")


if __name__ == '__main__':
    main()
