# import os
# import sys
# import os
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer, RobertaTokenizer, RobertaConfig, RobertaModel
from collections import defaultdict, Counter, OrderedDict
import torch.nn as nn
import pickle
import time
import logging
from tqdm import tqdm
import random
import os
from .data_tools import colorful_spectrum_mix
import unicodedata
from typing import List, Dict
import h5py

from models.do_question import MedicalQuestionPatternAndEntityExtractor



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VQADataLoader')

for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if env_var in os.environ:
        os.environ.pop(env_var)
os.environ['NO_PROXY'] = '*'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer



class VQADataset(Dataset):

    def __init__(self, data_dir, data_entries, image_dir, transform=None, max_length=32, tokenizer='roberta',
                 answer_vocab=None, mode='train', image_size=384, alpha=1.0, config=None):
        self.config = config or {}
        self.skip_ae_maml = self.config.get('skip_ae_maml', False)  # 模型未使用 ae/maml，跳过可节省每样本 2 次图像变换
        self.data_dir = data_dir
        self.image_size = image_size
        self.data_entries = data_entries
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.mode = mode
        self.extractor = MedicalQuestionPatternAndEntityExtractor()
        embeddings = np.load(os.path.join(self.config.get('embeddings_dir', 'data_med/embeddings_all'), 'embeddings.npz'))
        self.pattern_embeddings = embeddings['pattern_embeddings']
        self.entity_value_embeddings = embeddings['entity_value_embeddings']
        with open(os.path.join(self.config.get('embeddings_dir', 'data_med/embeddings_all'), 'embedding_index.json'), 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        self.pattern_insex = self.index['patterns']
        self.entity_value_index = self.index['entity_values']

        # 多进程 DataLoader 下每个 worker 一份 dataset：无界 image 缓存会随 epoch 撑爆内存，
        # persistent_workers=True 时更明显。image_cache_max_entries>0 时用 LRU（默认由 VQADataLoader 按 num_workers 设置）。
        self.image_cache_max = int(self.config.get('image_cache_max_entries', 0) or 0)
        if self.image_cache_max > 0:
            self.image_cache: OrderedDict = OrderedDict()
            self.ae_image_cache: OrderedDict = OrderedDict()
            self.maml_image_cache: OrderedDict = OrderedDict()
        else:
            self.image_cache = {}
            self.ae_image_cache = {}
            self.maml_image_cache = {}

        self.alpha = alpha

        self.pre_transform = transforms.Compose([transforms.Resize((image_size, image_size))])

        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load tokenizer: support aliases ('bert'/'roberta') and local HF paths.
        tok_name = str(tokenizer).strip()
        tok_lower = tok_name.lower()
        if tok_lower == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif tok_lower == 'roberta':
            roberta_path = config.get('roberta_path', 'pretrain/roberta-base') if config else 'pretrain/roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path, local_files_only=True)
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_token>']})
            self.visual_token_id = self.tokenizer.convert_tokens_to_ids('<visual_token>')
        elif ('roberta' in tok_lower) or os.path.exists(tok_name):
            # e.g. './pretrain/roberta-base-1'
            self.tokenizer = RobertaTokenizer.from_pretrained(tok_name, local_files_only=True)
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<visual_token>']})
            self.visual_token_id = self.tokenizer.convert_tokens_to_ids('<visual_token>')
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer}")

        # Set answer vocabulary
        self.answer_vocab = answer_vocab


    def _get_tokenized(self, idx, question):
        """Get tokenization result"""
        encoded = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def _resolve_image_path(self, img_name):
        """解析图像路径，支持多种目录结构：image_dir、image_dir/images、image_dir/imgs、data_dir/images、data_dir/imgs"""
        candidates = [
            os.path.join(self.image_dir, img_name),
            os.path.join(self.image_dir, 'images', img_name),
            os.path.join(self.image_dir, 'imgs', img_name),
            os.path.join(self.data_dir, img_name),
            os.path.join(self.data_dir, 'images', img_name),
            os.path.join(self.data_dir, 'imgs', img_name),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]  # 返回默认路径，后续加载失败时由 _load_image 处理

    def _image_cache_touch(self, img_path):
        if self.image_cache_max <= 0 or img_path not in self.image_cache:
            return
        self.image_cache.move_to_end(img_path)
        if not self.skip_ae_maml:
            if img_path in self.ae_image_cache:
                self.ae_image_cache.move_to_end(img_path)
            if img_path in self.maml_image_cache:
                self.maml_image_cache.move_to_end(img_path)

    def _image_cache_evict_if_needed(self):
        if self.image_cache_max <= 0:
            return
        while len(self.image_cache) > self.image_cache_max:
            k, _ = self.image_cache.popitem(last=False)
            if not self.skip_ae_maml:
                self.ae_image_cache.pop(k, None)
                self.maml_image_cache.pop(k, None)

    def _load_image(self, img_path):
        """Load image, return multiple sizes, use cache for efficiency"""
        _zeros_ae = torch.zeros(1, 128, 128)
        _zeros_maml = torch.zeros(1, 84, 84)
        try:
            # Check cache
            if img_path in self.image_cache:
                self._image_cache_touch(img_path)
                image = self.image_cache[img_path]
                if self.skip_ae_maml:
                    return image, _zeros_ae, _zeros_maml
                ae_image = self.ae_image_cache[img_path]
                maml_image = self.maml_image_cache[img_path]
                return image, ae_image, maml_image

            # Load original image
            original_image = Image.open(img_path).convert('RGB')

            # Main image transform
            if self.transform:
                image = self.transform(original_image)
            else:
                image = transforms.ToTensor()(original_image)

            if self.skip_ae_maml:
                self.image_cache[img_path] = image
                self._image_cache_evict_if_needed()
                return image, _zeros_ae, _zeros_maml

            # Autoencoder image (128x128)
            ae_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            ae_image = ae_transform(original_image)

            # MAML image (84x84)
            maml_transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            maml_image = maml_transform(original_image)

            # Update cache
            self.image_cache[img_path] = image
            self.ae_image_cache[img_path] = ae_image
            self.maml_image_cache[img_path] = maml_image
            self._image_cache_evict_if_needed()

            return image, ae_image, maml_image
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Create default zero tensor as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
            ae_image = torch.zeros(3, 128, 128)
            maml_image = torch.zeros(3, 84, 84)
            return image, ae_image, maml_image
        

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        """Get a single data entry, efficient processing, support frequency-based soft score"""
        entry = self.data_entries[idx]

        # Get current image path and ID
        # Support both 'image_name' (new) and 'img_name' (old)
        img_name = entry.get('image_name', entry.get('img_name', ''))
        img_path = self._resolve_image_path(img_name)
        
        # Use qid if exists, otherwise use img_id, otherwise use index
        qid = str(entry.get('qid', entry.get('img_id', str(idx))))

        # Load image
        image, ae_image, maml_image = self._load_image(img_path)

        # Load and preprocess sample image (retry if file not found)
        sample_image = None
        for _ in range(30):
            sample_idx = random.randint(0, len(self.data_entries) - 1)
            sample_entry = self.data_entries[sample_idx]
            sample_img_name = sample_entry.get('image_name', sample_entry.get('img_name', ''))
            sample_img_path = self._resolve_image_path(sample_img_name)
            if sample_img_path == img_path:
                continue
            try:
                if os.path.exists(sample_img_path):
                    sample_image = Image.open(sample_img_path).convert('RGB')
                    sample_image = self.pre_transform(sample_image)
                    sample_image = np.array(sample_image)
                    break
            except Exception:
                continue
        if sample_image is None:
            sample_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Load or_image (main image for FFT mix)
        try:
            or_image = Image.open(img_path).convert('RGB')
            or_image = self.pre_transform(or_image)
            or_image = np.array(or_image)
        except Exception:
            # Fallback: use image tensor from _load_image, denormalize to 0-255
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            or_image = image.permute(1, 2, 0).numpy()
            or_image = np.clip((or_image * std + mean) * 255, 0, 255).astype(np.uint8)

        # Apply FFT mix
        img21, img12 = colorful_spectrum_mix(or_image, sample_image, alpha=self.alpha, strategy='basic')

        # Convert to PIL image and apply post processing
        img21_pil = Image.fromarray(img21)
        img12_pil = Image.fromarray(img12)
        pos_image = self.post_transform(img21_pil)
        neg_image = self.post_transform(img12_pil)

        # Process question
        question = entry['question']
        if '?' in question:
            do_question = question.replace('?', ' <visual_token>?')
            do_tokenized = self._get_tokenized(idx, do_question)
            do_input_ids = do_tokenized['input_ids']
            do_attention_mask = do_tokenized['attention_mask']
        else:
            do_question = question + ' <visual_token>'
            do_tokenized = self._get_tokenized(idx, do_question)
            do_input_ids = do_tokenized['input_ids']
            do_attention_mask = do_tokenized['attention_mask']
        tokenized = self._get_tokenized(idx, question)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']


        pattern_entity = self.extractor.extract_pattern(question)
        if pattern_entity['syntax_pattern'] in self.pattern_insex:
            pattern_embedding = self.pattern_embeddings[self.pattern_insex[pattern_entity['syntax_pattern']]]
        else:
            pattern_embedding = torch.zeros(768)
        if pattern_entity['core_entity']['value'] in self.entity_value_index:
            entity_embedding = self.entity_value_embeddings[
                self.entity_value_index[pattern_entity['core_entity']['value']]]
        else:
            entity_embedding = torch.zeros(768)
        # 避免 torch.tensor(tensor) 的 UserWarning：numpy 用 torch.tensor，已有 tensor 用 clone().detach()
        if isinstance(pattern_embedding, torch.Tensor):
            pattern_embedding = pattern_embedding.clone().detach().float()
        else:
            pattern_embedding = torch.tensor(pattern_embedding, dtype=torch.float32)
        if isinstance(entity_embedding, torch.Tensor):
            entity_embedding = entity_embedding.clone().detach().float()
        else:
            entity_embedding = torch.tensor(entity_embedding, dtype=torch.float32)

        answer_text = preprocess_answer(entry['answer'])
        target = torch.zeros(self.answer_vocab['vocab_size'])
        answer_idx = self.answer_vocab['answer2idx'].get(answer_text, -1)
        if answer_idx < 0:
            # OOV: map to <UNK> index instead of scattering to -1 (last element)
            unk_idx = self.answer_vocab['answer2idx'].get('<UNK>', len(self.answer_vocab['answer2idx']) - 1)
            answer_idx = unk_idx
            scores = 0.3  # low confidence for OOV
        else:
            scores = self.answer_vocab['answer2score'].get(answer_text, 1.0)
        target.scatter_(0, torch.tensor([answer_idx]), torch.tensor([scores]))

        # Assume image_id and image path are already obtained from data dict
        image_path = img_path

        # Build mask path - in the same directory as the source image
        mask_path = os.path.join(os.path.dirname(image_path), 'mask.png')

        # If mask.png not found, try to find in imgs directory
        if not os.path.exists(mask_path):
            # Try in the same directory as current image
            alt_mask_path = os.path.join(os.path.dirname(image_path), f"mask.png")
            if os.path.exists(alt_mask_path):
                mask_path = alt_mask_path

        # Load and process mask (if exists)
        mask = None
        if os.path.exists(mask_path):
            try:
                # Load mask image
                mask_img = Image.open(mask_path).convert('L')

                # Check if mask is all black (invalid mask)
                mask_array = np.array(mask_img)
                if mask_array.max() > 0:  # Mask is not all black
                    # Create mask transform (only resize, no color transform)
                    mask_transform = transforms.Compose([
                        lambda img: np.array(img) > 0.5,  # Binarize to bool array
                        lambda x: Image.fromarray(x.astype(np.uint8) * 255),
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        lambda x: (x > 0.5).float()
                    ])
                    mask = mask_transform(mask_img)
                else:
                    # Mask is all black, use None
                    mask = torch.zeros((1, self.image_size, self.image_size))
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                mask = torch.zeros((1, self.image_size, self.image_size))
        else:
            # Mask does not exist, use None
            mask = torch.zeros((1, self.image_size, self.image_size))

        # Pack result (ensure qid is returned)
        result = {
            'image': image,
            'pos_image': pos_image,
            'neg_image': neg_image,
            'ae_image': ae_image,
            'maml_image': maml_image,
            'question': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            },
            'do_question': {
                'input_ids': do_input_ids,
                'attention_mask': do_attention_mask,
            },
            'question_text': question,
            'target': target,  # Frequency-based soft target vector
            'answer_idx': answer_idx,  # Main answer index
            'answer_text': answer_text,  # Main answer text
            'image_path': img_path,
            'mask': mask,  # Add mask
            'pattern_embedding': pattern_embedding,
            'entity_embedding': entity_embedding,
            'qid': qid  # Add question ID for tracking
        }

        # Add other optional fields (重点：列表里加上 'category')
        for key in ['location', 'modality', 'qid', 'answer_type', 'content_type', 'type', 'category']:
            if key in entry:
                result[key] = entry[key]

        # Concept 抽取：SLAKE 用四层语义单元，VQA-RAD 用 disease-first
        # 互斥：use_slake_concept 优先（SLAKE 数据集），否则 use_vqa_rad_concept（VQA-RAD）
        if self.config.get('use_slake_concept', False):
            try:
                from utils.slake_concept import get_concept, MISC_IDX
                concept_name, concept_idx = get_concept(
                    question=entry.get('question', ''),
                    answer=entry.get('answer', ''),
                    answer_type=result.get('answer_type', entry.get('answer_type', '')),
                )
                result['concept'] = concept_name
                result['concept_idx'] = concept_idx
            except Exception:
                from utils.slake_concept import MISC_IDX
                result['concept'] = 'misc'
                result['concept_idx'] = MISC_IDX
        elif self.config.get('use_vqa_rad_concept', False):
            try:
                from utils.vqa_rad_concept import get_concept
                concept_name, concept_idx = get_concept(
                    question=entry.get('question', ''),
                    answer=entry.get('answer', ''),
                    answer_type=result.get('answer_type', entry.get('answer_type', '')),
                )
                result['concept'] = concept_name
                result['concept_idx'] = concept_idx
            except Exception:
                result['concept'] = 'misc'
                result['concept_idx'] = 4  # MISC_IDX (5 classes: 0-4)

        # Fallback: 优先尝试拿 category 塞给 answer_type，解决 MEDVQA2019 不识别的问题
        if 'answer_type' not in result:
            if 'category' in result:
                result['answer_type'] = result['category']
            elif 'content_type' in result:
                result['answer_type'] = result['content_type']
            elif 'type' in result:
                result['answer_type'] = result['type']

        # SLAKE/VQA-RAD/MEDVQA: 统一映射为框架的 4 大类 (modality/plane/organ/abnormality)
        _content_map = {
            "modality": "modality", "position": "plane", "organ": "organ",
            "abnormality": "abnormality", "size": "organ", "plane": "plane"
        }
        at = (result.get('answer_type') or '').strip()
        at_u = at.upper()
        if at_u == 'CLOSE':
            at_u = 'CLOSED'

        def _norm_pathvqa_fine_subtype(ct_raw):
            """PathVQA 细类名 CLOSE → closed（与 test 里 category 键一致）。"""
            if not ct_raw:
                return ct_raw
            t = str(ct_raw).strip().lower()
            return "closed" if t == "close" else t

        # 提取类别变量 ct 时，也把 category 放在第一顺位（同时保留 open_closed 供 eval 粗分）
        if at_u in ('OPEN', 'CLOSED'):
            result['open_closed'] = 'open' if at_u == 'OPEN' else 'closed'
            ct = _norm_pathvqa_fine_subtype(
                (result.get('category') or result.get('content_type') or result.get('type') or "").strip().lower()
            )
            mapped = _content_map.get(ct, ct) if ct else ('open' if at_u == 'OPEN' else 'closed')
            result['answer_type'] = mapped if mapped else ('open' if at_u == 'OPEN' else 'closed')
        elif at and at.lower() not in _content_map:
            ct = _norm_pathvqa_fine_subtype(
                (result.get('category') or result.get('content_type') or result.get('type') or "").strip().lower()
            )
            if ct and ct in _content_map:
                result['answer_type'] = _content_map[ct]
            else:
                result['answer_type'] = at.lower()
        else:
            result['answer_type'] = at.lower()

        return result


class VQADataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.image_dir = config.get('image_dir', os.path.join(self.data_dir, 'slake/imgs'))
        self.train_json = config.get('train_json', os.path.join(self.data_dir, 'slake/train.json'))
        self.val_json = config.get('val_json', os.path.join(self.data_dir, 'slake/validate.json'))
        self.test_json = config.get('test_json', os.path.join(self.data_dir, 'slake/test.json'))
        self.batch_size = config.get('batch_size', 32)
        self.val_batch_size = config.get('val_batch_size', self.batch_size)  # 验证可用更大 batch 加速
        self.num_workers = config.get('num_workers', 16)
        self.val_num_workers = config.get('val_num_workers', 0)  # 0=单进程，避免验证时 OOM
        self.image_size = config.get('image_size', 224)
        self.max_length = config.get('max_length', 32)
        self.tokenizer = config.get('tokenizer', 'roberta-base')
        self.min_answer_freq = config.get('min_answer_freq', 5)
        self.rebuild_vocab = config.get('rebuild_vocab', False)
        self.device = config.get('device', 'cuda')

        # Initialization
        self._init_transforms()
        self._load_data()
        self._build_answer_vocab()
        self._init_datasets()
        self._init_loaders()

        # SLAKE concept 分布统计（训练前打印，misc>25% 时提示扩展词表）
        if self.config.get('use_slake_concept', False) and self.train_data:
            try:
                from utils.slake_concept import print_concept_stats
                sample = self.train_data[:20] if len(self.train_data) >= 20 else self.train_data
                at_key = 'answer_type' if any('answer_type' in e for e in sample) else ('content_type' if any('content_type' in e for e in sample) else 'type')
                if at_key == 'content_type':
                    print_concept_stats(self.train_data, answer_type_key=at_key, open_vals=['organ', 'abnormality'])
                else:
                    print_concept_stats(self.train_data, answer_type_key=at_key, open_val='open')
            except Exception as ex:
                logger.warning(f"SLAKE concept stats skipped: {ex}")

        # Clean up temporary variables to save memory
        if not config.get('keep_raw_data', False):
            self.train_data = None
            self.val_data = None
            self.test_data = None

    def _init_transforms(self):
        """Initialize image transforms (增强版: RandomRotation±15°, ColorJitter, 小batch_size下提升数据多样性)"""
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # ±15° 旋转, 尺度 0.9~1.1
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_data(self):
        """Load datasets, with error handling and format validation"""

        # General loading function
        def load_json(path, name):
            if not os.path.exists(path):
                logger.warning(f"{name} data file does not exist: {path}")
                return []

            logger.info(f"Loading {name} data: {path}")
            try:
                # Support JSONL
                if path.endswith('.jsonl'):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = [json.loads(line) for line in f if line.strip()]
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} {name} data entries")
                return data
            except Exception as e:
                logger.error(f"Failed to load {name} data: {e}")
                return []

        # Load each dataset
        self.train_data = load_json(self.train_json, "train")
        self.val_data = load_json(self.val_json, "val")
        self.test_data = load_json(self.test_json, "test")

        # Merge val into train if requested (for final training before test evaluation)
        if self.config.get('merge_val_train', False) and self.val_data:
            logger.info(f"Merging val ({len(self.val_data)}) into train ({len(self.train_data)})")
            self.train_data = self.train_data + self.val_data
            self.val_data = []
            logger.info(f"Merged train size: {len(self.train_data)}")

        # Dataset format validation
        if self.train_data:
            self._validate_data_format(self.train_data[0], "train")
        if self.val_data:
            self._validate_data_format(self.val_data[0], "val")
        if self.test_data:
            self._validate_data_format(self.test_data[0], "test")

    def _validate_data_format(self, entry, name):
        """Validate data format"""
        # Adapted for new dataset format (allow image_name or img_name)
        if 'image_name' not in entry and 'img_name' not in entry:
             logger.warning(f"{name} data missing image path field (image_name or img_name)")
        
        required_fields = ['question', 'answer']
        for field in required_fields:
            if field not in entry:
                logger.warning(f"{name} data missing required field: {field}")

        logger.info(f"{name} data format: {list(entry.keys())}")

    def _build_answer_vocab(self):
        """Build answer vocabulary, sort by frequency and compute soft scores"""
        vocab_path = os.path.join(self.data_dir, 'answer_vocab.json')

        # If vocab exists and does not need to be rebuilt
        if os.path.exists(vocab_path) and not self.rebuild_vocab:
            print(f"Loading existing answer vocabulary: {vocab_path}")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.answer_vocab = json.load(f)
            # Ensure vocab_size exists (legacy JSON may not have it)
            if 'vocab_size' not in self.answer_vocab:
                self.answer_vocab['vocab_size'] = len(self.answer_vocab.get('answer2idx', {}))

            # Ensure <UNK> exists for OOV handling
            if '<UNK>' not in self.answer_vocab.get('answer2idx', {}):
                av = self.answer_vocab['answer2idx']
                ia = self.answer_vocab['idx2answer']
                unk_idx = len(av)
                av['<UNK>'] = unk_idx
                ia[str(unk_idx)] = '<UNK>'  # JSON keys are strings
                self.answer_vocab['answer2freq']['<UNK>'] = 0
                self.answer_vocab['answer2score']['<UNK>'] = 0.3
                self.answer_vocab['vocab_size'] = len(av)

            # Add compatibility code - ensure answer2score key exists
            if 'answer2score' not in self.answer_vocab:
                print("Answer vocabulary missing frequency score info, adding...")
                self.answer_vocab['answer2freq'] = {}
                self.answer_vocab['answer2score'] = {}

                # Add default frequency score for all answers
                for ans in self.answer_vocab['answer2idx'].keys():
                    if ans != '<UNK>':  # Skip UNK token
                        # Use default frequency 1 (score 0.3)
                        self.answer_vocab['answer2freq'][ans] = 1
                        self.answer_vocab['answer2score'][ans] = 0.3

                # Save updated vocabulary
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.answer_vocab, f, ensure_ascii=False, indent=2)

            print(f"Answer vocabulary size: {self.answer_vocab['vocab_size']}")
            return

        print("Building new answer vocabulary...")

        answer_counter: Counter[str] = Counter()
        norm2raw: Dict[str, str] = {}  # normalized key -> first occurrence of original form
        answer_idx2text: Dict[int, str] = {}

        def process_dataset(dataset: List[Dict], name: str) -> int:
            if not dataset:
                return 0
            cnt = 0
            for item in dataset:
                answer_key = "answer" if "answer" in item else "a" if "a" in item else None
                if not answer_key:
                    continue

                ans_field = item[answer_key]

                norm = preprocess_answer(ans_field)
                answer_counter[norm] += 1
                norm2raw.setdefault(norm, ans_field)
                cnt += 1
            return cnt

        train_cnt = process_dataset(self.train_data, "train")
        val_cnt = process_dataset(self.val_data, "val")
        test_cnt = process_dataset(self.test_data, "test")

        print(f"Collected {train_cnt} answers from training set")
        print(f"Collected {val_cnt} answers from validation set")
        print(f"Collected {test_cnt} answers from test set")
        print(f"Collected {len(answer_counter)} unique normalized answers in total")

        # ---- Sort by frequency ----------------------------------------------------- #
        sorted_answers = sorted(answer_counter.items(), key=lambda kv: kv[1], reverse=True)
        print("Top 10 high-frequency answers (normalized):", sorted_answers[:10])

        answer2idx: Dict[str, int] = {}
        idx2answer: Dict[int, str] = {}
        answer2freq: Dict[str, int] = {}
        answer2score: Dict[str, float] = {}

        # --- Generate mapping & soft scores ---------------------------------------- #
        for i, (norm_ans, freq) in enumerate(sorted_answers):
            answer2idx[norm_ans] = i
            # Keep the first occurrence of the original form for display / reverse mapping
            idx2answer[i] = norm2raw[norm_ans]
            answer2freq[norm_ans] = freq

            # Soft score rules
            if freq == 0:
                score = 0.0
            elif freq == 1:
                score = 0.3
            elif freq == 2:
                score = 0.6
            elif freq == 3:
                score = 0.9
            else:
                score = 1.0
            answer2score[norm_ans] = score

        # ---- Add <UNK> for OOV answers ------------------------------------------- #
        UNK_TOKEN = '<UNK>'
        if UNK_TOKEN not in answer2idx:
            unk_idx = len(answer2idx)
            answer2idx[UNK_TOKEN] = unk_idx
            idx2answer[unk_idx] = UNK_TOKEN
            answer2freq[UNK_TOKEN] = 0
            answer2score[UNK_TOKEN] = 0.3  # low confidence for unknown

        # ---- Aggregate and save --------------------------------------------------- #
        self.answer_vocab = {
            "answer2idx": answer2idx,
            "idx2answer": idx2answer,
            "answer2freq": answer2freq,
            "answer2score": answer2score,
            "vocab_size": len(answer2idx),
        }

        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Saving answer vocabulary to: {vocab_path}")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.answer_vocab, f, ensure_ascii=False, indent=2)

    def _dataset_config_for_workers(self, loader_workers):
        """为每个 DataLoader 的 worker 数设置图像 LRU 上限，避免多进程 + persistent_workers 无界缓存 OOM。"""
        cfg = dict(self.config)
        if 'image_cache_max_entries' not in cfg:
            # 每条约数 MB（384 主图 + ae/maml）；过大仍易 OOM，过小则缓存收益低
            cfg['image_cache_max_entries'] = 256 if loader_workers > 0 else 0
        return cfg

    def _init_datasets(self):
        """Initialize dataset objects"""
        # Training set
        if self.train_data:
            self.train_dataset = VQADataset(
                self.data_dir,
                self.train_data,
                self.image_dir,
                transform=self.train_transform,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='train',
                image_size=self.image_size,
                config=self._dataset_config_for_workers(self.num_workers)
            )
            logger.info(f"Training set size: {len(self.train_dataset)}")
        else:
            logger.warning("Training set is empty")
            self.train_dataset = None

        # Validation set
        if self.val_data:
            self.val_dataset_full = VQADataset(
                self.data_dir,
                self.val_data,
                self.image_dir,
                transform=self.transform,  # Validation set uses basic transform
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='val',
                image_size=self.image_size,
                config=self._dataset_config_for_workers(self.val_num_workers)
            )
            val_subset_ratio = float(self.config.get('val_subset_ratio', 1.0))
            if 0 < val_subset_ratio < 1.0:
                np.random.seed(0)  # 固定种子，每 epoch 用同一批验证样本
                subset_size = int(len(self.val_dataset_full) * val_subset_ratio)
                subset_size = max(1, subset_size)
                indices = np.random.choice(len(self.val_dataset_full), subset_size, replace=False)
                self.val_dataset = Subset(self.val_dataset_full, indices)
                logger.info(f"Validation set: {len(self.val_dataset)} samples ({val_subset_ratio*100:.0f}% subset for fast monitoring)")
            else:
                self.val_dataset = self.val_dataset_full
            logger.info(f"Validation set size: {len(self.val_dataset)}")
        else:
            logger.warning("Validation set is empty")
            self.val_dataset = None
            self.val_dataset_full = None

        # Test set
        if self.test_data:
            self.test_dataset = VQADataset(
                self.data_dir,
                self.test_data,
                self.image_dir,
                transform=self.transform,  # Test set uses basic transform
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                answer_vocab=self.answer_vocab,
                mode='test',
                image_size=self.image_size,
                config=self._dataset_config_for_workers(self.num_workers)
            )
            logger.info(f"Test set size: {len(self.test_dataset)}")
        else:
            logger.info("Test set not provided")
            self.test_dataset = None

    def _init_loaders(self):
        """Initialize data loaders"""
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Create training loader
        if self.train_dataset:
            abn_oversample = float(self.config.get('abn_oversample_ratio', 1.0))
            open_oversample = float(self.config.get('open_oversample_ratio', 1.0))
            use_oversample = (abn_oversample > 1.0 or open_oversample > 1.0) and self.train_data
            if use_oversample:
                # Oversampling: abnormality (MedVQA) 或 open (VQA-RAD/SLAKE)
                weights = []
                for entry in self.train_data:
                    at = (entry.get('answer_type') or entry.get('content_type') or entry.get('type') or '').strip().lower()
                    w = 1.0
                    if at == 'abnormality':
                        w = abn_oversample
                    elif at == 'open':
                        w = open_oversample
                    weights.append(w)
                sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights))
                _drop_last = len(self.train_dataset) > self.batch_size  # 小数据集避免 0 batch
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    drop_last=_drop_last,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                    pin_memory=True,
                    prefetch_factor=2 if self.num_workers > 0 else None,
                    persistent_workers=self.num_workers > 0
                )
                msg = []
                if abn_oversample > 1.0:
                    msg.append(f"abnormality={abn_oversample}x")
                if open_oversample > 1.0:
                    msg.append(f"open={open_oversample}x")
                print(f"Oversampling: {' '.join(msg)}")
            else:
                _drop_last = len(self.train_dataset) > self.batch_size  # 小数据集避免 0 batch
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=_drop_last,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                    pin_memory=True,
                    prefetch_factor=2 if self.num_workers > 0 else None,
                    persistent_workers=self.num_workers > 0
                )

        # Create validation loader - val_num_workers=0 避免 OOM（AutoDL 等内存受限环境）
        # val 不 drop_last，可用更大 batch_size 加速
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.val_num_workers,
                persistent_workers=False,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

        # Create test loader - also use single-process mode
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,  # Use single-process mode
                persistent_workers=False,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

    def collate_fn(self, batch):
        # Basic batch data collection
        images = torch.stack([item['image'] for item in batch])
        pos_images = torch.stack([item['pos_image'] for item in batch])
        neg_images = torch.stack([item['neg_image'] for item in batch])
        ae_images = torch.stack([item['ae_image'] for item in batch])
        maml_images = torch.stack([item['maml_image'] for item in batch])
        input_ids = torch.stack([item['question']['input_ids'] for item in batch])
        attention_mask = torch.stack([item['question']['attention_mask'] for item in batch])
        do_input_ids = torch.stack([item['do_question']['input_ids'] for item in batch])
        do_attention_mask = torch.stack([item['do_question']['attention_mask'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        mask = torch.stack([item['mask'] for item in batch])
        answer_indices = torch.tensor([item['answer_idx'] for item in batch], dtype=torch.float32)
        question_texts = [item['question_text'] for item in batch]
        answer_texts = [item['answer_text'] for item in batch]
        answer_types = [item.get('answer_type', item.get('content_type', item.get('type', ''))) for item in batch]
        open_closed = [item.get('open_closed') for item in batch]
        image_paths = [item['image_path'] for item in batch]
        qids = [item.get('qid', str(i)) for i, item in enumerate(batch)]
        concepts = [item.get('concept', 'misc') for item in batch]
        _misc_idx = 9 if self.config.get('use_slake_concept', False) else 4
        concept_indices = [item.get('concept_idx', _misc_idx) for item in batch]

        pattern_embedding = torch.stack([item['pattern_embedding'] for item in batch])
        entity_embedding = torch.stack([item['entity_embedding'] for item in batch])
        
        # Build the complete batch (excluding image batch)
        main_batch = {
            'images': images,
            'pos_images': pos_images,
            'neg_images': neg_images,
            'ae_images': ae_images,
            'maml_images': maml_images,
            'questions': {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            'do_questions': {
                'input_ids': do_input_ids,
                'attention_mask': do_attention_mask
            },
            'targets': targets,
            'answer_indices': answer_indices,
            'question_texts': question_texts,
            'answer_texts': answer_texts,
            'answer_types': answer_types,
            'open_closed': open_closed,
            'image_paths': image_paths,
            'qid': qids,
            'concepts': concepts,
            'concept_indices': concept_indices,
            'mask': mask,
            'pattern_embedding': pattern_embedding,
            'entity_embedding': entity_embedding
        }

        return main_batch

    def get_loaders(self):
        """Get all data loaders"""
        loaders = {}

        if hasattr(self, 'train_loader'):
            loaders['train'] = self.train_loader

        if hasattr(self, 'val_loader'):
            loaders['val'] = self.val_loader

        if hasattr(self, 'test_loader'):
            loaders['test'] = self.test_loader

        return loaders

    def get_answer_vocab(self):
        """Get answer vocabulary"""
        return self.answer_vocab

    def idx2answer(self, idx):
        """Convert index to answer text"""
        if isinstance(idx, int):
            idx_str = str(idx)
        else:
            idx_str = idx

        return self.answer_vocab['idx2answer'].get(idx_str, '<UNK>')

    def answer2idx(self, answer):
        """Convert answer text to index"""
        return self.answer_vocab['answer2idx'].get(answer, self.answer_vocab['answer2idx']['<UNK>'])

    def _load_json_data(self, json_path):

        logger.info(f"Loading data from {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = []
        for item in data:
            if 'answers' in item and isinstance(item['answers'], list):
                # Handle multiple answers
                answers = [a['answer'] for a in item['answers']]
                scores = [a.get('answer_confidence', 1.0) for a in item['answers']]

                # Compute average score for each unique answer
                answer_scores = {}
                for ans, score in zip(answers, scores):
                    if ans not in answer_scores:
                        answer_scores[ans] = []
                    answer_scores[ans].append(score)

                # Merge into unique answer list and their average scores
                unique_answers = list(answer_scores.keys())
                avg_scores = [sum(answer_scores[ans]) / len(answer_scores[ans]) for ans in unique_answers]

                # Normalize scores
                total = sum(avg_scores)
                if total > 0:
                    avg_scores = [s / total for s in avg_scores]

                # Convert to our format
                labels = []
                for ans in unique_answers:
                    if ans in self.answer_vocab['answer2idx']:
                        labels.append(self.answer_vocab['answer2idx'][ans])

                answer_obj = {
                    'labels': labels,
                    'scores': avg_scores
                }
                item['answer'] = answer_obj

            entries.append(item)

        logger.info(f"Loaded {len(entries)} data entries")
        return entries

