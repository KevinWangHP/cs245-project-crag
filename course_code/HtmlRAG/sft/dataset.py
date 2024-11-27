import os
import re
import time
import glob
import json
import random
import pickle
import inspect
import hashlib

import loguru
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import Union, Callable
from functools import cached_property

import json5
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer
)

import recordio
from noisy_utils import add_noise


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_file: str,
                 processor: Callable,  # tokenize处理
                 max_length: int = 4096,  # 最大长度
                 disable_packing: bool = False,  # 是否关闭样本打包，因为部分模型不支持
                 switch_rate: float = 0.0,  # 组合时按话题切换拼接的概率
                 cache_dir: str = 'cached',  # 磁盘缓存目录
                 shm_dir: Union[str, bool, None] = None,  # 内存缓存目录
                 overwrite: bool = False,  # 是否覆盖缓存
                 num_workers: int = 80,  # 多进程处理
                 data_seed: int = 42,  # 数据采样随机数
                 logger=None
                 ):
        self.data_file = data_file
        self.processor = processor
        self.max_length = max_length
        self.disable_packing = disable_packing
        self.switch_rate = switch_rate
        assert 0. <= self.switch_rate <= 1.
        self.overwrite = overwrite
        self.data_seed = data_seed
        self.cache_dir = self.get_cache_dir(cache_dir)
        self.shm_dir = shm_dir
        self.num_workers = num_workers
        self.rng = random.Random(self.data_seed)
        self._logger = logger
        self.sample_length = {
            "16k": 0,
            "32k": 0,
            "64k": 0,
            "128k": 0,
            "256k": 0,
            "more": 0
        }

    def __getitem__(self, idx):
        groups = []
        for i in self.packed[idx]:
            if groups and self.rng.random() < self.switch_rate:
                groups[-1].append(i)
            else:
                groups.append([i])

        input_ids, labels, seqlens, position_ids = [], [], [], []
        seqlen = 0
        for group in groups:
            for i in group:
                data = self.tokenized[i]
                if len(data['input_ids']) == 0:
                    continue
                input_ids.extend(data['input_ids'])
                labels.extend(data['labels'])
                seqlen += len(data['input_ids'])
            if seqlen < self.max_length:
                seqlens.append(seqlen)
            position_ids.extend(list(range(len(input_ids) - len(position_ids))))
        sample_len = len(input_ids)
        if sample_len < 16384:
            self.sample_length["16k"] += 1
        elif sample_len < 32768:
            self.sample_length["32k"] += 1
        elif sample_len < 65536:
            self.sample_length["64k"] += 1
        elif sample_len < 131072:
            self.sample_length["128k"] += 1
        elif sample_len < 262144:
            self.sample_length["256k"] += 1
        else:
            self.sample_length["more"] += 1
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        position_ids = position_ids[:self.max_length]
        pad_length = self.max_length - len(input_ids)
        if pad_length:
            input_ids.extend([0] * pad_length)
            labels.extend([-100] * pad_length)
            position_ids.extend([0] * pad_length)
        seqlens.extend([0] * (self.max_length - len(seqlens)))
        return {
            'input_ids': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(labels),
            'position_ids': torch.LongTensor(position_ids),
            'seqlens': torch.LongTensor(seqlens)
        }

    def __len__(self):
        return len(self.packed)

    def update(self, epoch):
        rng = random.Random(self.data_seed + epoch + 1)

        def random_chunks():
            indice, start = [], 0
            for end, rate in self.tokenized[-1]:
                ids = list(range(start, end))
                start = end
                sample_num = int(len(ids) * rate)
                while sample_num >= len(ids) > 0:
                    indice.extend(ids)
                    sample_num -= len(ids)
                if sample_num:
                    offset = epoch * sample_num % len(ids)  # 根据epoch计算采样位置
                    ids = ids + ids
                    indice.extend(ids[offset: (offset + sample_num)])

            rng.shuffle(indice)
            step = (len(indice) - 1) // self.num_workers + 1
            return [indice[i * step: (i + 1) * step] for i in range(self.num_workers)]

        rank = dist.get_rank() if dist.is_initialized() else 0
        cache_file = os.path.join(self.cache_dir, f'packed-{epoch}.rec')
        if not os.path.exists(cache_file) or self.overwrite:
            if rank == 0:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with recordio.RecordIO(cache_file, 'w') as rec:
                    desc = f'[Rank {rank}][Epoch {epoch}] split_and_pack'
                    with mp.Pool(self.num_workers) as pool, tqdm(desc=desc) as pbar:
                        split_and_pack = partial(self._split_and_pack, rng=rng)
                        for packed in pool.map(split_and_pack, random_chunks()):
                            for data in packed:
                                rec.append(pickle.dumps(data))
                                pbar.update()
                    loguru.logger.info(f'[Rank {rank}][Epoch {epoch}] Save #{len(rec)} samples to {cache_file}')
            if dist.is_initialized():
                dist.barrier()
                # in case some stupid device that is too damn slow to sync file
                while not os.path.exists(cache_file):
                    time.sleep(1)
        self.packed = recordio.Record(cache_file, auto_pickle=True, shm_dir=self.shm_dir)
        #rng = random.Random(self.data_seed + epoch + 1)
        #rng.shuffle(self.packed)
        loguru.logger.info(f'[Rank {rank}][Epoch {epoch}] Load #{len(self.packed)} samples from {cache_file}')

    @cached_property
    def tokenized(self):
        cache_file = os.path.join(self.cache_dir, 'tokenized.rec')
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not os.path.exists(cache_file):
            rng = random.Random(self.data_seed)
            examples, ori_infos = self._load_examples(rng)
            auto_pickle, self.processor.auto_pickle = self.processor.auto_pickle, True
            with mp.Pool(self.num_workers) as pool:
                encoded = pool.map(self.processor, examples)
            self.processor.auto_pickle = auto_pickle
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with recordio.RecordIO(cache_file, 'w') as rec:
                unique, infos = set(), []
                start, offset, total_sample_num = 0, 0, 0
                for file, end, rate in ori_infos:
                    buffers = []
                    no_loss_num, dedup_num = 0, 0
                    for buffer in encoded[start:end]:
                        if buffer is None:
                            no_loss_num += 1
                        elif buffer in unique:
                            dedup_num += 1
                        else:
                            buffers.append(buffer)
                    sample_num = int(len(buffers) * rate)
                    self._log_memory(f'Tokenized {file}: total={end - start}, noloss={no_loss_num}, '
                                     f'deduped={dedup_num}, sample {len(buffers)} -> {sample_num}')
                    start = end
                    if sample_num:
                        rng.shuffle(buffers)
                        for buffer in buffers:
                            rec.append(buffer)
                            unique.add(buffer)  # 只进行跨文件去重
                        offset = len(rec)
                        total_sample_num += sample_num
                        infos.append((offset, rate))

                loguru.logger.info(f'Save #{len(rec)} samples to {cache_file}, total sample {total_sample_num}')
                rec.append(pickle.dumps(infos))

            with open(os.path.join(self.cache_dir, 'config.json'), 'w') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)

        elif rank == 0 and not hasattr(self, 'memory'):
            self.memory = json.load(open(os.path.join(self.cache_dir, 'config.json')))
            for message in self.memory:
                loguru.logger.info(message)
        return recordio.Record(cache_file, auto_pickle=True, shm_dir=self.shm_dir)

    def _log(self, message):
        if self._logger is None:
            loguru.logger.info(message, flush=True)
        else:
            self._logger.info(message)

    def _log_memory(self, message):
        if not hasattr(self, 'memory'):
            self.memory = []
        self.memory.append(message)
        loguru.logger.info(message)

    def _load_examples(self, rng):
        data_info = json5.load(open(self.data_file))
        train_files = data_info['train_files']
        data_root = train_files['root']
        examples, infos = [], []  # 记录文件粒度的分组信息
        for pattern, rate in train_files['sample_rate'].items():
            for file in sorted(glob.glob(os.path.join(data_root, pattern + '*.json*'))):
                try:
                    all_data_list = list(map(json.loads, open(file)))
                except:
                    all_data_list = json.load(open(file))

                data_list = [
                    data for data in all_data_list
                    if self.processor.valid(data, rng)
                ]
                invalid_num = len(all_data_list) - len(data_list)
                sample_num = int(len(data_list) * rate)
                loguru.logger.info(f'Loading {file}: total={len(all_data_list)}, invalid={invalid_num}, '
                                 f'sample {len(data_list)} -> {sample_num}')

                if sample_num > 0:
                    examples.extend(data_list)
                    infos.append((file, len(examples), rate))

        # self._log_memory(f'Loaded {len(examples)} examples from {self.data_file}')
        # exit(-1)
        return examples, infos

    def _split_and_pack(self, indice, rng, max_try=100):
        lengths = [len(self.tokenized[i]['input_ids']) for i in indice]
        candidates = set(range(len(indice)))
        packed = []
        for i, input_length in sorted(enumerate(lengths), key=lambda d: d[1], reverse=True):
            if i not in candidates:
                continue
            candidates.remove(i)
            collection = [indice[i]]
            if input_length < self.max_length and not self.disable_packing:
                rand_candidates = list(candidates)
                rng.shuffle(rand_candidates)
                ntry = 0
                for j in rand_candidates:
                    if input_length + lengths[j] <= self.max_length:
                        input_length += lengths[j]
                        collection.append(indice[j])
                        candidates.remove(j)
                        if input_length == self.max_length:
                            break
                        else:
                            ntry = 0
                    else:
                        ntry += 1
                        if ntry == max_try:
                            break
                rng.shuffle(collection)
            packed.append(collection)
        return packed

    def get_cache_dir(self, cache_dir):
        if dist.is_initialized():
            dist.barrier()

        if not os.path.exists(cache_dir) or all([
            os.path.exists(os.path.join(cache_dir, file))
            for file in ('config.json', 'tokenized.rec')
        ]):
            return cache_dir

        def get_md5(content):
            if isinstance(content, str):
                content = content.encode()
            assert isinstance(content, bytes)
            return hashlib.md5(content).hexdigest()

        meta_info = {}
        meta_info['config'] = get_md5(open(self.data_file).read())
        meta_info['processor'] = get_md5(str(self.processor))
        meta_info['code'] = get_md5(json.dumps({
            'dataset': get_md5(inspect.getsource(self.__class__)),
            'processor': inspect.getsource(self.processor.__class__)
        }))
        data_info = json5.load(open(self.data_file))
        train_files = data_info['train_files']
        data_root = train_files['root']
        meta_info['train_files'] = {}
        for pattern in train_files['sample_rate']:
            for file in sorted(glob.glob(os.path.join(data_root, pattern + '*.json*'))):
                meta_info['train_files'][file] = get_md5(str(os.stat(file).st_size))
        meta_info['disable_packing'] = self.disable_packing
        meta_info['max_length'] = self.max_length
        meta_info['data_seed'] = self.data_seed
        meta_md5 = get_md5(json.dumps(meta_info))
        cache_dir = os.path.join(cache_dir, meta_md5)
        return cache_dir

class Processor:
    def __init__(self):
        self.max_length = 1024
        self.role_tokens = {
        }
        self.sep_tokens = {
        }
        self.ignore_token = -100
        self.truncated = 0
        self.tokenizer = None

    def tokenize(self, text, max_length=None):
        """在tokenizer外部对sep_tokens进行保护映射"""
        max_length = max_length or self.max_length
        if max_length <= 0:
            return []
        pieces = [text.strip()]

        pieces_ids = self.tokenizer(pieces, max_length=max_length, truncation=True, add_special_tokens=False)['input_ids']
        input_ids = pieces_ids[0]
        if len(input_ids) > max_length:
            self.truncated += 1
            loguru.logger.warning(f"input_ids: {len(input_ids)}, max_length: {max_length}")
        return input_ids[:max_length]

    def stop_token(self, role, learnable):
        """停止 token，有两个关键作用:
        1. 作为 assistant 的 next_token, 指导 LLM 停止生成
        2. 指示外围系统接下来应该触发哪个 role 的逻辑，比如 user 或者 function
        简单起见，user role 的停止 token 复用 eos_token_id
        """
        if learnable == True:
            if role == 'user':
                return self.tokenizer.eos_token_id
            else:
                return self.role_tokens[role]
        else:
            return self.ignore_token

    def __str__(self):
        pdir, base = os.path.split(os.path.abspath(self.tokenizer.name_or_path))
        if pdir != '/' and base.startswith('checkpoint-'):
            name = os.path.basename(pdir)
        else:
            name = base
        return '{}(\n{}\n)'.format(self.__class__.__name__, '\n'.join([
            '  ' + line for line in '\n'.join([
                f'tokenizer="{name}", max_length={self.max_length},',
                f'role_tokens={json.dumps(self.role_tokens, indent=2)},',
                f'sep_tokens={json.dumps(self.sep_tokens, indent=2)},',
                f'ignore_token={self.ignore_token}'
            ]).split('\n')
        ]))

    def valid(self, example, rng=random):
        dirty_words = ['openai', 'chatgpt', 'gpt4', 'gpt3']
        for i, message in enumerate(example['messages']):
            if message['role'] == 'system':
                assert i == 0, f'System role can only act as the first message (i={i}).'
            if message['role'] == 'user_system':
                assert i in (0, 1), f'User system role can only act as the first or second message (i={i}).'
                remove_prefix = '你是TeacherGPT，'
                if message['content'].startswith(remove_prefix):
                    message['content'] = message['content'][len(remove_prefix):]
            if message['role'] != 'assistant':
                dirty_words = [
                    dirty_word for dirty_word in dirty_words
                    if dirty_word not in message['content'].lower()
                ]
                if message['role'] == 'user' and rng.random() <= 0.01:
                    message['content'] = add_noise(message['content'], rng)
            else:
                if any([
                    dirty_word in message['content'].lower()
                    for dirty_word in dirty_words + ['teachergpt']
                ]):
                    return False
        return True


class BaichuanProcessor:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 role_tokens=None,
                 ignore_token=-100,
                 auto_pickle=False
                 ):
        if role_tokens is None:
            role_tokens = {
                'system': 194,
                'user': 198,
                'assistant': 199,
                'user_system': 197,
                'function': 370,
                'code': 371,
                '<function_calling>': 373,
                '<calc_start>': 380,
                '<calc_end>': 381
            }
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.role_tokens, self.sep_tokens = {}, {}
        for token, token_id in role_tokens.items():
            if token[0] + token[-1] == '<>':
                self.sep_tokens[token] = token_id
            else:
                self.role_tokens[token] = token_id
        self.ignore_token = ignore_token
        self.sep_pattern = re.compile('|'.join(self.sep_tokens.keys()))
        self.auto_pickle = auto_pickle

    def __call__(self, example):
        ## 标记 message 的训练方式，只有 assistant 默认是可学习的，可选状态
        # - true: 参与训练，加 eos。例如：正常 assistant
        # - false: 不参与训练。例如：一个错误的 assistant message 仅作为多轮 history
        # - incomplete: 参与训练，不加 eos。例如：当前轮生成中断，下一轮继续生成。
        learnable = False

        input_ids, labels = [], []
        for message in example['messages']:
            input_ids.append(self.role_tokens[message['role']])
            labels.append(self.stop_token(message['role'], learnable))
            #tokens = self.tokenize(message['content'], self.max_length - len(input_ids))
            tokens = self.tokenize(message['content'], self.max_length)
            if len(tokens) + len(input_ids) >= self.max_length:
                break
            input_ids.extend(tokens)
            learnable = message.get('learnable', message['role'] == 'assistant')
            labels.extend(tokens if learnable else [self.ignore_token] * len(tokens))
        if len(input_ids) < self.max_length and learnable == True:
            # input_ids.append(self.stop_token('user', learnable))
            # labels.append(self.stop_token('user', learnable))
            pass
        elif len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        if any([label >= 0 for label in labels]):
            encoded = {'input_ids': input_ids, 'labels': labels}
            return pickle.dumps(encoded) if self.auto_pickle else encoded

    def tokenize(self, text, max_length=None):
        """在tokenizer外部对sep_tokens进行保护映射"""
        max_length = max_length or self.max_length
        if max_length <= 0:
            return []

        offset, pieces, sep_ids = 0, [], []
        for match in self.sep_pattern.finditer(text):
            pieces.append(text[offset:match.start()])
            sep_ids.append(self.sep_tokens[text[match.start():match.end()]])
            offset = match.end()
        pieces.append(text[offset:])

        pieces_ids = self.tokenizer(pieces, max_length=max_length, truncation=True)['input_ids']
        input_ids = pieces_ids[0]
        for sep_id, ids in zip(sep_ids, pieces_ids[1:]):
            input_ids.append(sep_id)
            input_ids.extend(ids)
        return input_ids[:max_length]

    def stop_token(self, role, learnable):
        """停止 token，有两个关键作用:
        1. 作为 assistant 的 next_token, 指导 LLM 停止生成
        2. 指示外围系统接下来应该触发哪个 role 的逻辑，比如 user 或者 function
        简单起见，user role 的停止 token 复用 eos_token_id
        """
        if learnable == True:
            if role == 'user':
                return self.tokenizer.eos_token_id
            else:
                return self.role_tokens[role]
        else:
            return self.ignore_token

    def __str__(self):
        pdir, base = os.path.split(os.path.abspath(self.tokenizer.name_or_path))
        if pdir != '/' and base.startswith('checkpoint-'):
            name = os.path.basename(pdir)
        else:
            name = base
        return '{}(\n{}\n)'.format(self.__class__.__name__, '\n'.join([
            '  ' + line for line in '\n'.join([
                f'tokenizer="{name}", max_length={self.max_length},',
                f'role_tokens={json.dumps(self.role_tokens, indent=2)},',
                f'sep_tokens={json.dumps(self.sep_tokens, indent=2)},',
                f'ignore_token={self.ignore_token}'
            ]).split('\n')
        ]))




class GLMProcessor(Processor):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 role_tokens=None,
                 ignore_token=-100,
                 auto_pickle=False
                 ):
        super().__init__()
        if role_tokens is None:
            role_tokens = {
                'system': 151335,
                'user': 151336,
                'assistant': 151337,
                "[MASK]": 151330,
                "[gMASK]": 151331,
                "[sMASK]": 151332,
                "<sop>": 151333,
            }
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.role_tokens, self.sep_tokens, self.mask_tokens = {}, {}, {}
        for token, token_id in role_tokens.items():
            if token[0] + token[-1] in ['<>']:
                self.sep_tokens[token] = token_id
            elif token[0] + token[-1] in ['[]']:
                self.mask_tokens[token] = token_id
            else:
                self.role_tokens[token] = token_id
        self.ignore_token = ignore_token
        self.sep_pattern = re.compile('|'.join(self.sep_tokens.keys()))
        # loguru.logger.info(f"sep_pattern: {self.sep_pattern}")
        self.auto_pickle = auto_pickle
        self.truncated = 0

    def __call__(self, example):
        ## 标记 message 的训练方式，只有 assistant 默认是可学习的，可选状态
        # - true: 参与训练，加 eos。例如：正常 assistant
        # - false: 不参与训练。例如：一个错误的 assistant message 仅作为多轮 history
        # - incomplete: 参与训练，不加 eos。例如：当前轮生成中断，下一轮继续生成。
        learnable = False

        input_ids, labels = [self.mask_tokens["[gMASK]"], self.sep_tokens["<sop>"]], [self.ignore_token,
                                                                                      self.ignore_token]
        for message in example['messages']:
            input_ids.append(self.role_tokens[message['role']])
            labels.append(self.stop_token(message['role'], learnable))
            #tokens = self.tokenize(message['content'], self.max_length - len(input_ids))
            tokens = self.tokenize(message['content'], self.max_length)
            if len(tokens) + len(input_ids) >= self.max_length:
                break
            input_ids.extend(tokens)
            learnable = message.get('learnable', message['role'] == 'assistant')
            labels.extend(tokens if learnable else [self.ignore_token] * len(tokens))
        if len(input_ids) < self.max_length and learnable:
            input_ids.append(self.stop_token('user', learnable))
            labels.append(self.stop_token('user', learnable))
        elif len(input_ids) > self.max_length:
            loguru.logger.warning(f'Input length {len(input_ids)} exceeds max length {self.max_length}')
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        if any([label >= 0 for label in labels]):
            encoded = {'input_ids': input_ids, 'labels': labels}
            return pickle.dumps(encoded) if self.auto_pickle else encoded

    def tokenize(self, text, max_length=None):
        """在tokenizer外部对sep_tokens进行保护映射"""
        max_length = max_length or self.max_length
        if max_length <= 0:
            return []

        offset, pieces, sep_ids = 0, [], []
        for match in self.sep_pattern.finditer(text):
            sep_text = text[match.start():match.end()]
            assert sep_text in self.sep_tokens, f"sep_text: {sep_text}"
            pieces.append(text[offset:match.start()])
            sep_ids.append(self.sep_tokens[sep_text])
            offset = match.end()
        pieces.append(text[offset:])

        pieces_ids = self.tokenizer(pieces, max_length=max_length, truncation=True)['input_ids']
        input_ids = pieces_ids[0]
        for sep_id, ids in zip(sep_ids, pieces_ids[1:]):
            input_ids.append(sep_id)
            input_ids.extend(ids)
        if len(input_ids) > max_length:
            self.truncated += 1
            loguru.logger.warning(f"input_ids: {len(input_ids)}, max_length: {max_length}")
        return input_ids[:max_length]


class Phi3Processor(Processor):
    def __init__(self, tokenizer: PreTrainedTokenizer, role_tokens=None, ignore_token=-100, auto_pickle=False):
        super().__init__()
        self.role_tokens = {
            "assistant": 32001, # "<|assistant|>"
            "system": 32006, # "<|system|>"
            "user": 32010 # "<|user|>"
        }
        self.sep_tokens = {
            "<unk>": 0,
            "<s>": 1,

        }
        self.end_tokens = {
            "<|end|>": 32007,
            "<|endoftext|>": 32000,
        }
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.ignore_token = ignore_token
        self.sep_pattern = re.compile('|'.join(self.sep_tokens.keys()))
        # loguru.logger.info(f"sep_pattern: {self.sep_pattern}")
        self.auto_pickle = auto_pickle
        self.truncated = 0

    def __call__(self, example):
        ## 标记 message 的训练方式，只有 assistant 默认是可学习的，可选状态
        # - true: 参与训练，加 eos。例如：正常 assistant
        # - false: 不参与训练。例如：一个错误的 assistant message 仅作为多轮 history
        # - incomplete: 参与训练，不加 eos。例如：当前轮生成中断，下一轮继续生成。
        learnable = False
        #. "<|system|>\nYou are a helpful assistant.<|end|>\n"
        input_ids= [32006, 887, 526, 263, 8444, 20255, 29889, 32007]
        labels=[self.ignore_token] * len(input_ids)
        for message in example['messages']:
            #. add role token
            input_ids.append(self.role_tokens[message['role']])
            labels.append(self.stop_token(message['role'], learnable))
            #. tokenize message content
            tokens = self.tokenize(message['content'], self.max_length)
            # . add <|end|> token to the end of each message
            tokens.append(self.end_tokens["<|end|>"])
            if len(tokens) + len(input_ids) >= self.max_length:
                break
            input_ids.extend(tokens)
            #. control learnable. if message role is assistant, learnable is True
            learnable = message.get('learnable', message['role'] == 'assistant')
            labels.extend(tokens if learnable else [self.ignore_token] * len(tokens))
        if len(input_ids) < self.max_length and learnable:
            input_ids.append(self.stop_token('user', learnable))
            labels.append(self.stop_token('user', learnable))
        elif len(input_ids) > self.max_length:
            loguru.logger.warning(f'Input length {len(input_ids)} exceeds max length {self.max_length}')
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        if any([label >= 0 for label in labels]):
            encoded = {'input_ids': input_ids, 'labels': labels}
            return pickle.dumps(encoded) if self.auto_pickle else encoded



class LlamaProcessor(Processor):
    def __init__(self, tokenizer: PreTrainedTokenizer, ignore_token=-100, auto_pickle=False):
        """
        128000 <|begin_of_text|>
        128001 <|end_of_text|>
        128002 <|reserved_special_token_0|>
        128003 <|reserved_special_token_1|>
        128004 <|finetune_right_pad_id|>
        128005 <|reserved_special_token_2|>
        128006 <|start_header_id|>
        128007 <|end_header_id|>
        128008 <|eom_id|>
        128009 <|eot_id|>
        """
        super().__init__()
        self.role_tokens = {
            "assistant": 78191, # "assistant"
            "system": 9125, # "system"
            "user": 882 # "user"
        }
        self.sep_tokens = {
            "<|begin_of_text|>": 128000,
        }
        self.end_tokens = {
            "<|begin_of_text|>": 128000,
            "<|start_header_id>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
            "\n\n": 271
        }
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.ignore_token = ignore_token
        self.auto_pickle = auto_pickle
        self.truncated = 0
        self.default_system_ids=[128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724,
                                2696, 25, 220, 972, 5020, 220, 2366, 19, 271, 128009]

    def __call__(self, example):
        ## 标记 message 的训练方式，只有 assistant 默认是可学习的，可选状态
        # - true: 参与训练，加 eos。例如：正常 assistant
        # - false: 不参与训练。例如：一个错误的 assistant message 仅作为多轮 history
        # - incomplete: 参与训练，不加 eos。例如：当前轮生成中断，下一轮继续生成。
        learnable = False
        #. "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 18 Oct 2024\n\n<|eot_id|>"
        # input_ids= [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724,
        #             2696, 25, 220, 972, 5020, 220, 2366, 19, 271, 128009]
        input_ids= [128000]
        labels=[]
        if "system" not in [message['role'] for message in example['messages']]:
            input_ids.extend(self.default_system_ids)
        else:
            assert example['messages'][0]['role'] == 'system', "The first message should be system"
            input_ids.extend([self.end_tokens["<|start_header_id>"], self.role_tokens["system"], self.end_tokens["<|end_header_id|>"], self.end_tokens["\n\n"]])
            tokens = self.tokenize(example['messages'][0]['content'], self.max_length)
            input_ids.extend(tokens)
            input_ids.extend([self.end_tokens["\n\n"], self.end_tokens["<|eot_id|>"]])
            #  remove the first message
            example['messages'] = example['messages'][1:]
        labels.extend([self.ignore_token] * len(input_ids))

        for message in example['messages']:
            #. add role token
            input_ids.extend([self.end_tokens["<|start_header_id>"], self.role_tokens[message['role']], self.end_tokens["<|end_header_id|>"], self.end_tokens["\n\n"]])
            labels.extend([self.end_tokens["<|start_header_id>"], self.role_tokens[message['role']], self.end_tokens["<|end_header_id|>"], self.end_tokens["\n\n"]])
            #. tokenize message content
            tokens = self.tokenize(message['content'], self.max_length)
            # . add <|end|> token to the end of each message
            tokens.extend([self.end_tokens["\n\n"], self.end_tokens["<|eot_id|>"]])
            if len(tokens) + len(input_ids) >= self.max_length:
                break
            input_ids.extend(tokens)
            #. control learnable. if message role is assistant, learnable is True
            learnable = message.get('learnable', message['role'] == 'assistant')
            labels.extend(tokens if learnable else [self.ignore_token] * len(tokens))
        if len(input_ids) < self.max_length and learnable:
            # input_ids.append(self.stop_token('user', learnable))
            # labels.append(self.stop_token('user', learnable))
            pass
        elif len(input_ids) > self.max_length:
            loguru.logger.warning(f'Input length {len(input_ids)} exceeds max length {self.max_length}')
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        if any([label >= 0 for label in labels]):
            encoded = {'input_ids': input_ids, 'labels': labels}
            # print(self.tokenizer.decode(input_ids))
            return pickle.dumps(encoded) if self.auto_pickle else encoded


class Qwen2Processor(Processor):
    def __init__(self, tokenizer: PreTrainedTokenizer, ignore_token=-100, auto_pickle=False):
        """
        128000 <|begin_of_text|>
        128001 <|end_of_text|>
        128002 <|reserved_special_token_0|>
        128003 <|reserved_special_token_1|>
        128004 <|finetune_right_pad_id|>
        128005 <|reserved_special_token_2|>
        128006 <|start_header_id|>
        128007 <|end_header_id|>
        128008 <|eom_id|>
        128009 <|eot_id|>
        """
        super().__init__()
        self.role_tokens = {
            "assistant": 78191,  # "assistant"
            "system": 9125,  # "system"
            "user": 882  # "user"
        }
        self.sep_tokens = {
            "<|begin_of_text|>": 128000,
        }
        self.end_tokens = {
            "<|begin_of_text|>": 128000,
            "<|start_header_id>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length
        self.ignore_token = ignore_token
        self.auto_pickle = auto_pickle
        self.truncated = 0

    def __call__(self, example):
        ## 标记 message 的训练方式，只有 assistant 默认是可学习的，可选状态
        # - true: 参与训练，加 eos。例如：正常 assistant
        # - false: 不参与训练。例如：一个错误的 assistant message 仅作为多轮 history
        # - incomplete: 参与训练，不加 eos。例如：当前轮生成中断，下一轮继续生成。
        learnable = False
        # . "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 18 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        input_ids = [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724,
                     2696, 25, 220, 972, 5020, 220, 2366, 19, 271, 128009, 128006, 882, 128007, 271]
        labels = [self.ignore_token] * len(input_ids)
        for message in example['messages']:
            # . add role token
            input_ids.extend([self.end_tokens["<|start_header_id>"], self.role_tokens[message['role']],
                              self.end_tokens["<|end_header_id|>"]])
            labels.extend([self.end_tokens["<|start_header_id>"], self.role_tokens[message['role']],
                           self.end_tokens["<|end_header_id|>"]])
            # . tokenize message content
            tokens = self.tokenize(message['content'], self.max_length)
            # . add <|end|> token to the end of each message
            tokens.append(self.end_tokens["<|eot_id|>"])
            if len(tokens) + len(input_ids) >= self.max_length:
                break
            input_ids.extend(tokens)
            # . control learnable. if message role is assistant, learnable is True
            learnable = message.get('learnable', message['role'] == 'assistant')
            labels.extend(tokens if learnable else [self.ignore_token] * len(tokens))
        if len(input_ids) < self.max_length and learnable:
            input_ids.append(self.stop_token('user', learnable))
            labels.append(self.stop_token('user', learnable))
        elif len(input_ids) > self.max_length:
            loguru.logger.warning(f'Input length {len(input_ids)} exceeds max length {self.max_length}')
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        if any([label >= 0 for label in labels]):
            encoded = {'input_ids': input_ids, 'labels': labels}
            return pickle.dumps(encoded) if self.auto_pickle else encoded


if __name__ == '__main__':
    # model_path = '../../../huggingface/Phi-3.5-mini-instruct/'
    model_path = '../../../model/Llama-3.2-1B-Instruct/'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    print(f"model max length: {tokenizer.model_max_length}")
    for processor_class in [LlamaProcessor]:
        processor = processor_class(tokenizer)
        dataset = SupervisedDataset(
            'experiments/v1107.json5',
            processor,
            max_length=131072,
            overwrite=True,
            disable_packing=True,
        )
        #  calulate the average length of the samples
        sample_len = []
        for i in range(1):
            dataset.update(i)

            for item in dataset:
                sample_len.append(len(item['input_ids']))
            print(f"truncate: {dataset.sample_length}")
    # from IPython import embed
    #
    # embed()
    # exit()
