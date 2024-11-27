# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json

import loguru
import math
import pathlib
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
import deepspeed
import transformers
from transformers.training_args import TrainingArguments
import sys
sys.path.append("./")
from utils import init_logger, add_custom_callback, DistributedSampler
from dataset import GLMProcessor, SupervisedDataset, Phi3Processor, LlamaProcessor
import torch.distributed as dist
from transformers.trainer_utils import seed_worker
deepspeed.init_distributed()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    role_config: str = field(default=None)

@dataclass
class DataArguments:
    data_file: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    disable_packing: bool = field(
        default=False,
        metadata={"help": "Whether disable example packing, heavily slowdown training speed."})
    switch_rate: float = field(default=0.,
                               metadata={"help": "组合时按话题切换拼接的概率"})
    data_cache_dir: str = field(default=None,
                                metadata={"help": "Path to the save data cache."})
    overwrite_data_cache: bool = field(default=False,
                                       metadata={"help": "Overwrite the data cache."})
    data_shm_dir: str = field(default=None,
                              metadata={"help": "Path to the share memory cache, optional True or path."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    loss_average: str = field(
        default="token",
        metadata={
            "help": "Loss averages with the granularity of `token`, `response` or `mixture`"
        }
    )
    seq_parallel_size: int = field(default=1, metadata={"help": "序列并行的size"})


class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        token_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.loss_average == 'token':
            loss = token_loss
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            labels = inputs['labels'].to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            # average on response
            response_loss = []
            for row in loss:
                #  the condition checks for a zero followed by a positive number
                #  shift the index by 1 to get the index of the positive number
                start = (torch.where((row[:-1] == 0) & (row[1:] > 0))[0] + 1).tolist()
                #  the condition checks for a positive number followed by a zero
                #  shift the index by 1 to get the index of the zero
                end = (torch.where((row[:-1] > 0) & (row[1:] == 0))[0] + 1).tolist()
                if len(end) < len(start):
                    end.append(len(row))
                assert len(start) == len(end)
                #  calculate the mean of the response seperated by zeros
                for s, e in zip(start, end):
                    response_loss.append(row[s:e].mean())
            response_loss = torch.stack(response_loss).mean()
            if self.args.loss_average == 'response':
                loss = response_loss
            else:
                loss = (token_loss + response_loss) / 2
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self) -> DataLoader:
        """
        如果存在序列并行，直接用distibutedsampler根据划分情况直接
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory
            #"persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # 同一个seq 内部采用相同的batch
            loguru.logger.info(f"dataloader_{dist.get_rank()}",dist.get_world_size()//self.args.seq_parallel_size,dist.get_rank() // self.args.seq_parallel_size, self.args.dataloader_drop_last)
            dataloader_params["sampler"] = DistributedSampler(train_dataset,
                                                                               num_replicas=dist.get_world_size()//self.args.seq_parallel_size, 
                                                                               rank=dist.get_rank() // self.args.seq_parallel_size,
                                                                               shuffle=True,
                                                                               drop_last=self.args.dataloader_drop_last,
                                                                               seed=self.args.seed)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)

def train():
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger = init_logger(
        os.path.join(training_args.output_dir, 'train.log'),
        training_args.local_rank
    )
    logger.info(f'model args: {model_args}')
    logger.info(f'data args: {data_args}')
    logger.info(f'training args: {training_args}')
    #初始化序列并行group划分
    # initialize_seq_parallel(training_args.seq_parallel_size)
    os.environ["SEQ_PARALLEL_SIZE"] = str(training_args.seq_parallel_size)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        # empty_init=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir
    )
    if "glm" in model_args.model_name_or_path:
        processor_class=GLMProcessor
    elif "Phi" in model_args.model_name_or_path:
        processor_class=Phi3Processor
    elif "Llama" in model_args.model_name_or_path:
        processor_class=LlamaProcessor
    else:
        raise ValueError(f"No processor supports model {model_args.model_name_or_path}")

    if model_args.role_config:
        processor = processor_class(
            tokenizer=tokenizer,
            role_tokens=json.load(open(model_args.role_config))
        )
    else:
        processor = processor_class(tokenizer=tokenizer)
    logger.info(f'processor: {processor}')
    dataset = SupervisedDataset(
        data_file=data_args.data_file,
        processor=processor,
        max_length=training_args.model_max_length,
        disable_packing=data_args.disable_packing,
        switch_rate=data_args.switch_rate,
        cache_dir=data_args.data_cache_dir or os.path.join(training_args.output_dir, 'cached'),
        overwrite=data_args.overwrite_data_cache,
        shm_dir=data_args.data_shm_dir,
        logger=logger
    )
    for epoch in range(math.ceil(training_args.num_train_epochs)):
        dataset.update(epoch)
    dataset.update(0)
    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=dataset,
                            tokenizer=tokenizer)
    add_custom_callback(trainer, logger)

    if training_args.resume_from_checkpoint and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    train()
