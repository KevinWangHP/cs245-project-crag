import os
import torch
import logging
import transformers
import torch.distributed as dist
import torch
import math

# global var
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_SIZE = 1

def init_logger(fpath='', local_rank=0):
    if transformers.trainer_utils.is_main_process(local_rank):
        if fpath:
            if os.path.dirname(fpath):
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
            file_handler = logging.FileHandler(fpath, mode='a')  # to file
            transformers.logging.add_handler(file_handler)
        transformers.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()  # reduce
    transformers.logging.enable_explicit_format()
    return transformers.logging.get_logger()

class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def set_epoch(self, epoch):
        # 重载Sample 保证每个epoch dataset更新后sampler 重新更新
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        super().set_epoch(epoch)

def add_custom_callback(trainer, logger):
    if 'PrinterCallback' in trainer.callback_handler.callback_list:
        trainer.pop_callback(transformers.PrinterCallback)
    trainer.add_callback(LogCallback(logger))
    logger.info('Add custom LogCallback')
    trainer.add_callback(DatasetUpdateCallback(trainer))
    logger.info('Add custom DatasetUpdateCallback')
    trainer.add_callback(SaveDiskCallback())
    logger.info('Add custom SaveDiskCallback')
    logger.info(f"trainer's callbacks: {trainer.callback_handler.callback_list}")


class LogCallback(transformers.TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints with logger.
    """
    def __init__(self, logger, exclude=('total_flos', 'epoch')):
        self.logger = logger
        self.exclude = exclude

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            self.logger.info(''.join([
                f"[global_steps={state.global_step}]",
                f"[epochs={logs['epoch']}]",
                ','.join(f'{k}={v}' for k, v in logs.items()
                         if k not in self.exclude)
            ]))


class DatasetUpdateCallback(transformers.TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control,train_dataloader, **kwargs):
        self.trainer.train_dataset.update(int(state.epoch))
        train_dataloader.sampler.set_epoch(int(state.epoch))


class SaveDiskCallback(transformers.TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.local_rank != 0:
            return

        for ckpt in os.listdir(args.output_dir):
            # remove out-of-date deepspeed checkpoints
            if ckpt.startswith('checkpoint-') and not ckpt.endswith(f'-{state.global_step}'):
                for pattern in ['global_step*', '*.pth']:
                    os.system("rm -rf " + os.path.join(args.output_dir, ckpt, pattern))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and False:
            for pattern in ['global_step*', '*.pth']:
                os.system("rm -rf " + os.path.join(args.output_dir, "checkpoint-*", pattern))


def register_nan_hook(model):
    torch.autograd.set_detect_anomaly(True)

    def add_module_name(module):
        for name, sub_module in module.named_modules():
            sub_module.name = name

    def add_check_nan_hook(module):
        def check_nan(module, inputs, outputs):
            any_nan = False
            for i, tensor in enumerate(inputs):
                if isinstance(tensor, torch.Tensor) and tensor.isnan().any():
                    print(f'module {module.name} contains nan in its {i}th input.')
                    any_nan = True
            for i, tensor in enumerate(outputs):
                if isinstance(tensor, torch.Tensor) and tensor.isnan().any():
                    print(f'module {module.name} contains nan in its {i}th output.')
                    any_nan = True
            if any_nan:
                if torch.distributed.get_rank() == 0:
                    torch.save({
                        'state_dict': module.state_dict(),
                        'inputs': inputs,
                        'outputs': outputs,
                        'type': module.__class__.__name__
                    }, module.name + '.pth')
                    # from ipdb import set_trace; set_trace()
                # else:
                    # import time; time.sleep(10000)

        module.register_forward_hook(lambda module, inputs, outputs: check_nan(module, inputs, outputs))
        module.register_forward_hook(lambda module, inputs, outputs: check_nan(module, inputs, outputs))
    
    model.apply(add_module_name)
    model.apply(add_check_nan_hook)


def initialize_seq_parallel(
    sequence_parallel_size,
):
    if sequence_parallel_size <= 1:
        return None
    num_sequence_parallel_groups: int = dist.get_world_size() // sequence_parallel_size
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_SIZE
    _SEQUENCE_PARALLEL_SIZE = sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if dist.get_rank() in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_size():
    return _SEQUENCE_PARALLEL_SIZE

def get_sequence_parallel_rank():
    return torch.distributed.get_rank(group=get_sequence_parallel_group())

# 设置序列并行参数来保证优化器正确平均
from deepspeed.utils import groups
groups._get_sequence_parallel_world_size = get_sequence_parallel_size