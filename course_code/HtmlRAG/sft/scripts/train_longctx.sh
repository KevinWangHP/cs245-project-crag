#!/bin/bash
# 默认参数值
config_path="../llm_modeling/Llama323B/"
#config_path="../llm_modeling/Phi35/"
#config_path="../llm_modeling/ChatGLM/"
model_path="../../../model"
pretrain="../../../model/Llama-3.2-3B-Instruct/"
#pretrain="../../../model/Phi-3.5-mini-instruct/"
#pretrain="../../../model/glm-4-9b-chat-1m/"
exp_name="v1019"
data_file="experiments/${exp_name}.json5"
output_dir="${model_path}/train-tree-rerank-llama32/${exp_name}"
#output_dir="${model_path}/train-tree-rerank-glm9b/${exp_name}"
max_length=35000
batch_size=1
epochs=3
gradient_accumulation_steps=8

# 解析长格式命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrain) pretrain="$2"; shift 2;;
        --data_file) data_file="$2"; shift 2;;
        --max_length) max_length="$2"; shift 2;;
        --batch_size) batch_size="$2"; shift 2;;
        --epochs) epochs="$2"; shift 2;;
        --output_dir) output_dir="$2"; shift 2;;
        --gradient_accumulation_steps) gradient_accumulation_steps="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done
# 输出解析后的参数值
echo "pretrain: $pretrain"
echo "data_file: $data_file"
echo "max_length: $max_length"
echo "batch_size: $batch_size"
echo "epochs: $epochs"
echo "output_dir: $output_dir"

cp -r $config_path/* "$pretrain"
# 配置环境变量
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
# 启动Python环境
cd "$(dirname "$0")" && cd ..
echo "workspace_dir=$PWD"


echo "python=$(which python)"
if [[ $RANK -eq 0 ]]; then
    read -r -d '' meta_info << EOF
- workspace_dir: $PWD
- pretrain: $pretrain
- data_file: $data_file
- max_length: $max_length
- batch_size: $batch_size
- epochs: $epochs
- output_dir: $output_dir
- python: $(which python)
EOF
    #python tools/feishu_robot.py "$meta_info"
fi
set -x
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --use-env \
    train_chat_ds.py \
        --model_name_or_path $pretrain \
        --data_file $data_file \
        --output_dir $output_dir \
        --model_max_length $max_length \
        --num_train_epochs $epochs \
        --per_device_train_batch_size $batch_size \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --seq_parallel_size 8 \
        --save_strategy epoch \
        --learning_rate 2e-5 \
        --lr_scheduler_type constant \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --weight_decay 1e-4 \
        --warmup_ratio 0.01 \
        --logging_steps 4 \
        --gradient_checkpointing True \
        --deepspeed scripts/ds_config.json \
        --bf16 True \
        --tf32 True \
        --report_to tensorboard \
        --data_cache_dir "cached" \
        --disable_packing True \
        --resume_from_checkpoint False \
        --data_shm_dir True  # 只有云盘乱序读取慢,影响数据制作,才需要加这个

