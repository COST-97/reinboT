#!/usr/bin/env bash
cd /PATH/TO/reinboT

export PYTHONPATH="${PYTHONPATH}:/PATH/TO/reinbot"

GPUS_PER_NODE=$1 
master_port=$2
use_reward=$3
discounting_factor=$4
data_rate=$5
use_REINFORCE_loss=$6
RTG_pred=$7
RTG_loss_ratio=$8
use_play=0 # finetune phase, only use language labeled data
use_labeled=1
use_autonomous=0
reward_dim=${9}
data_name=${10} #"calvin"
RTG_tau=${11}
SubGoalReasoning=${12}
vision_latent_conditioned=${13}
RTG_decoder_output_embeddings_repeat_num=${14}

data_sub_name="task_ABC_D"
data_dir="/PATH/TO/calvin/"
reward_data_dir="/PATH/TO/reinbot/reward_data/"

outputs="outputs_finetune/"


resume="/PATH/TO/xxx.ckpt"


IFS='/' read -r -a array <<< "$resume"
exp_name=${array[-2]}  # 输出 python 
exp_name="xxx"

if [ ! -d "$outputs" ]; then
    mkdir -p "$outputs"
fi

subfix=`date "+%Y%m%d-%H%M"`
log_file="${outputs}/logs_training_"${subfix}"_${data_name}_${data_sub_name}_${exp_name}.log"

config="policy/config/train_finetune.json"

NNODES=1 # number of machines
JOB_ID=107

START_PORT=50003
END_PORT=50020
for ((PORT=$START_PORT; PORT<=$END_PORT; PORT++)); do
    if ! nc -z localhost $PORT; then
        echo "Using port $PORT"
        torchrun \
            --nproc_per_node $GPUS_PER_NODE \
            --nnodes $NNODES \
            --node_rank 0 \
            --master_port $PORT \
            --rdzv_id $JOB_ID \
            --rdzv_backend c10d \
            policy/main.py \
            --config $config \
            --devices $GPUS_PER_NODE \
            --data_dir $data_dir \
            --use_reward $use_reward \
            --discounting_factor $discounting_factor \
            --exp_name $exp_name \
            --resume $resume \
            --data_rate $data_rate \
            --use_REINFORCE_loss $use_REINFORCE_loss \
            --RTG_pred $RTG_pred \
            --RTG_loss_ratio $RTG_loss_ratio \
            --RTG_tau $RTG_tau \
            --use_play $use_play \
            --use_autonomous $use_autonomous \
            --use_labeled $use_labeled \
            --data_name $data_name \
            --reward_dim $reward_dim \
            --SubGoalReasoning $SubGoalReasoning \
            --vision_latent_conditioned $vision_latent_conditioned \
            --RTG_decoder_output_embeddings_repeat_num $RTG_decoder_output_embeddings_repeat_num \
            --reward_data_dir $reward_data_dir > ${log_file} 2>&1
        break
    fi
done

# bash policy/main_finetune.sh 8 12345 1 0.1 0.1 0 1 0.001 17 calvin 0.9 0 1 1

