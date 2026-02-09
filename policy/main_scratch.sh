#!/usr/bin/env bash
cd /PATH/TO/reinbot

GPUS_PER_NODE=$1 
master_port=$2
use_reward=$3
discounting_factor=$4
data_rate=$5
use_REINFORCE_loss=$6
RTG_pred=$7
RTG_loss_ratio=$8
use_play=$9
use_autonomous=$9
use_labeled=${10}
reward_dim=${11}
data_name=${12} #"calvin"
RTG_tau=${13}
vision_latent_conditioned=${14}
reward_name=${15}
reward_comp=${16}

if [ $data_name == "calvin" ]
then
    data_sub_name="task_ABC_D"
    data_dir="/PATH/TO/calvin/"
    reward_data_dir="/PATH/TO/reward_data/"
fi

pretrained_model_path="/PATH/TO/model_ckpt/pretrained.pt"

config="policy/config/train_from_scratch.json"
exp_name="xxx"


outputs="outputs_${data_name}/"
if [ ! -d "$outputs" ]; then
    mkdir -p "$outputs"
fi

subfix=`date "+%Y%m%d-%H%M"`
log_file="${outputs}/logs_training_"${subfix}"_${exp_name}.log"

NNODES=1 # number of machines
JOB_ID=107

START_PORT=50012
END_PORT=50020
for ((PORT=$START_PORT; PORT<=$END_PORT; PORT++)); do
    if ! nc -z localhost $PORT; then
        echo "Using port $PORT"
    
        torchrun \
            --nproc_per_node $GPUS_PER_NODE \
            --nnodes $NNODES \
            --node_rank 0 \
            --rdzv_id $JOB_ID \
            --rdzv_backend c10d \
            policy/main.py \
            --config $config \
            --devices $GPUS_PER_NODE \
            --data_dir $data_dir \
            --use_reward $use_reward \
            --discounting_factor $discounting_factor \
            --exp_name $exp_name \
            --pretrained_model_path $pretrained_model_path \
            --data_rate $data_rate \
            --use_REINFORCE_loss $use_REINFORCE_loss \
            --RTG_pred $RTG_pred \
            --RTG_loss_ratio $RTG_loss_ratio \
            --RTG_tau $RTG_tau \
            --use_play $use_play \
            --use_autonomous $use_autonomous \
            --use_labeled $use_labeled \
            --reward_dim $reward_dim \
            --data_name $data_name \
            --SubGoalReasoning 0 \
            --vision_latent_conditioned $vision_latent_conditioned \
            --RTG_decoder_output_embeddings_repeat_num 1 \
            --reward_name $reward_name \
            --reward_comp $reward_comp \
            --reward_data_dir $reward_data_dir > ${log_file} 2>&1
        break
    fi
done

# bash policy/main_scratch.sh 1 12345 1 0.3 0.1 0 1 0.001 1 0 1 calvin 0.9 1 densed 


