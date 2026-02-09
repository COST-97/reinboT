#!/usr/bin/env bash
cd /PATH/TO/reinbot

RTG_pred=$1
reward_dim=${2}
vision_latent_conditioned=${3}

export EPOCHS=(xxx)
export CKPT_DIR="xxx"

IFS='/' read -r -a array <<< "$CKPT_DIR"
exp_name=${array[-1]} 

export MESA_GL_VERSION_OVERRIDE=3.3
echo $EPOCHS
echo $CKPT_DIR
chmod 777 -R ${CKPT_DIR}

config="/PATH/TO/policy/config/train_finetune.json"

dataset_path="/PATH/TO/task_ABC_D"

outputs="outputs_test/"

if [ ! -d "$outputs" ]; then
    mkdir -p "$outputs"
fi
data_name="calvin"
data_sub_name="task_ABC_D"
subfix=`date "+%Y%m%d-%H%M"`
log_file="xxx"

export COUNTER=-1
# Use a for loop to iterate through a list
for epoch in "${EPOCHS[@]}"; do
    export COUNTER=$((${COUNTER} + 1))
    export CUDA_VISIBLE_DEVICES=${COUNTER}
    python3 evaluate/eval_scratch.py \
        --dataset_path ${dataset_path} \
        --ckpt_dir ${CKPT_DIR} \
        --epoch ${epoch} \
        --RTG_pred $RTG_pred \
        --reward_dim $reward_dim \
        --vision_latent_conditioned $vision_latent_conditioned \
        --config_path $config > ${log_file} 2>&1
done
wait

