# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from pathlib import Path
import copy
import datetime
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything
from data.calvin_dataset import CalvinDataset_Policy
from data.ego4d_dataset import Ego4DDataset_Policy
from training.trainer import Policy_Trainer
from torch.utils.data import DataLoader

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)


def get_date_str():
    return str(datetime.date.today())

class SetupCallback(Callback):
    def __init__(self, now, logdir, ckptdir):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir

    def on_train_start(self, trainer, model):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)

def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(' ', '_'),
        logdir=config['log_dir'],
        ckptdir=config['ckpt_dir']
    )

def init_trainer_config(configs):
    trainer_config = copy.deepcopy(configs['trainer']["pl_config"])
    trainer_config['devices'] = configs.get('devices', 'auto')
    trainer_config['num_nodes'] = configs.get('num_nodes', 1) 

    if 'strategy' not in trainer_config or trainer_config['strategy'] == 'ddp':
        trainer_config['strategy'] = DDPStrategy(find_unused_parameters=True) 

    exp_name = configs['exp_name']

    # init loggers
    log_dir = os.path.join(get_date_str(), exp_name)
    log_dir = os.path.join(configs['log_root'], log_dir)
    configs['log_dir'] = log_dir
    print("log_dir:",log_dir)

    Path(configs['log_dir']).mkdir(parents=True, exist_ok=True)     
    trainer_config['logger'] = [TensorBoardLogger(log_dir, name=exp_name)]


    # TODO: make callbacks configurable
    ckpt_dir = os.path.join(get_date_str(), exp_name)
    ckpt_dir = os.path.join(configs['ckpt_root'], ckpt_dir)
    configs['ckpt_dir'] = ckpt_dir
    print("ckpt_dir:",ckpt_dir)
    
    Path(configs['ckpt_dir']).mkdir(parents=True, exist_ok=True)
    trainer_config['callbacks'] = [
        init_setup_callback(configs),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1, every_n_epochs=configs["trainer"]["save_epoch"]) # if you have only limited space, just set save_top_k=1 to save the best model
    ]

    return trainer_config

def experiment(variant):
    set_seed(variant['seed'])
    trainer_config = init_trainer_config(variant)
    trainer = Trainer(**trainer_config)   
    model = Policy_Trainer(variant)

    print("use_reward:",variant["use_reward"])
    print("use_REINFORCE_loss:",variant["use_REINFORCE_loss"])
    print("RTG_pred:",variant["RTG_pred"])

    # dataset
    if variant["trainer"]["finetune"]:
        if variant["data_name"]=="calvin":    
            data_dir = variant["data_dir"]
            if "task_D_D" in data_dir: subfolder= "task_D_D"
            else: subfolder= "task_ABC_D"

            # print("main.py use_labeled:",variant["use_labeled"])

            train_data= CalvinDataset_Policy(
                # data_dir="PATH_TO_CALVIN/calvin_data",
                data_dir=data_dir,
                use_data_augmentation=True,
                subfolder= subfolder,
                mode= "train",
                forward_n_max=25,
                use_play=variant["use_play"],
                use_autonomous=variant["use_autonomous"],
                use_labeled=variant["use_labeled"],
                use_reward=variant["use_reward"],
                discounting_factor=variant["discounting_factor"],
                data_rate=variant["data_rate"],
                SubGoalReasoning=variant["SubGoalReasoning"],
                reward_data_dir=variant["reward_data_dir"],
                is_hard_tasks=variant["is_hard_tasks"],
                reward_dim=variant["reward_dim"],
                reward_name=variant["reward_name"],
                reward_comp=variant["reward_comp"]
                )
            val_data= CalvinDataset_Policy(
                data_dir=data_dir,
                use_data_augmentation=False,
                subfolder= subfolder,
                mode= "validate",
                forward_n_max=25,
                use_play=False,
                use_labeled=True,
                use_reward=variant["use_reward"],
                discounting_factor=variant["discounting_factor"],
                 data_rate=variant["data_rate"],
                 SubGoalReasoning=variant["SubGoalReasoning"],
                 reward_data_dir=variant["reward_data_dir"],
                 reward_dim=variant["reward_dim"],
                 reward_name=variant["reward_name"],
                  reward_comp=variant["reward_comp"]
                )
    else:
        train_data= Ego4DDataset_Policy(
                data_dir="PATH_TO_Ego4d_Videos",
                preprocess=None,
                video_sample_rate=2,
                seq_len=10,
                annotation_file= "PATH_TO_Ego4d_800k_annotations",
                use_data_augmentation=True,
                goal_interval=7)
        val_data= Ego4DDataset_Policy(
                data_dir="PATH_TO_Ego4d_Videos",
                preprocess=None,
                video_sample_rate=2,
                seq_len=10,
                annotation_file= "PATH_TO_Ego4d_800k_annotations",
                use_data_augmentation=False,
                goal_interval=7)
    
    train_dataloader= DataLoader(train_data, 
        batch_size=variant["trainer"]["batch_size"],
        num_workers=variant["trainer"]["num_workers"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
        )
    if val_data is not None:
        val_dataloader= DataLoader(val_data, 
                                    batch_size=variant["trainer"]["batch_size"],
                                    num_workers=variant["trainer"]["num_workers"],
                                    shuffle=True,
                                    pin_memory=True,
                                    persistent_workers=True
                                    )
    
    _kwargs = {
        'model': model,
        'train_dataloaders':train_dataloader,
        # 'val_dataloaders':val_dataloader,
        'ckpt_path': variant['resume']  # when you want to restore your training, modify this variant
    }
    if val_data is not None: _kwargs['val_dataloaders']=val_dataloader

    if _kwargs['ckpt_path'] is not None:
        print(f"Resuming from {variant['resume']}...")

    trainer.fit(**_kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Experiment
    parser.add_argument('--config', type=str,default="")
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int) 
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_root', default=None, type=str)
    parser.add_argument('--ckpt_root', default=None, type=str)
    parser.add_argument('--data_dir', type=str)
    # parser.add_argument('--batch_size', type=int)
    parser.add_argument("--resume",default=None,type=str)
    parser.add_argument("--data_name",default=None,type=str)

    parser.add_argument('--exp_name', default="baseline",type=str) 
    parser.add_argument('--is_hard_tasks', default=0, type=int)    
    
    parser.add_argument('--use_play', default=0, type=int)
    parser.add_argument('--use_labeled', default=1, type=int)
    parser.add_argument('--use_autonomous', default=0, type=int)
    
    parser.add_argument('--use_reward', default=0, type=int)
    parser.add_argument('--reward_data_dir', default=None, type=str)
    parser.add_argument('--use_REINFORCE_loss', default=0, type=int)
    parser.add_argument('--discounting_factor', default=0.9, type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--data_rate', default=1., type=float)
    parser.add_argument('--SubGoalReasoning', default=0, type=int)
    #RTG
    parser.add_argument('--RTG_pred', default=0, type=int)
    parser.add_argument('--RTG_tau', default=0.9, type=float)
    parser.add_argument('--RTG_loss_ratio', default=0.1, type=float)
    parser.add_argument('--reward_dim', default=1, type=int)
    parser.add_argument('--reward_name', default="densed", type=str)
    parser.add_argument('--reward_comp', default="all", type=str)
    
    parser.add_argument('--vision_latent_conditioned', default=0, type=int)
    parser.add_argument('--RTG_decoder_output_embeddings_repeat_num', default=1, type=int)
    parser.add_argument('--returnToGo_pred_conditioned', default=1, type=int)
    parser.add_argument('--use_goal_rgb', default=1, type=int)

    parser.add_argument('--data_curation', default=0, type=int)
    
    temp_args = vars(parser.parse_args())
    config_path=temp_args.pop("config")
    # load config files
    configs = json.load(open(config_path))
    for (k, v) in temp_args.items():
        if k not in configs or k=="resume":
            configs[k]=v
    return configs



if __name__ == '__main__':
    configs=parse_args()
    os.system(f"sudo chmod 777 -R {configs['ckpt_root']}")
    os.system(f"sudo chmod 777 -R {configs['log_root']}")
    experiment(variant=configs)
