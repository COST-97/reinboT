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
from PIL import Image
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import time
import numpy as np
from data.utils import *
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageChops
import math
import random

ACTION_POS_SCALE = 50
ACTION_ROT_SCALE = 20


# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    # perform random shifts independently for each image in the batch
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class RandomShiftsSingleAug(nn.Module):
    #  perform same random shifts for each image in the batch
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate") # pad before shift
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class CalvinDataset_Policy(Dataset):
    def __init__(self,
                 data_dir,
                 seq_len=10,
                 act_len=5, 
                 forward_n_max=25, 
                 mode='train',
                 subfolder='task_ABC_D',
                 use_data_augmentation=True,
                 task_num=525, #calvin task num max. # 10000,
                 use_play=0,
                 use_labeled=True,
                 use_reward=False,
                 discounting_factor=0.9,
                 data_rate=0.1,
                 SubGoalReasoning=0,
                 reward_data_dir=None,
                 is_hard_tasks=False,
                 reward_dim=1,
                 use_autonomous=0,
                 reward_name="densed",
                 reward_comp= "all"
                 ):
        super().__init__()
        '''
        seq_len: length of observation
        act_len: length of output action trajectroy
        forward_n_max:  the maximum interval of goal image
        use_data_augmentation: whether to use RandomShiftsSingleAug
        task_num: It determins the maximum number of trajectories that could be sampled from a trajectory. If you want to train with all the trajectories,
        just set a very great value. When you want to do fewshot experiments, you should set it as a specific value, such as 52( 10% of total data ).
        use_full: whether to use play data
        use_labeled: whether to use data annotated with language
        '''
        self.dataset_dir = os.path.join(data_dir, subfolder)
        self.seq_len = seq_len 
        self.act_len = act_len
        self.forward_n_max = forward_n_max
        self.mode = mode
        self.use_data_augmentation = use_data_augmentation
        
        self.data_rate = data_rate
        self.task_num = int(data_rate*task_num)

        self.use_play=use_play
        self.use_autonomous = use_autonomous
        self.use_labeled=use_labeled
        self.reward_data_dir=reward_data_dir
        
        self.SubGoalReasoning = SubGoalReasoning

        self.action_dim = 7
        self.state_dim = 7  # ( x,y,z,r,p,y,gripper) 
        self.reward_dim=reward_dim
        self.reward_comp=reward_comp

        if mode == 'validate':
            tag = "validation"
        elif mode == 'train':
            tag = "training"
        else:
            raise ValueError("Mode must be either train or validate")
        
        self.is_hard_tasks=is_hard_tasks
        if self.is_hard_tasks:
            print("=="*50)
            print("is_hard_tasks")
            print("=="*50)
            self.funcMap = {
               "shift": DataAugmentation.randomShift, 
               "rotation": DataAugmentation.randomRotation,
               "color": DataAugmentation.randomColor,
                "cutout": DataAugmentation.randomCutout,
                "erasing": DataAugmentation.randomErasing,
               }
            self.image_modify_value = {
               "shift":[0.01,0.2], 
                 "rotation": [2,10],
               "color": [11,20],
                "cutout": [1,10],
                "erasing": [0.01,0.1],
               }

        self.use_reward=use_reward
        if self.use_reward:
            self.reward_name=reward_name
            self.discounting_factor=discounting_factor #0.3 # 0.9  
            print("discounting_factor:",self.discounting_factor)  
            json_file_path = reward_data_dir+"reward_dict_ABC_D_{}.json".format(tag)
            with open(json_file_path, 'r') as file:
                self.data_reward = json.load(file)
            self.data_reward=eval(self.data_reward)
            if self.use_play:
                # json_file_path = reward_data_dir+"reward_dict_ABC_D_training_playData.json"
                # 2024/12/5
                json_file_path = reward_data_dir+"reward_dict_ABC_D_training_playData_EpisodeLength64.json"

                with open(json_file_path, 'r') as file:
                    self.data_reward_play = json.load(file)
                self.data_reward_play=eval(self.data_reward_play)
            
            if self.use_autonomous:
                json_file_path = reward_data_dir+"reward_dict_training_autonomousData.json"
                with open(json_file_path, 'r') as file:
                    self.data_reward_autonomous = json.load(file)
                self.data_reward_autonomous=eval(self.data_reward_autonomous)

        # As we use MAE as Encoder, we need to resize and normalize
        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406) 
        self.clip_std = (0.229, 0.224, 0.225)

        if self.use_data_augmentation:
            self.static_rgb_preprocess_train = T.Compose([
                RandomShiftsSingleAug(pad=10), # 10 for static rgb
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            
            self.hand_rgb_preprocess_train = T.Compose([
                RandomShiftsSingleAug(pad=4), # 4 for hand rgb
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        else:
            self.static_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            self.hand_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        
        self.static_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])
        self.hand_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])

        self.autonomous_sigal=None
        if self.use_autonomous:
            self.autonomous_data_root_path = "/PATH/TO/non_experts_data"
            self.autonomous_data_name_list = []
            self.autonomous_sigal = "/datasets_autonomous"
            # for i in range(1,13):
            for i in range(8,13): # 2024/12/9
                if i==12:self.autonomous_data_name_list.append(self.autonomous_sigal+"/RobotFlamingoDBG/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_final_weights.pth")
                else: self.autonomous_data_name_list.append(self.autonomous_sigal+"/RobotFlamingoDBG/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_{}_iter.pth".format(str(i*10000)))
            for i in [0.05,0.1,0.15]:self.autonomous_data_name_list.append(self.autonomous_sigal+"_noise_{}/RobotFlamingoDBG/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_final_weights.pth".format(str(i)))

        self.dataset_dir = os.path.join(self.dataset_dir, tag)
        self._initialize()
        print(f'{self.mode}:{len(self)} samples in total')

    def load(self,path):
        # Sometimes we can not immediately load the data when training in our GPU cluster. We set this protection mechanism
        t1=time.time()
        while True:
            try:
                result=np.load(path)
                break
            except:
                t2=time.time()
                if (t2-t1) >60:
                    assert False , "bad load"
                else:
                    continue
        return result

    def _initialize(self):
        """Generate the sequence index pair."""
        self.anns = np.load(
            os.path.join(self.dataset_dir, "lang_annotations", "auto_lang_ann.npy"), 
            allow_pickle=True).item()
        
        # (traj_index, start, end, act_end, traj_st)
        self.seq_tuple = []
        self.seq_labbeled_list = []
        
        task_dict = {}
        n_trajs = len(self.anns['info']['indx'])
        
        # np.random.seed(42)
        # arr = np.arange(n_trajs)
        # num = int(self.data_rate*np.shape(arr)[0])
        # selected_elements = np.random.choice(arr, size=num, replace=False)

        print(f"DataRate:{self.data_rate}, {self.task_num} labeled trajectories per task.")

        # print("self.use_labeled:",self.use_labeled)

        if self.use_labeled: 
            for traj_idx in range(n_trajs):
            # for traj_idx in selected_elements:
                traj_st, traj_ed = self.anns['info']['indx'][traj_idx]
                traj_task = self.anns['language']['task'][traj_idx]
                text=self.anns["language"]["ann"][traj_idx]

                if self.SubGoalReasoning:
                    if torch.rand(1).item() < 0.5: text = f"""The instruction is "{text}"."""
                # self.text = f"""The language instruction for this task is not "{self.text}", but something else."""

                if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                    continue # exclude extremely short trajectory

                # sample trajectories based on task_num
                if traj_task not in task_dict:
                    task_dict[traj_task] = 1
                else:
                    task_dict[traj_task] = task_dict[traj_task] + 1
                if task_dict[traj_task] > self.task_num: # when arriving the maximum number, stop sampling from the task
                    continue

                # the action in the last frame is discarded
                for st in range(traj_st, traj_ed - self.seq_len + 1):
                    ed = st + self.seq_len
                    act_ed = st + self.seq_len + self.act_len - 1 
                    self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,text])
                    self.seq_labbeled_list.append(traj_idx)

        Labeled_samples = len(self.seq_tuple)
        print("Labeled samples:",Labeled_samples)

        if self.use_play:
            # if you want to train with play data, please first generate play.json, and replace the following path
            # playData_path = self.reward_data_dir+"play.json"
            playData_path = self.reward_data_dir+"play_EpisodeLength64.json"
            # with open(os.path.join(self.dataset_dir, "play.json"), 'r') as file:
            with open(os.path.join(playData_path), 'r') as file:
                self.play_traj = json.load(file)
            n_trajs = len(self.play_traj['st_ed_list'])
            print(f"{n_trajs} play trajectories!")
            
            text = ''
            if self.SubGoalReasoning:
                text = np.random.choice(self.anns["language"]["ann"])
                # if torch.rand(1).item() < 0.5: text = f"""The language instruction of this task is: {text}."""
                if torch.rand(1).item() < 0.5: text = f"""The instruction is not "{text}", but something else."""

            for traj_idx in range(n_trajs):
                traj_st, traj_ed = self.play_traj['st_ed_list'][traj_idx]
                if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                    continue
                # the action in the last frame is discarded
                for st in range(traj_st, traj_ed - self.seq_len + 1):
                    ed = st + self.seq_len
                    act_ed = st + self.seq_len + self.act_len - 1
                    self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,text])
            Allsamples = len(self.seq_tuple)
            print("play samples:",Allsamples-Labeled_samples)

            Labeled_play_samples = len(self.seq_tuple)
            print("Labeled_play_samples:",Labeled_play_samples)

        if self.use_autonomous:
            n_trajs = 0
            for data_name in self.autonomous_data_name_list:
                anns = np.load(
                    os.path.join(self.autonomous_data_root_path + data_name, "lang_ann.npy"), 
                    allow_pickle=True).item()

                n_traj = len(anns['info']['indx'])
                n_trajs += n_traj
            
                for traj_idx in range(n_traj):
                    # traj_st, traj_ed = self.play_traj['st_ed_list'][traj_idx]
                    traj_st, traj_ed = anns['info']['indx'][traj_idx]
                    if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                        continue
                    
                    # for ind,(start,end) in enumerate(indx):
                    text = ''
                    if self.SubGoalReasoning:
                        text = np.random.choice(self.anns["language"]["ann"])
                        # if torch.rand(1).item() < 0.5: text = f"""The language instruction of this task is: {text}."""
                        if torch.rand(1).item() < 0.5: text = f"""The instruction is not "{text}", but something else."""

                    if traj_ed-traj_st==63 and anns['info']['success'][traj_idx]==False:
                        # the action in the last frame is discarded
                        for st in range(traj_st, traj_ed - self.seq_len + 1):
                            ed = st + self.seq_len
                            act_ed = st + self.seq_len + self.act_len - 1
                            self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,text,data_name])

            Allsamples = len(self.seq_tuple)
            print(f"{n_trajs} autonomous trajectories!")
            print("autonomous samples:",Allsamples-Labeled_play_samples)

    def __len__(self):
        return len(self.seq_tuple)

    def __getitem__(self, index):
        time_get_data_start = time.time()

        curr_tuple = self.seq_tuple[index]
        traj_idx = curr_tuple[0]
        st = curr_tuple[1]
        ed = curr_tuple[2]
        act_ed = curr_tuple[3]
        traj_st = curr_tuple[4]
        traj_ed = curr_tuple[5]       
        self.text=curr_tuple[6]

        if len(curr_tuple)==8:
            # print("curr_tuple",curr_tuple)
            autonomous_data_name = curr_tuple[7]
            dataset_dir = self.autonomous_data_root_path+autonomous_data_name
        else: 
            dataset_dir=self.dataset_dir


        static_rgbs = []
        hand_rgbs = []
        states = []


        state_buffer = [np.zeros(self.state_dim) for _ in range(self.seq_len + self.act_len - 1)]
        state_valid = [0 for _ in range(self.seq_len + self.act_len - 1)]

        tlen = ed - st # sequence length
        assert tlen == self.seq_len

        for i in range(st, act_ed):
            if i <= traj_ed:
                frame = self.load(os.path.join(dataset_dir, f"episode_{i:07d}.npz"))
                if i <= traj_ed:
                    state_buffer[i - st] = frame['robot_obs']
                    state_valid[i - st] = 1
            
            if i < ed:
                states += [frame['robot_obs']]
                
                static_rgb = frame['rgb_static']
                static_rgb = Image.fromarray(static_rgb)

                if self.is_hard_tasks:
                    # rgb_obs np.array 200, 200, 3
                    rand_name = np.random.choice(list(self.funcMap.keys()))
                    low,high = self.image_modify_value[rand_name][0],self.image_modify_value[rand_name][1]
                    
                    # print(rand_name,low,high)

                    rand_value = np.random.uniform(low, high)
                    if rand_name in ["rotation","color","cutout"]:
                        rand_value = int(rand_value)
                    static_rgb = self.funcMap[rand_name](static_rgb,rand_value)
                
                static_rgb = T.ToTensor()(static_rgb.convert("RGB"))
                static_rgbs.append(static_rgb)

                hand_rgb = frame['rgb_gripper']
                hand_rgb = Image.fromarray(hand_rgb)
                hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
                hand_rgbs.append(hand_rgb)

        if act_ed <= traj_ed:
            act_ed_frame = self.load(os.path.join(dataset_dir, f"episode_{act_ed:07d}.npz"))
            state_buffer += [act_ed_frame['robot_obs']]
            state_valid += [1]
        else:
            state_buffer += [np.zeros(self.action_dim)] # mask non-existent action
            state_valid += [0]
        
        assert len(state_buffer) == (self.seq_len + self.act_len)
        assert len(state_valid) == (self.seq_len + self.act_len)

        # we use relative action represented in current end effector coordinate
        action_buffer = []
        action_valid = []

        returns_buffer=[]
        RTG_data=torch.zeros(self.seq_len,self.reward_dim).float()

        self.current_progress = (ed-traj_st)/(traj_ed-traj_st)
        self.subgoal_dict=None
        self.current_s_t = st+self.seq_len-1-traj_st

        for k in range(0, self.seq_len + self.act_len - 1):
            if (state_valid[k] and state_valid[k+1]):
                # t1 = time.time()

                xyz0 = state_buffer[k][0:3]
                rpy0 = state_buffer[k][3:6]
                rotm0 = euler2rotm(rpy0)
                xyz1 = state_buffer[k+1][0:3]
                rpy1 = state_buffer[k+1][3:6]
                rotm1 = euler2rotm(rpy1)
                gripper1 = state_buffer[k+1][-1]
                delta_xyz = rotm0.T @ (xyz1 - xyz0)
                delta_rotm = rotm0.T @ rotm1
                delta_rpy = rotm2euler(delta_rotm)
                temp_action = np.zeros(7)
                temp_action[:3] = delta_xyz * ACTION_POS_SCALE
                temp_action[3:6] = delta_rpy * ACTION_ROT_SCALE
                temp_action[-1] = (gripper1 + 1.0) / 2.0  # (-1, 1) -> (0, 1)
                action_buffer += [temp_action]
                action_valid += [1]

                # t2 = time.time()
                # reward 

                if self.use_reward:
                    if self.use_play:
                        if len(curr_tuple)==8: # autonomous data
                            returns_single,RTG = self.get_reward(st+k-traj_st,traj_idx,self.data_reward_autonomous[autonomous_data_name],0)
                        else:
                            if str(traj_idx) in self.data_reward_play.keys(): # play data
                                returns_single,RTG = self.get_reward(st+k-traj_st,traj_idx,self.data_reward_play,0)
                            elif str(traj_idx) in self.data_reward.keys(): # labeled data
                                returns_single,RTG = self.get_reward(st+k-traj_st,traj_idx,self.data_reward,1)
                    else:                        
                        returns_single,RTG = self.get_reward(st+k-traj_st,traj_idx,self.data_reward,1) # labeled data
                    
                    # print("returns_single",returns_single)
                    returns_buffer += [returns_single]
                    if k<self.seq_len:
                        RTG_data[k] = RTG
                    
            else:
                action_buffer += [np.zeros(self.action_dim)]
                action_valid += [0]
        
                # reward 
                if self.use_reward:
                    # print("bad")
                    returns_buffer += [np.zeros(1)]
        
        # print("returns_buffer",returns_buffer)
        assert len(action_buffer) == (self.seq_len + self.act_len - 1)
        assert len(action_valid) == (self.seq_len + self.act_len - 1)

        # Prepare goal Image   
        goal_min_idx = ed
        goal_max_idx = min(traj_ed + 1, ed + self.forward_n_max)
        goal_ids = list(range(goal_min_idx, goal_max_idx))
        goal_idx = random.choice(goal_ids)

        assert (goal_idx >= ed) and (goal_idx <= traj_ed)
        goal_frame = self.load(os.path.join(dataset_dir, f"episode_{goal_idx:07d}.npz"))
        goal_rgb = goal_frame['rgb_static']
        goal_rgb = Image.fromarray(goal_rgb)
        goal_rgb = T.ToTensor()(goal_rgb.convert("RGB"))
        static_rgbs.append(goal_rgb) # put goal_rgb at the end of static_rgbs

        # preprocess all Images
        static_rgbs = torch.stack(static_rgbs, dim=0)
        hand_rgbs = torch.stack(hand_rgbs, dim=0)
        if self.mode == 'train':
            static_rgbs = self.static_rgb_preprocess_train(static_rgbs)
            hand_rgbs = self.hand_rgb_preprocess_train(hand_rgbs)
        else:
            static_rgbs = self.static_rgb_preprocess_val(static_rgbs)
            hand_rgbs = self.static_rgb_preprocess_val(hand_rgbs)
        goal_rgb = static_rgbs[-1]
        static_rgbs = static_rgbs[:-1]

        # transform them into torch tensor
        _, C, H, W = static_rgbs.shape
        rgb_data  = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        _, C, H, W = hand_rgbs.shape
        hand_rgb_data = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        rgb_data [:tlen] = static_rgbs
        hand_rgb_data[:tlen] = hand_rgbs
        goal_rgb_data = goal_rgb.float() # (C, H, W)
    
        # State
        states = np.array(states)
        arm_states = states[:, :6]
        gripper_states = (states[:, -1:] + 1.0) / 2.0 # (-1, 1) -> (0, 1)
        states = np.hstack([arm_states, gripper_states])
        states = torch.from_numpy(states)
        state_data = torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        state_data[:tlen] = states

        # Relative state
        rel_states = np.zeros((tlen, self.state_dim), dtype=np.float32)
        first_xyz = arm_states[0, 0:3]
        first_rpy = arm_states[0, 3:6]
        first_rotm = euler2rotm(first_rpy)
        first_gripper = gripper_states[0]
        rel_states[0, -1] = first_gripper
        for i in range(1, tlen):
            curr_xyz = arm_states[i, 0:3]
            curr_rpy = arm_states[i, 3:6]
            curr_rotm = euler2rotm(curr_rpy)
            curr_gripper = gripper_states[i]
            rel_rotm = first_rotm.T @ curr_rotm
            rel_rpy = rotm2euler(rel_rotm)
            rel_xyz = first_rotm.T @ (curr_xyz - first_xyz)
            rel_states[i, 0:3] = rel_xyz
            rel_states[i, 3:6] = rel_rpy
            rel_states[i, 6] = curr_gripper
        rel_state_data= torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        rel_state_data[:tlen] = torch.from_numpy(rel_states)

        # Action and action mask
        actions = []
        action_masks = []
        returns = []
        for i in range(0, ed - st):
            actions.append(action_buffer[i:(i+self.act_len)])
            action_masks.append(action_valid[i:(i+self.act_len)])

            if self.use_reward: returns.append(returns_buffer[i:(i+self.act_len)])

        actions = np.array(actions) # (len, act_len, act_dim)
        actions = torch.from_numpy(actions)
        action_data  = torch.zeros(self.seq_len, self.act_len, self.action_dim).float() # (len, act_len, action_dim)
        action_data [:tlen] = actions
        action_masks = np.array(action_masks) # (len, act_len, act_dim)
        action_masks = torch.from_numpy(action_masks).long()
        action_mask_data = torch.zeros(self.seq_len, self.act_len).long() # (len, act_len, action_dim)
        action_mask_data[:tlen] = action_masks

        returns_data  = torch.zeros(self.seq_len, self.act_len, 1).float() # (len, act_len, 1)
        if self.use_reward:
            returns = np.array(returns) # (len, act_len)
            returns = torch.from_numpy(returns)
            # print("returns",returns)
            returns_data [:tlen] = returns

        # Attention mask (should be all 1 for full dataset)
        attention_mask = np.ones(self.seq_len, dtype=np.int32) # (len)
        attention_mask[tlen:] = 0.0
        assert np.sum(attention_mask) == self.seq_len
        attention_mask_data = torch.from_numpy(attention_mask).long()


        #progress
        progress_data=torch.zeros(self.seq_len).float()
        for i in range(self.seq_len):
            progress=(st+i-traj_st)/(traj_ed-traj_st)
            progress_data[i]=progress

        # print("self.subgoal_dict",self.subgoal_dict)

        if self.SubGoalReasoning and self.subgoal_dict is not None:
            if torch.rand(1).item() < 0.3:
                self.text += f""" {self.subgoal_dict["subgoal_num_completed"]} of the {self.subgoal_dict["subgoal_num"]} sub-goal(s) have been completed. 
                            """
            elif torch.rand(1).item() < 0.6:
                self.text += f""" The completion progress of the next sub-goal and the overall task are {round(100*self.subgoal_dict["subgoal_current_progress"],2)}% 
                                    and {round(100*self.current_progress,2)}% respectively.
                            """
            elif torch.rand(1).item() < 0.9:
                self.text += f""" {self.subgoal_dict["subgoal_num_completed"]} of the {self.subgoal_dict["subgoal_num"]} sub-goal(s) have been completed. 
                            The completion progress of the next sub-goal and the overall task are {round(100*self.subgoal_dict["subgoal_current_progress"],2)}% 
                                    and {round(100*self.current_progress,2)}% respectively.
                            """
                
        time_get_data_end = time.time()
        
        data = dict()
        data['goal_rgb'] = goal_rgb_data # (C, H, W)
        data['rgb'] = rgb_data # (len, C, H, W)
        data['hand_rgb'] = hand_rgb_data # (len, C, H, W)
        data['state'] = state_data # (len, state_dim) 
        data['rel_state'] = rel_state_data # (len, state_dim)
        data['action'] = action_data # (len, act_len, action_dim)
        data['attention_mask'] = attention_mask_data # (len,)
        data['action_mask'] = action_mask_data # (len, act_len, action_dim)
        data["progress"]=progress_data #(len)
        data["RTG"]=RTG_data #(len,reward_dim)
        data["returns"]=returns_data #(len, act_len, 1)
        data['text'] = [self.text]
        data["time_get_data"]=time_get_data_end-time_get_data_start
        return data

    def reward_design(self,reward, labeled_reward,ed):
        if self.reward_comp in ["all", "mse_state_no","mse_obs_no","ssim_obs_no","orb_obs_no"]:
            reward=reward[:,0:16]
            reward[:,9:15] = 0.5*reward[:,9:15] # T, D
            
            if self.reward_comp == "all":
                if self.reward_dim==1:
                    reward=np.sum(reward,axis=-1) # T
                elif self.reward_dim==17:
                    reward_total=np.sum(reward,axis=-1,keepdims=True) # T
                    reward = np.concatenate((reward, reward_total),axis=-1) # 17
            
            elif self.reward_comp == "mse_state_no":
                reward = np.concatenate((reward[:,0:6], reward[:,9:]),axis=-1) # reward dim: 14
                reward_total=np.sum(reward,axis=-1,keepdims=True) # T
                reward = np.concatenate((reward, reward_total),axis=-1) # 14

            elif self.reward_comp == "mse_obs_no":
                reward = np.concatenate((reward[:,1:3], reward[:,4:]),axis=-1) # reward dim: 15
                reward_total=np.sum(reward,axis=-1,keepdims=True) # T
                reward = np.concatenate((reward, reward_total),axis=-1) # 15

            elif self.reward_comp == "ssim_obs_no":
                reward = np.concatenate((reward[:,0:1], reward[:,2:4],reward[:,5:]),axis=-1) # reward dim: 15
                reward_total=np.sum(reward,axis=-1,keepdims=True) # T
                reward = np.concatenate((reward, reward_total),axis=-1) # 15

            elif self.reward_comp == "orb_obs_no":
                reward = np.concatenate((reward[:,0:2], reward[:,3:5],reward[:,6:]),axis=-1) # reward dim: 15
                reward_total=np.sum(reward,axis=-1,keepdims=True) # T
                reward = np.concatenate((reward, reward_total),axis=-1) # 15

            reward_min,reward_max = np.min(reward,axis=0),np.max(reward,axis=0)
            reward= (reward[ed:] - reward_min)/( reward_max - reward_min + 10**(-6)) # t:T
            reward = 0.1*(reward+labeled_reward) # T, D or T
        
        elif self.reward_comp == "goal_wo": # reward_dim 8
            reward=reward[:,9:16]
            reward[:,9:15] = 0.5*reward[:,9:15] # T, D
            
            reward_total=np.sum(reward,axis=-1,keepdims=True) # T
            reward = np.concatenate((reward, reward_total),axis=-1) # 17
                
            reward_min,reward_max = np.min(reward,axis=0),np.max(reward,axis=0)
            reward= (reward[ed:] - reward_min)/( reward_max - reward_min + 10**(-6)) # t:T
            
            reward = 0.1*(reward+labeled_reward) # T, D or T

        elif self.reward_comp == "smooth_wo": # reward_dim 11
            # reward=reward[:,0:9]
            reward=np.concatenate((reward[:,0:9],reward[:,15:16]),axis=-1)
            reward_total=np.sum(reward,axis=-1,keepdims=True) # T
            reward = np.concatenate((reward, reward_total),axis=-1) # 17
                
            reward_min,reward_max = np.min(reward,axis=0),np.max(reward,axis=0)
            reward= (reward[ed:] - reward_min)/( reward_max - reward_min + 10**(-6)) # t:T
            reward = 0.1*(reward+labeled_reward) # T, D or T

        elif self.reward_comp == "progress_wo":  # reward_dim 16
            reward=reward[:,0:15]
            reward[:,9:15] = 0.5*reward[:,9:15] # T, D
            
            reward_total=np.sum(reward,axis=-1,keepdims=True) # T
            reward = np.concatenate((reward, reward_total),axis=-1) # 17
                
            reward_min,reward_max = np.min(reward,axis=0),np.max(reward,axis=0)
            reward= (reward[ed:] - reward_min)/( reward_max - reward_min + 10**(-6)) # t:T
            reward = 0.1*(reward+labeled_reward) # T, D or T

        elif self.reward_comp == "labeled_wo":  # reward_dim 17
            reward=reward[:,0:16]
            reward[:,9:15] = 0.5*reward[:,9:15] # T, D
            
            reward_total=np.sum(reward,axis=-1,keepdims=True) # T
            reward = np.concatenate((reward, reward_total),axis=-1) # 17
                
            reward_min,reward_max = np.min(reward,axis=0),np.max(reward,axis=0)
            reward= (reward[ed:] - reward_min)/( reward_max - reward_min + 10**(-6)) # t:T
            reward = 0.1*reward # T, D or T

        return reward
        
    def get_reward(self,ed,traj_idx,data_reward,labeled_reward=0):
        # traj_idx = str(traj_idx)
        # if self.SubGoalReasoning and ed==self.current_s_t:
  
        episode_keypoints=np.array(data_reward[str(traj_idx)]["episode_keypoints"])
        if self.SubGoalReasoning and ed==self.current_s_t:
            subgoal_ind_old=0
            for ind,subgoal_ind in enumerate(episode_keypoints):
                if ed<=subgoal_ind:
                    # subgoal_id=subgoal_ind
                    self.subgoal_dict={
                        "subgoal_num":np.shape(episode_keypoints)[0],
                        "subgoal_num_completed":ind,
                        "subgoal_current_progress":(ed-subgoal_ind_old)/(subgoal_ind-subgoal_ind_old)
                    }
                    break
                subgoal_ind_old=subgoal_ind

        reward=np.array(data_reward[str(traj_idx)]["reward"])
        # reward=np.concatenate((reward[:,0:3],reward[:,6:16]),axis=-1)
        episode_length = np.shape(reward)[0]

        if self.reward_name=="densed":
            reward = self.reward_design(reward,labeled_reward,ed)

        elif self.reward_name=="sparse":
            reward=np.zeros((episode_length))
            if labeled_reward>0: reward[-3:]=1.
            reward=reward[ed:]

        elif self.reward_name=="subgoalSparse":
            reward=np.zeros((episode_length))
            reward[list(episode_keypoints)]=1.
            if labeled_reward>0: reward[-3:]=1.
            reward=reward[ed:]

        # print("self.reward_name",self.reward_name,reward)

        # discounting_factor=0.8
        if self.reward_dim==1:
            returns = np.sum(reward * self.discounting_factor ** np.arange(len(reward)),keepdims=True)
        else:
            returns = np.sum(reward[:,-1] * self.discounting_factor ** np.arange(len(reward)),keepdims=True)

        RTG = np.sum(reward, axis=0,keepdims=True) # t+t_1+...
        
        returns = returns #/episode_length
        RTG = torch.from_numpy(RTG) #/episode_length
        
        # print("RTG",RTG)

        return returns, RTG
    
        #                   # image goal tracking
        #                   mse_value, 0
        #                   ssim_value, 1
        #                   orb_similarity, 2
        #                   mse_value_gripper, 3
        #                   ssim_value_gripper, 4
        #                   orb_similarity_gripper, 5

        #                   # prop. goal tracking
        #                   error_pos, 6
        #                   error_ori, 7
        #                   error_joint, 8
                          
        #                   # auxiliary rewards
        #                     error_pos_vel, 9
        #                     error_ori_angle_vel, 10 
        #                   error_joint_vel, 11
        #                   error_joint_acc, 12
        #                   error_action_vel, 13
        #                   error_action_acc, 14
                        
        #                   # sub-goal progress
        #                   reward_subgoal 15
        #                    reward_total 16




class DataAugmentation:
    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")
 
    @staticmethod
    # def randomShift(image):
    def randomShift(image, shift_strength=0.2):
        """
        对图像进行平移操作
        :param image: PIL的图像image
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        :return: 翻转之后的图像
        """
        xoff = np.random.randint(0, math.ceil(image.size[0]*shift_strength))
        yoff = np.random.randint(0, math.ceil(image.size[1]*shift_strength))
        # return image.offset(xoffset = random_xoffset, yoffset = random_yoffset)
        # return image.offset(random_xoffset)
        width, height = image.size
        c = ImageChops.offset(image,xoff,yoff)
        c.paste((0,0,0),(0,0,xoff,height))
        c.paste((0,0,0),(0,0,width,yoff))
        return c

    @staticmethod
    def randomRotation(image, rotation_max_angle=30,mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, rotation_max_angle)
        return image.rotate(random_angle, mode)
 
    @staticmethod
    def randomCrop(image,crop_strength=0.2):
        """
        对图像随意剪切,裁剪图像大小宽和高的2/3
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_image_width = math.ceil(image_width*crop_strength)
        crop_image_height = math.ceil(image_height*crop_strength)
        x = np.random.randint(0, image_width - crop_image_width)
        y = np.random.randint(0, image_height - crop_image_height) 
        random_region = (x, y, x + crop_image_width, y + crop_image_height)
        return image.crop(random_region)
 
    @staticmethod
    def randomColor(image, color_strength=30):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, color_strength) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, color_strength) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, color_strength) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, color_strength) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
 
    @staticmethod
    def randomGaussian(image, mean=0.2):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """
        mean, sigma = mean,1.

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im
 
        # 将图像转化成数组
        img = np.array(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        # try:
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        # except:
        #     img = img
        return Image.fromarray(np.uint8(img))
 
    @staticmethod
    def randomCutout(image, length=20):
        """
         对图像进行遮挡处理
        :param image:
        :return:
        """
        image = np.array(image)
        image.flags.writeable = True

        h, w, c = image.shape
        mask = np.ones((h, w), np.float32)

        n_holes=np.random.randint(1,4)

        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - length // 2, 0, h))
            y2 = int(np.clip(y + length // 2, 0, h))
            x1 = int(np.clip(x - length // 2, 0, w))
            x2 = int(np.clip(x + length // 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, c, axis=2)
        image = image * mask
        return Image.fromarray(np.uint8(image))
 
    @staticmethod
    def randomErasing(image, region_strength=0.2):
        """
         对图像进行遮挡处理
        :param image:
        :return:
        """
        image = np.array(image)
        image.flags.writeable = True

        region_h = math.ceil(image.shape[0]*region_strength)
        region_w = math.ceil(image.shape[1]*region_strength)

        x1 = random.randint(0, image.shape[1] - region_w)
        y1 = random.randint(0, image.shape[0] - region_h)

        image[y1:y1+region_h, x1:x1+region_w, 0] = np.random.randint(0, 255, size=(region_h, region_w))
        image[y1:y1+region_h, x1:x1+region_w, 1] = np.random.randint(0, 255, size=(region_h, region_w))
        image[y1:y1+region_h, x1:x1+region_w, 2] = np.random.randint(0, 255, size=(region_h, region_w))

        return Image.fromarray(np.uint8(image))
    
    @staticmethod
    def saveImage(image, path):
        try:
            image.save(path)
        except:
            print('not save img: ', path)
            pass



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_dir = ""
    subfolder = 'task_ABC_D'
    seq_len = 10
    act_len = 5
    DS0 = CalvinDataset_Policy(
        data_dir,
        seq_len,
        act_len,
        forward_n_max=25,
        mode='train',
        subfolder=subfolder,
        use_data_augmentation=False,
        use_play=False,
        task_num=10000)
    
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std  = np.array([0.229, 0.224, 0.225])
    for i in range(0, len(DS0),100):
        data = DS0[i]
        text = data['text']
        goal_rgb = data['goal_rgb']
        rgb = data['rgb']
        hand_rgb = data['hand_rgb']
        rel_state = data['rel_state']
        action = data['action']
        print(f"{text}")
        fig, ax = plt.subplots(seq_len // 3 + 1, 3)
        for k in range(seq_len):
            temp_rgb = rgb[k].permute(1, 2, 0).numpy()
            temp_rgb = temp_rgb * rgb_std + rgb_mean
            temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
            ax[k // 3, k % 3].imshow(temp_rgb)
        temp_rgb = goal_rgb.permute(1, 2, 0).numpy()
        temp_rgb = temp_rgb * rgb_std + rgb_mean
        temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
        ax[seq_len // 3, 2].imshow(temp_rgb)
        plt.savefig("debug_goal_rgb.png", dpi=300)

        fig, ax = plt.subplots(seq_len // 3 + 1, 3)
        for k in range(seq_len):
            temp_rgb = hand_rgb[k].permute(1, 2, 0).numpy()
            temp_rgb = temp_rgb * rgb_std + rgb_mean
            temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
            ax[k // 3, k % 3].imshow(temp_rgb)
        plt.savefig("debug_goal_hand_rgb.png", dpi=300)
        
        accumulated_action = np.zeros(7)
        # the following is to check whether the computation of relative actions is correct
        for k in range(seq_len-1):
            curr_action = action[k, 0].numpy()
            curr_xyz = curr_action[0:3] / 50.0
            curr_rpy = curr_action[3:6] / 20.0
            curr_rotm = euler2rotm(curr_rpy)
            curr_gripper = curr_action[-1]

            accumulated_xyz = accumulated_action[0:3]
            accumulated_rpy = accumulated_action[3:6]
            accumulated_rotm = euler2rotm(accumulated_rpy)

            accumulated_xyz += np.dot(accumulated_rotm, curr_xyz)
            accumulated_rotm = accumulated_rotm @ curr_rotm
            accumulated_rpy = rotm2euler(accumulated_rotm) 

            next_rel_state = rel_state[k+1].numpy()
            assert np.allclose(curr_gripper, next_rel_state[-1]), "gripper"
            assert np.allclose(accumulated_xyz, next_rel_state[0:3]), "xyz"
            assert np.allclose(accumulated_rpy, next_rel_state[3:6]), "rpy"
            accumulated_action[0:3] = accumulated_xyz
            accumulated_action[3:6] = accumulated_rpy
            accumulated_action[-1] = curr_gripper