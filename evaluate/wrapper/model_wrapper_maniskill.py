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
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
import policy.model.vision_transformer as vits
from utils.utils import euler2rotm, rotm2euler
from copy import deepcopy
from policy.model.model import reinbot
import time
import clip
from configs.state_vec import STATE_VEC_IDX_MAPPING

GRIPPER_OPEN = 1
GRIPPER_CLOSE = 0

MANISKILL_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]
DATA_STAT = {'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0], 'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033], 'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0], 'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0]}

class CustomManiSkillModel():
    def __init__(self,
                 ckpt_path,
                 configs,
                 device):
        self.device = device
        # model config
        self.configs = configs
        self.seq_len = configs["policy"]['seq_len']
        self.act_len = configs["policy"]["act_len"]
        self.device = device
        input_size = (224, 224)
        clip_mean = (0.485, 0.456, 0.406)
        clip_std = (0.229, 0.224, 0.225)
        self.preprocess = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(clip_mean, clip_std)])
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        model_mae.to(self.device)
        training_target = []
        if configs["trainer"]['act_pred']:
            training_target.append('act_pred')
        if configs["trainer"]['fwd_pred']:
            training_target.append('fwd_pred')
        if configs["trainer"]['fwd_pred_hand']:
            training_target.append('fwd_pred_hand')
        if configs["trainer"]["progress_pred"]:
            training_target.append('progress_pred')
        if configs["RTG_pred"]:
            training_target.append('RTG_pred')
        print(f"training target: {training_target}")

        self.state_min = torch.tensor(DATA_STAT['state_min']).to(device)
        self.state_max = torch.tensor(DATA_STAT['state_max']).to(device)
        self.action_min = torch.tensor(DATA_STAT['action_min']).to(device)
        self.action_max = torch.tensor(DATA_STAT['action_max']).to(device)
        self.state_token_dim = 128
        self.control_frequency = 25

        #modify before release
        #clip model
        # clip_name = "ViT-B/32"
        clip_name="/PATH/TO/model_ckpt/ViT-B-32.pt"
        clip_model, clip_preprocess = clip.load(clip_name)
        # freeze clip
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        policy_config = self.configs['policy']
        input_config=self.configs["input"]
        trainer_config=self.configs["trainer"]
        self.policy = reinbot(
                state_dim=input_config['state_dim'],
                act_dim=input_config['act_dim'],
                reward_dim=configs['reward_dim'],
                act_len=policy_config['act_len'],
                act_latent_dim=policy_config['act_latent_dim'],
                act_encoder_dim=policy_config['act_encoder_dim'],
                act_decoder_dim=policy_config['act_decoder_dim'],
                progress_decoder_dim=self.configs["policy"]["progress_decoder_dim"],
                returnToGo_decoder_dim=self.configs["policy"]["returnToGo_decoder_dim"],
                hidden_size=policy_config['embed_dim'],
                model_mae=model_mae,
                clip_model=clip_model,
                img_feat_dim=policy_config["img_feat_dim"],
                lang_feat_dim = policy_config["lang_feat_dim"],
                patch_feat_dim=policy_config["patch_feat_dim"],
                resampler_params=policy_config['resampler_params'],
                max_length=policy_config['seq_len'],
                training_target=training_target,
                without_norm_pix_loss=trainer_config['without_norm_pix_loss'],
                use_hand_rgb=input_config['use_hand_rgb'],
                use_state=input_config['use_state'],
                use_resampler=policy_config['use_resampler'],
                n_layer=policy_config['n_layer'],
                n_head=policy_config['n_head'],
                n_inner=4*policy_config['embed_dim'],
                activation_function=policy_config['activation_function'],
                n_positions=1024,
                resid_pdrop=policy_config['dropout'],
                attn_pdrop=policy_config['dropout'],
                vision_latent_conditioned=configs["vision_latent_conditioned"],
                RTG_decoder_output_embeddings_repeat_num=configs["RTG_decoder_output_embeddings_repeat_num"],
                returnToGo_pred_conditioned=configs["returnToGo_pred_conditioned"],
                device=self.device)
        
        # Set up the model
        payload = torch.load(ckpt_path)
        epoch = payload['epoch']
        state_dict = payload['state_dict']
        print(f"loading state dict from epoch {epoch}...")

        del payload
        # Remove the prefix "model." from pl models
        pure_state_dict = dict()
        for k, v in state_dict.items():
            if "model." in k:
                new_k = k[6:]
                pure_state_dict[new_k] = v
        msg = self.policy.load_state_dict(pure_state_dict, strict=True)
        print(msg)
        self.policy.to(self.device)
        self.policy.eval()

    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0

    def _format_joint_to_state(self, joints):
        """
        Format the robot joint state into the unified state vector.

        Args:
            joints (torch.Tensor): The joint state to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted state for RDT ([B, N, 128]). 
        """
        # Rescale the gripper
        # joints = joints / torch.tensor(
        #     [[[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]],
        #     device=joints.device, dtype=joints.dtype
        # )
        
        # normalize to -1,1
        joints = (joints - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.state_token_dim), 
            device=joints.device, dtype=joints.dtype
        )
        # assemble the unifed state vector
        state[:, :, MANISKILL_INDICES] = joints
        state_elem_mask = torch.zeros(
            (B, self.state_token_dim),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, MANISKILL_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        action_indices = MANISKILL_INDICES
        joints = action[:, :, action_indices]
        
        # denormalize to action space 

        joints = (joints + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        
        return joints
    
    def step(self, obs, goal, text):
        """Step function."""
        rgb = obs['rgb_obs']['rgb_static'] # (200, 200, 3)
        # hand_rgb = obs['rgb_obs']['rgb_gripper']

        if goal is not None:
            goal_rgb = goal[0]
            goal_rgb = Image.fromarray(goal_rgb)
            goal_rgb = T.ToTensor()(goal_rgb.convert("RGB"))
            goal_rgb = self.preprocess(goal_rgb) # (3, 224, 224)

        rgb = Image.fromarray(rgb)
        rgb = T.ToTensor()(rgb.convert("RGB"))
        rgb = self.preprocess(rgb) # (3, 224, 224)
        self.rgb_list.append(rgb)

        # hand_rgb = Image.fromarray(hand_rgb)
        # hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
        # hand_rgb = self.preprocess(hand_rgb)
        # self.hand_rgb_list.append(hand_rgb)
        joints = torch.from_numpy(obs['robot_obs'])
        joints = joints.unsqueeze(0).unsqueeze(0)   # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(self.device), state_elem_mask.to(self.device)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(self.device )
        
        # state = obs['robot_obs'] # (15,)
        # xyz_state = state[:3]
        # rpy_state = state[3:6]
        # rotm_state = euler2rotm(rpy_state)
        # gripper_state = state[-1]
        # state = (xyz_state, rotm_state, gripper_state)
        self.state_list.append(states.squeeze())

        buffer_len = len(self.rgb_list)
        if buffer_len > self.seq_len:
            self.rgb_list.pop(0)
            # self.hand_rgb_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_list) == self.seq_len
            # assert len(self.hand_rgb_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_list)
        
        # Static RGB
        c, h, w = rgb.shape
        if goal is not None:
            c2,h2,w2=goal_rgb.shape
            assert c==c2 and h==h2 and w==w2

        rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_tensor = torch.stack(self.rgb_list, dim=0) # (len, c, h, w)
        rgb_data[0, :buffer_len] = rgb_tensor
        goal_rgb_data=torch.zeros((1, c, h, w))
        if goal is not None:
            goal_rgb_data[0]=goal_rgb

        # Hand RGB
        # c, h, w = hand_rgb.shape
        # hand_rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        # hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0) # (len, c, h, w)
        # hand_rgb_data[0, :buffer_len] = hand_rgb_tensor

        # State
        state = torch.cat(self.state_list,dim=0).unsqueeze(0)
        arm_state = torch.concatenate((state[:, :, :10],state[:, :, 11:]),dim=-1)  # (b, l, state_dim - 1)
        gripper_state = (state[:, :, 10]>0.5).long()   # (b, l)

        # arm_state, gripper_state = CustomModel.compute_rel_state(self.state_list)

        arm_state_data = torch.zeros((1, self.seq_len, 6))
        # arm_state_tensor = torch.from_numpy(arm_state)
        arm_state_data[0, :buffer_len] = arm_state

        # gripper_state_tensor = torch.from_numpy(gripper_state)
        # gripper_state_tensor = (gripper_state_tensor + 1.0) / 2.0
        # gripper_state_tensor = gripper_state_tensor.long()

        gripper_state_data = torch.zeros((1, self.seq_len)).long()
        gripper_state_data[0, :buffer_len] = gripper_state
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).type_as(arm_state_data)

        # Attention mask
        attention_mask = torch.zeros(1, self.seq_len).long()
        attention_mask[0, :buffer_len] = 1

        # Action placeholder
        arm_action_data = torch.zeros((1, self.seq_len, self.configs["policy"]['act_len'], self.state_token_dim-1))
        gripper_action_data = torch.zeros(1, self.seq_len, self.configs["policy"]['act_len'])

        #progress_placeholder
        progress_data=torch.zeros(1, self.seq_len)# do not need
        # RTG
        RTG_data=torch.zeros(1, self.seq_len)# do not need
        state_elem_mask_all = torch.zeros(1, self.state_token_dim-1) # do not need

        input_dict = dict()
        input_dict['rgb'] = rgb_data.to(self.device)
        # input_dict['hand_rgb'] = hand_rgb_data.to(self.device)
        input_dict["goal_rgb"]=goal_rgb_data.to(self.device)
        input_dict['arm_state'] = arm_state_data.to(self.device)
        input_dict['gripper_state'] = gripper_state_data.to(self.device)
        input_dict['arm_action'] = arm_action_data.to(self.device)
        input_dict['gripper_action'] = gripper_action_data.to(self.device)
        input_dict['attention_mask'] = attention_mask.to(self.device)
        input_dict["text"]=[text]
        input_dict["progress"]=progress_data
        input_dict["RTG"]=RTG_data
        input_dict['state_elem_mask']=state_elem_mask_all
        input_dict['ctrl_freq']=self.control_frequency
        
        # Forward pass
        with torch.no_grad():
            # action,action_traj = self.policy.evaluate(input_dict)
            # action,progress = self.policy.evaluate(input_dict,return_progress=True)
            # progress=int(int(progress * 10) *10)
            action = self.policy.evaluate_maniskill(input_dict,return_progress=False) # (1,act_len,act_dim)

        action = self._unformat_action_to_joint(action).to(torch.float32) # (1,act_len,8)
        self.rollout_step_counter += 1
    
        action = action.squeeze(0).cpu().numpy() # act_len, 8
        return action
