# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import re
import copy
from copy import deepcopy
import os
# python_path = sys.executable
# print("Python executable path:", python_path)

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

import matplotlib.pyplot as plt
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
from utils.utils import print_and_save
from wrapper.model_wrapper import CustomModel
 
logger = logging.getLogger(__name__)
EP_LEN = 360
NUM_SEQUENCES = 1000
SAVE_DIR = None
FAIL_COUNTER=0

def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    # insert your own env wrapper
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, ip2p_model,save_data_path):
    """Run this function to evaluate a model on the CALVIN challenge."""
    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = get_sequences(NUM_SEQUENCES)
    results = []
    sequence_i = 0
     
    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for index,(initial_state, eval_sequence) in enumerate(eval_sequences):
        result= evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, sequence_i,ip2p_model,save_data_path)
        results.append(result)
        # success_list = count_success(results)
        # with open(eval_sr_path, 'a') as f:
        #     line =f"{sequence_i}/{NUM_SEQUENCES}: "
        #     for sr in success_list:
        #         line += f"{sr:.3f} | "
        #     sequence_i += 1
        #     line += "\n"
        #     f.write(line)
        eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + f"ASSL : {np.mean(results)}" + "|"
            )

    eval_sequences = extract_iter_from_tqdm(eval_sequences)
    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    device_num=1
    # all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)]
    # torch.distributed.gather_object(res_tup, all_res_tup, dst=0)
    # if torch.distributed.get_rank() == 0:
    res_tup_list = merge_multi_list(all_res_tup)
    res_list = [_[0] for _ in res_tup_list]
    eval_seq_list = [_[1] for _ in res_tup_list]
    print_and_save(res_list, eval_seq_list, eval_result_path, 0)

    #     if index%100==0 and index!=0: #save every 100 sequences
    #         print_and_save(results, eval_sequences[:index+1], eval_result_path[:-5]+f"_{index+1}"+".json", None)
    # print_and_save(results, eval_sequences, eval_result_path, None)
    
    return results

def extract_iter_from_tqdm(tqdm_iter):
    return [_ for _ in tqdm_iter]

def merge_multi_list(res):
    tmp = []
    for l in res:
        tmp.extend(l)
    return tmp

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, sequence_i,ip2p_model,save_data_path):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, subtask_i, sequence_i,ip2p_model,save_data_path)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, subtask_i, sequence_i,ip2p_model,save_data_path):
    """Run the actual rollout on one subtask."""
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    debug_image=[]
    progress=0
    rtg_list=[]
    time_model_list=[]
    for i in range(EP_LEN):
        goal_image=None
        if ip2p_model is not None:
            time_model_inference_start = time.time()
            if i % 20 == 0:  # hardcode
                static_rgb = obs['rgb_obs']['rgb_static'] # (200, 200, 3)
                hand_rgb = obs['rgb_obs']['rgb_gripper']
                image_patch=[static_rgb]
                text_patch=[lang_annotation + f".And {progress}% of the instruction has been finished."]
                # print(text_patch)
                goal_image=ip2p_model.inference(image_patch,text_patch)
                # temp_image=[static_rgb,goal_image[0],hand_rgb]
                # debug_image.append(temp_image)

            time_model_inference_start_step = time.time()
            action,progress,rtg = model.step(obs,deepcopy(goal_image),[lang_annotation],True) 
            time_model_inference_end = time.time()
            # print("low-level model inference time:",time_model_inference_end-time_model_inference_start_step)
            # print("model inference time:",time_model_inference_end-time_model_inference_start)

            rtg_list.append(rtg)
        else:
            time_model_inference_start = time.time()
            action,progress,rtg = model.step(obs,deepcopy(goal_image),[lang_annotation]) 
            time_model_inference_end = time.time()
            rtg_list.append(rtg)
            
        time_model = time_model_inference_end - time_model_inference_start
        time_model_list.append(time_model)

        obs, _, _, current_info = env.step(action)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            # print("success!")
            save_datas(subtask,i,rtg_list,time_model_list,save_data_path)
            return True
    
    save_datas(subtask,i,rtg_list,time_model_list,save_data_path)

    # print("fail!")
    
    # global FAIL_COUNTER
    # FAIL_COUNTER+=1
    # if FAIL_COUNTER % 30 ==0:  # save every 30 failure cases
    #     length=len(debug_image) 
    #     fig, ax = plt.subplots(length, 2,figsize=(5.5, 46.58))
    #     for ax_ in ax.flat:
    #         # ax_.plot([1, 2, 3], [4, 5, 6])
    #         ax_.axis('off')  # 隐藏每个子图的刻度和边框
    #     for i in range(length):
    #         ax[i][0].imshow(debug_image[i][0])
    #         ax[i][1].imshow(debug_image[i][1])
    #         # ax[i][2].imshow(debug_image[i][2])
    #     plt.tight_layout()
    #     plt.axis('off')
    #     plt.savefig(os.path.join(SAVE_DIR, f"{sequence_i}-{subtask_i}-{subtask}.png"),dpi=100)
    #     plt.close()

    return False

def save_datas(task,epi_length,rtg_list,time_model_list,save_data_path):   
    if save_data_path is not None:
        file_path = os.path.join(save_data_path, 'tasks.txt')
        with open(file_path, 'a') as file:
            file.write(task + '\n')
        file_path = os.path.join(save_data_path, 'episode_length.txt')
        with open(file_path, 'a') as file:
            np.savetxt(file, np.array([epi_length]), fmt='%d')
        
        file_path = os.path.join(save_data_path, 'rtg.txt')
        with open(file_path, 'a') as file:
            np.savetxt(file, rtg_list, fmt='%f')
        
        file_path = os.path.join(save_data_path, 'time_model.txt')
        with open(file_path, 'a') as file:
            np.savetxt(file, time_model_list, fmt='%f')

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", default='/mnt/bn/robotics/manipulation_data/calvin_data/task_ABC_D',
                        type=str, help="Path to the dataset root directory.")  # modify it before opensource
    # evaluation
    parser.add_argument('--config_path', type=str, default="", help='path to the policy config file')
    parser.add_argument('--ckpt_dir', type=str, default="",help="path to the policy ckpt file")
    parser.add_argument('--epoch', type=int,default=41, help="epoch index for evaluating")
    parser.add_argument('--device_id', default=0, type=int, help="CUDA device")
    parser.add_argument('--RTG_pred', default=0, type=int)
    parser.add_argument('--reward_dim', default=1, type=int)
    parser.add_argument('--vision_latent_conditioned', default=0, type=int)
    parser.add_argument('--RTG_decoder_output_embeddings_repeat_num', default=1, type=int)
    parser.add_argument('--returnToGo_pred_conditioned', default=1, type=int)
    parser.add_argument('--use_goal_rgb', default=1, type=int)
    parser.add_argument('--ip2p_ckpt_path', default=None, type=str, help="ip2p ckpt path")
    parser.add_argument('--save_data_path', default=None, type=str)
    
    args = parser.parse_args()
    
    config_path = args.config_path
    ckpt_dir = args.ckpt_dir

    epoch = args.epoch
    device_id = args.device_id
    ip2p_ckpt_path=args.ip2p_ckpt_path
    assert config_path != None
    # Load config file
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    configs["RTG_pred"]=args.RTG_pred
    configs["reward_dim"]=args.reward_dim
    configs["vision_latent_conditioned"]=args.vision_latent_conditioned
    configs["RTG_decoder_output_embeddings_repeat_num"]=args.RTG_decoder_output_embeddings_repeat_num
    configs["returnToGo_pred_conditioned"]=args.returnToGo_pred_conditioned
    configs["use_goal_rgb"]=args.use_goal_rgb
    configs["save_data_path"]=args.save_data_path

    # Get checkpoint path
    ckpt_path = None
    ckpt_files = os.listdir(ckpt_dir)
    for ckpt_file in ckpt_files:
        match = re.search(r'epoch=(\d+)', ckpt_file)
        if match:
            temp_epoch = int(match.group(1))
            if temp_epoch == epoch:
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                break

    print("ckpt_path:",ckpt_path)

    device = torch.device('cuda', device_id)

    model = CustomModel(
        ckpt_path=ckpt_path,
        configs=configs,
        device=device)
    
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']} 
    env = make_env(args.dataset_path, observation_space, device_id) 
    # Success rate and result files
    flag="opensourcesd"
    sub_dir=f"{flag}_{epoch}_epoch"
    # set a global variable
    global SAVE_DIR
    SAVE_DIR=os.path.join(ckpt_dir,sub_dir)
    print("SAVE_DIR:",SAVE_DIR)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    sr_path = os.path.join(SAVE_DIR, f"success_rate.txt")
    result_path = os.path.join(SAVE_DIR, f"results.json")

    ip2p_model=None
    if ip2p_ckpt_path is not None:
        from goal_gen.evaluate import IP2PEvaluation
        ip2p_model=IP2PEvaluation(ip2p_ckpt_path)
    
    evaluate_policy(
        model, 
        env,
        eval_sr_path=sr_path,
        eval_result_path=result_path,
        ip2p_model=ip2p_model,
        save_data_path=args.save_data_path
        )
if __name__ == "__main__":
    main()