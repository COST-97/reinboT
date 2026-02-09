# import torch
# import logging
from typing import List
import matplotlib.pyplot as plt

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
# import bisect
# from PIL import Image
import json


class ImageSimilarityCalculator:
    def __init__(self):
        self.orb = cv2.ORB_create(
            edgeThreshold=0,
            fastThreshold=40
        )

    def calculate_mse(self, img1, img2):
        """
        Calculate MSE between two images.
        
        Args:
            img1 (numpy.ndarray): The first image.
            img2 (numpy.ndarray): The second image.
        
        Returns:
            float: The MSE value between the two images.
        """
        return mean_squared_error(img1, img2)

    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM between two images.
        
        Args:
            img1 (numpy.ndarray): The first image.
            img2 (numpy.ndarray): The second image.
        
        Returns:
            float: The SSIM value between the two images.
        """
        return ssim(img1, img2, channel_axis=-1)

    def calculate_orb_similarity(self, img1, img2, save_image=False):
        """
        Calculate similarity between two images using ORB feature matching.
        
        Args:
            img1_path (str): Path to the first image.
            img2_path (str): Path to the second image.
        
        Returns:
            float: The ratio of matched points to total points as a measure of similarity.
        """
        # Read images
        # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        # img1 = Image.fromarray(img1).convert('L')
        # img2 = Image.fromarray(img2).convert('L')

        img1_original = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_original = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        img1 = cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0, None  # No keypoints detected
        
        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # print("matches:",matches)

        img_matches=None
        if save_image:
            # 绘制前N个匹配项
            N = 500  # 选择前N个最好的匹配项
            # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
            img_matches = cv2.drawMatches(img1_original, kp1, 
                                          img2_original, kp2, 
                                          matches[:N], None, flags=2)


        # Calculate the ratio of matched points to total points
        num_matches = len(matches)
        num_keypoints = min(len(kp1), len(kp2))
        # print("num_matches:",num_matches)
        # print("num_keypoints:",num_keypoints)
        
        similarity_ratio = num_matches / num_keypoints if num_keypoints > 0 else 0.0
        
        return similarity_ratio, img_matches

    def calculate_visual_reward(self, img1, img2, save_image=False):
        mse_value = self.calculate_mse(img1, img2)
        ssim_value = self.calculate_ssim(img1, img2)
        orb_similarity,img_matches = self.calculate_orb_similarity(img1, 
                                                             img2,
                                                             save_image=save_image
                                                             )
        r_mse = np.exp(-0.01*mse_value)
        r_ssim = np.exp(1.*(ssim_value-1.))
        r_orb = np.exp(1.*(orb_similarity-1.))
        return r_mse, r_ssim, r_orb, img_matches

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs["gripper_open"] == demo[i + 1]["gripper_open"] and
             obs["gripper_open"] == demo[i - 1]["gripper_open"] and
             demo[i - 2]["gripper_open"] == demo[i - 1]["gripper_open"]))
    small_delta = np.allclose(obs["joint_velocities"], 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def keypoint_discovery(demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0]["gripper_open"]
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs["gripper_open"] != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs["gripper_open"]
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        # logging.debug('Found %d keypoints.' % len(episode_keypoints),
        #               episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 定义期望的命令行参数
parser.add_argument('--mode', help='Output file', default='training')

# 解析命令行参数
args = parser.parse_args()

mode=args.mode

data_name = "task_ABC_D"
data_root = "/PATH/TO/calvin/{}/{}/".format(data_name,mode)

lang_ann = np.load(data_root+"lang_annotations/auto_lang_ann.npy",allow_pickle=True).item()


ann=lang_ann['language']['ann']
task=lang_ann['language']['task']
indx=lang_ann['info']['indx']

reward_dict = {}

for ind,(start,end) in enumerate(indx):
    # if ind != 8592:
    #     continue

    print("ind:",ind)

    # if ind>3:
    #     break

    # val_annotations.append(ann[ind])
    # if ann[ind] != "take the red block and rotate it left":
    #     continue
    # if ann[ind] != "remove the top block":
    #     continue
    # if ann[ind] != "take the pink block and turn it right":
    #     continue

    # print("ann_ind",ann[ind])
    # print("task_ind",task[ind])
    rgb_image_static = []
    rgb_image_gripper = []
    state_action = []
    
    tcp_position_list = []
    tcp_orientation_list = []
    gripper_opening_width_list = []
    arm_joint_states_list = []
    gripper_action_list = []
    action_list = []
    
    for data_ind in range(start,end):
        data = np.load(data_root+"episode_{:0>7d}.npz".format(data_ind))
        # initial_state.append((data["robot_obs"],data["scene_obs"]))
        rgb_image_static.append(data["rgb_static"])
        rgb_image_gripper.append(data["rgb_gripper"])
        
        # done = np.array([1.]) if data_ind==end else np.array([0.])
        # sa = np.concatenate((data["robot_obs"],data["rel_actions"],done),axis=-1)
        sa = np.concatenate((data["robot_obs"],data["rel_actions"]),axis=-1)
        tcp_position = data["robot_obs"][0:3]
        tcp_orientation = data["robot_obs"][3:6]
        gripper_opening_width = data["robot_obs"][6:7]
        arm_joint_states = data["robot_obs"][7:14]
        gripper_action = data["robot_obs"][14:15]   

        tcp_position_list.append(tcp_position)
        tcp_orientation_list.append(tcp_orientation)
        gripper_opening_width_list.append(gripper_opening_width)
        arm_joint_states_list.append(arm_joint_states)
        gripper_action_list.append(gripper_action)
        action_list.append(data["rel_actions"])
    
    # pos = np.array(tcp_position_list)
    # vel=np.diff(pos,axis=0)
    # print("shape: ",np.shape(pos),np.shape(vel))
    diff_arr = np.zeros_like(arm_joint_states_list)
    # print("diff_arr shape:",np.shape(diff_arr))

    diff_arr[1:] = np.diff(arm_joint_states_list, axis=0)
    # print("diff_arr: ", np.max(diff_arr),np.min(diff_arr))

    diff_arr_2 = np.zeros_like(arm_joint_states_list)
    diff_arr_2[1:] = np.diff(diff_arr, axis=0)

    error_action = np.zeros_like(action_list)
    error_action[1:] = np.diff(action_list, axis=0)
    error_action_2 = np.zeros_like(action_list)
    error_action_2[1:] = np.diff(error_action, axis=0)

    error_vel = np.zeros_like(tcp_position_list)
    error_vel[1:] = np.diff(tcp_position_list, axis=0)
    
    tcp_orientation_list_modified = tcp_orientation_list
        # if i>0:
        #     for j in range(3):
        #         if  np.abs(tcp_orientation_list_modified[i][j]-tcp_orientation_list_modified[i-1][j])>5.5:
        #             tcp_orientation_list_modified[i][j] = np.abs(tcp_orientation_list_modified[i][j])*np.sign(tcp_orientation_list_modified[subgoal_id][0])
    
    for i in range(1,len(tcp_orientation_list_modified)):
        for j in range(3):
            if np.abs(tcp_orientation_list_modified[i][j]-tcp_orientation_list_modified[i-1][j])>5.:
                # tcp_orientation_list_modified[i][j]=tcp_orientation_list_modified[i-1][j]    
                if tcp_orientation_list_modified[i][j]>2.:
                    tcp_orientation_list_modified[i][j] = tcp_orientation_list_modified[i][j] - 6.28
                elif tcp_orientation_list_modified[i][j]<-2.:
                    tcp_orientation_list_modified[i][j] = tcp_orientation_list_modified[i][j] + 6.28

    error_ori_vel = np.zeros_like(tcp_orientation_list_modified)
    error_ori_vel[1:] = np.diff(tcp_orientation_list_modified, axis=0)
    # print("error_ori_vel",ind,error_ori_vel)
    if (error_ori_vel<-2.).any():
        print("Be careful! ",ann[ind])

    demo=[]
    for i in range(len(gripper_action_list)):
        demo.append({
            "joint_velocities": diff_arr[i],
            "gripper_open":gripper_action_list[i],
        })
    episode_keypoints = keypoint_discovery(demo, stopping_delta=0.005)
    print("episode_keypoints:",episode_keypoints)

    # reward calculation by using keypoints
    calculator = ImageSimilarityCalculator()

    # Calculate MSE
    reward_list = []
    for i in range(end-start):
        reward_subgoal = 0.
        for subgoal_ind in episode_keypoints:
            reward_subgoal += 1/len(episode_keypoints)
            if i<=subgoal_ind:
                subgoal_id=subgoal_ind
                break

        # subgoal_id = bisect.bisect_left(episode_keypoints, i)
        # print("i, subgoal id: ",i,subgoal_id)
        # mse_value = calculator.calculate_mse(rgb_image_static[i], rgb_image_static[subgoal_id])
        # ssim_value = calculator.calculate_ssim(rgb_image_static[i], rgb_image_static[subgoal_id])
        # orb_similarity,img_matches = calculator.calculate_orb_similarity(rgb_image_static[i], 
        #                                                      rgb_image_static[subgoal_id],
        #                                                      save_image=False
        #                                                      )
        save_image=False
        mse_value,ssim_value,orb_similarity,img_matches =calculator.calculate_visual_reward( rgb_image_static[i], 
                                                                                            rgb_image_static[subgoal_id], 
                                                                                            save_image=save_image)
        # print(f"ORB Similarity: {orb_similarity}")
        if img_matches is not None: cv2.imwrite('my_test_keypoint/matches_result_{}.png'.format(str(i)), img_matches)
        
        # gripper
        # mse_value_gripper = calculator.calculate_mse(rgb_image_gripper[i], rgb_image_gripper[subgoal_id])
        # ssim_value_gripper = calculator.calculate_ssim(rgb_image_gripper[i], rgb_image_gripper[subgoal_id])
        # orb_similarity_gripper,img_matches_gripper = calculator.calculate_orb_similarity(rgb_image_gripper[i], 
        #                                                      rgb_image_gripper[subgoal_id],
        #                                                      save_image=False
        #                                                      )
        mse_value_gripper,ssim_value_gripper,orb_similarity_gripper,img_matches_gripper =calculator.calculate_visual_reward(rgb_image_gripper[i], 
                                                                                             rgb_image_gripper[subgoal_id], 
                                                                                            save_image=save_image)
        
        # print(f"ORB Similarity: {orb_similarity}")
        if img_matches_gripper is not None: 
            print("str(i)",str(i))
            cv2.imwrite('my_test_keypoint/matches_result_{}_gripper.png'.format(str(i)), img_matches_gripper)
        

        error_pos = np.linalg.norm(tcp_position_list[i]-tcp_position_list[subgoal_id])
        error_pos = np.exp(-1.*error_pos)

        error_ori = np.linalg.norm(tcp_orientation_list_modified[i]-tcp_orientation_list_modified[subgoal_id])
        error_ori = np.exp(-1.*error_ori)

        error_joint = np.linalg.norm(arm_joint_states_list[i]-arm_joint_states_list[subgoal_id])
        error_joint = np.exp(-1.*error_joint)

        # smoothing
        error_pos_vel = np.sum(np.abs(error_vel[i]))
        error_ori_angle_vel = np.sum(np.abs(error_ori_vel[i]))
        error_joint_acc = np.sum(np.abs(diff_arr_2[i]))        
        error_joint_vel = np.sum(np.abs(diff_arr[i]))
        error_joint_acc = np.sum(np.abs(diff_arr_2[i]))
        error_action_vel = np.sum(np.abs(error_action[i]))
        error_action_acc = np.sum(np.abs(error_action_2[i]))

        # print("diff_arr: ", np.max(diff_arr),np.min(diff_arr))
        # print(            error_pos,
        #                   error_ori,
        #                   error_joint,
                          
        #                   # auxiliary rewards
        #                   error_joint_vel,
        #                   error_joint_acc,
        #                   error_action_vel,
        #                   error_action_acc,
                        
        #                   # sub-goal progress
        #                   reward_subgoal)
        
        reward_single = np.array([
                          # image goal tracking
                          mse_value,
                          ssim_value,
                          orb_similarity,
                          mse_value_gripper,
                          ssim_value_gripper,
                          orb_similarity_gripper,
                          # prop. goal tracking
                          error_pos,
                          error_ori,
                          error_joint,
                          
                          # auxiliary rewards
                            error_pos_vel,
                            error_ori_angle_vel,
                          error_joint_vel,
                          error_joint_acc,
                          error_action_vel,
                          error_action_acc,
                        
                          # sub-goal progress
                          reward_subgoal
                          ])
        reward_weight = np.array([
                                      # image goal tracking
                                    1/len(reward_single),
                                  1/len(reward_single),
                                  1/len(reward_single),
                                  1/len(reward_single),
                                  1/len(reward_single),
                                  1/len(reward_single),

                                   # prop. goal tracking
                                    1/len(reward_single),
                                    1/len(reward_single),
                                    1/len(reward_single),
                                    
                                    # auxiliary rewards
                                    -1/len(reward_single)*10**(1),
                                    -1/len(reward_single)*10**(1),
                                    -1/len(reward_single)*10**(1),
                                    -1/len(reward_single)*10**(1),
                                    -1/len(reward_single)*10**(-1),
                                    -1/len(reward_single)*10**(-1),

                                    # sub-goal progress
                                    1/len(reward_single)
                                  ])
        reward_total = reward_weight*reward_single
        # reward_list.append(np.sum(reward_total))
        # print("reward_total",reward_total)
        # print("reward_total sum",np.sum(reward_total,keepdims=True))
        # print("np.concatenate(reward_total,np.sum(reward_total,keepdims=True))",np.concatenate(reward_total,np.sum(reward_total,keepdims=True)))
        reward_list.append(np.concatenate((reward_total,np.sum(reward_total,keepdims=True))))

    reward_list=np.round(np.array(reward_list),6)

    reward_dict[ind]={
        "episode_keypoints":episode_keypoints,
        "reward":reward_list.tolist()
    }

reward_dict = json.dumps(reward_dict)
with open('reward_dict_ABC_D_{}.json'.format(mode), 'w') as f:
    json.dump(reward_dict, f, indent=4)

    # reward_list[:,-1] = (reward_list[:,-1] - np.min(reward_list[:,-1]))/(np.max(reward_list[:,-1]) - np.min(reward_list[:,-1]))

    # reward_single_name = [
    #                       # image goal tracking
    #                       "mse_value",
    #                       "ssim_value",
    #                       "orb_similarity",
    
    #                       "mse_value_gripper",
    #                       "ssim_value_gripper",
    #                       "orb_similarity_gripper",

    #                       # prop. goal tracking
    #                       "error_pos",
    #                       "error_ori",
    #                       "error_joint",
                          
    #                       # auxiliary rewards
    #                       "error_pos_vel",
    #                       "error_ori_angle_vel",
    #                       "error_joint_vel",
    #                       "error_joint_acc",
    #                       "error_action_vel",
    #                       "error_action_acc",
                        
    #                       # sub-goal progress
    #                       "reward_subgoal",
    #                       "reward_total"
    #                       ]
    
    # plt.figure(figsize=(40,14))
    # for i in range(len(reward_single_name)):
    #     ax=plt.subplot(4,5,i+1)
    #     ax.plot(reward_list[:,i])
    #     ax.set_ylabel(reward_single_name[i], fontsize=20)
        
    #     plt.tick_params(axis='both', which='major', labelsize=16)
    #     for j in range(len(episode_keypoints)):
    #         ax.axvline(x=episode_keypoints[j], color='black', linestyle='--')

    # plt.tight_layout()
    # plt.savefig("my_test_keypoint/reward/keypoint_{}_{}_pos_keypoint_reward.png".format(ind, ann[ind]))
    # plt.close()

    # print("Finish {}".format(ind))


    data_list = [
        tcp_position_list,
        # np.diff(np.diff(np.array(tcp_position_list),axis=0),axis=0),
        tcp_orientation_list_modified,
        # np.diff(np.diff(np.array(tcp_orientation_list),axis=0),axis=0),
        gripper_opening_width_list,
        # arm_joint_states_list,
        diff_arr,
        # np.diff(np.diff(np.array(arm_joint_states_list),axis=0),axis=0),
        gripper_action_list,
        action_list
    ]

    data_list_name = [
        "tcp_position",
        # "tcp_position_vel",
        #  "tcp_position_acc",
        "tcp_orientation",
        # "tcp_orientation_vel",
        # "tcp_orientation_acc",
        "gripper_opening_width",
        "arm_joint_states",
        # "arm_joint_states_vel",
        # "arm_joint_states_acc",
        "gripper_action",
        "action"
    ]
    plt.figure(figsize=(16,8))
    for i in range(6):
        ax=plt.subplot(2,3,i+1)
        ax.plot(data_list[i])
        ax.set_ylabel(data_list_name[i])
        for j in range(len(episode_keypoints)):
            ax.axvline(x=episode_keypoints[j], color='black', linestyle='--')

    plt.savefig("my_test_keypoint/keypoint_{}_{}_pos_keypoint.png".format(ind, ann[ind]))
    print("Finish {}".format(ind))



"""
pip install opencv-python-headless -i https://pypi.mirrors.ustc.edu.cn/simple/ && pip install scikit-image -i https://pypi.mirrors.ustc.edu.cn/simple/ && cd /reinbot/ICML2025/GR-MG && python reward_calculation.py

cd /reinbot/ICML2025/GR-MG && python reward_calculation.py --mode training
cd /reinbot/ICML2025/GR-MG && python reward_calculation.py --mode validation
"""
