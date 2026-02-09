import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageChops

# from data.filelock import FileLock
# from data.hdf5_vla_dataset import HDF5VLADataset
# from train.image_corrupt import image_corrupt

import fcntl
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from data.image_corrupt import image_corrupt



class RandomAugmentations(nn.Module):
    def __init__(self):
        super(RandomAugmentations, self).__init__()
        self.shift_limit = 0.2
        self.rotate_limit = 30
        self.crop_limit = 0.2
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))

    def random_shift(self, img, shift_limit):
        angle = random.uniform(-shift_limit, shift_limit)
        img = TF.affine(img, angle=0, translate=(angle, 0), scale=1, shear=0)
        return img

    def random_rotation(self, img, rotate_limit):
        angle = random.uniform(-rotate_limit, rotate_limit)
        img = TF.affine(img, angle=angle, translate=(0, 0), scale=1, shear=0)
        return img

    def random_crop(self, img, crop_limit):
        height, width = img.shape[-2:]
        new_height = random.randint(int(height * (1 - crop_limit)), height)
        new_width = random.randint(int(width * (1 - crop_limit)), width)
        top = random.randint(0, height - new_height)
        left = random.randint(0, width - new_width)
        img = TF.crop(img, top, left, new_height, new_width)
        return img

    def random_color_jitter(self, img):
        img = self.color_jitter(img)
        return img

    def random_gaussian_blur(self, img):
        img = self.gaussian_blur(img)
        return img

    def forward(self, img):
        # Convert PIL Image to Tensor
        # img = TF.to_tensor(img)
        
        # Apply random transformations
        img = self.random_shift(img, self.shift_limit)
        # print("random_shift",img.size())
        img = self.random_rotation(img, self.rotate_limit)
        # print("random_rotation",img.size())
        img = self.random_crop(img, self.crop_limit)
        # print("random_crop",img.size())
        img = self.random_color_jitter(img)
        img = self.random_gaussian_blur(img)
        
        return img


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
        random_factor = np.random.randint(10, color_strength) / 10.  # 随机因子
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
        mean, sigma = mean,0.3

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



class FileLock:
    """
    A file lock class.
    """
    def __init__(self, filename):
        self.filename = filename
        self.handle = None

    def acquire_read_lock(self):
        self.handle = open(self.filename + '.lock', 'r')
        fcntl.flock(self.handle, fcntl.LOCK_SH | fcntl.LOCK_NB)

    def acquire_write_lock(self):
        self.handle = open(self.filename + '.lock', 'w')
        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release_lock(self):
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

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



def get_clean_item(chunk_dir):
    """
    Get indexes of clean items in a chunk.
    """
    dirty_bit = read_dirty_bit(chunk_dir)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    """
    Save the dirty bit to the chunk directory.
    """
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()
            with open(file_path, 'wb') as file:
                file.write(dirty_bit.tobytes())
            lock.release_lock()
            return
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    """
    Read the dirty bit from the chunk directory.
    """
    # If error occurs, retry
    time_stmp = time.time()
    while time.time() - time_stmp < 10.: #10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            # print("file_path:",file_path)

            lock = FileLock(file_path)
            lock.acquire_read_lock()
            with open(file_path, 'rb') as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
                # print("dirty_bit:",len(dirty_bit))
                # input()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")

class VLAConsumerDataset(Dataset):
    """A vision-languange-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """
    
    def __init__(
        self, 
        config,
        num_cameras=2,
        seq_len=1,
        act_len=1,
        action_dim=128,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type='pretrain',
        # cond_mask_prob=0.1,
        # cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        mode='train',
        use_reward=False,
        discounting_factor=0.9,
        SubGoalReasoning=0,
        reward_dim=1,
        hdf5_dataset=None,
        buffer_dir=None
    ):
        super(VLAConsumerDataset, self).__init__()
        
        # Load the control frequency for each dataset
        with open("configs/dataset_control_freq.json", 'r') as fp:
            self.control_freq = json.load(fp)

        # Load the dataset names
        dataset_names_cfg = 'configs/pretrain_datasets.json' \
            if dataset_type == 'pretrain' else 'configs/finetune_datasets.json'
        
        with open(dataset_names_cfg, 'r') as file:
            DATASET_NAMES = json.load(file)

        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}
        
        self.mode = mode
        if mode == 'validate':
            tag = "validation"
        elif mode == 'train':
            tag = "training"
        else:
            raise ValueError("Mode must be either train or validate")
        
        self.use_reward=use_reward
        self.discounting_factor=discounting_factor
        self.SubGoalReasoning=SubGoalReasoning
        self.reward_dim=reward_dim

        self.use_hdf5 = use_hdf5 # false
        self.hdf5_dataset = hdf5_dataset

        self.buffer_dir = buffer_dir

        self.num_chunks = config["buf_num_chunks"] # 512
        self.chunk_size = config["buf_chunk_size"] # 512
        
        # self.tokenizer_max_length = config["tokenizer_max_length"]
        self.image_aspect_ratio = config["image_aspect_ratio"] # pad
        self.act_len = act_len
        self.action_dim = action_dim
        
        self.state_noise_snr = state_noise_snr # none
        self.num_cameras = num_cameras # 3
        self.seq_len = seq_len # 2
        # self.cond_mask_prob = cond_mask_prob # 0.1
        # self.cam_ext_mask_prob = cam_ext_mask_prob # -1, meaning same with cond_mask_prob

            
        # self.use_precomp_lang_embed = use_precomp_lang_embed
        # if use_precomp_lang_embed:
        #     self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")
        
        # Load dataset stat
        # with open("configs/dataset_stat.json", 'r') as f:
        #     dataset_stat = json.load(f)
        # self.dataset_stat = dataset_stat
        
        # As we use MAE as Encoder, we need to resize and normalize
        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406) 
        self.clip_std = (0.229, 0.224, 0.225)
        
        self.image_aug=image_aug
        if self.image_aug:
            # self.funcMap=[DataAugmentation.randomColor,DataAugmentation.randomGaussian]
        
            self.static_rgb_preprocess_train = T.Compose([
                # RandomShiftsSingleAug(pad=10), # 10 for static rgb
                # RandomAugmentations(),
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                RandomShiftsSingleAug(pad=10),
                T.Normalize(self.clip_mean, self.clip_std)])
        else:
            self.static_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])

        self.static_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])

        # self.tokenizer = tokenizer
        self.image_size = image_size # None
        self.auto_adjust_image_brightness = auto_adjust_image_brightness # false
        self.image_aug = image_aug # false
        
        self.last_content = None
        self.last_meta = None
    
    def get_dataset_name2id(self):
        return self.dataset_name2id
    
    def get_dataset_id2name(self):
        return self.dataset_id2name
        
    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    @staticmethod
    def _load_data_from_chunk(chunk_dir, chunk_item_idx):
        # If error occurs, retry
        time_stmp = time.time()
        while time.time() - time_stmp < 10.: #10.0:
            try:
                locks = []
                file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
                # print("file_path:",file_path)

                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, 'r') as file:
                    json_content = json.load(file)
                lock.release_lock()
                file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
                # print("file_path:",file_path)

                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, 'rb') as file:
                    sample_dict = np.load(file)
                    meta = tuple(sample_dict.values())
                lock.release_lock()

                return json_content, meta
            
            except KeyboardInterrupt:
                for lock in locks:
                    lock.release_lock()
                raise KeyboardInterrupt
            except BaseException:
                for lock in locks:
                    lock.release_lock()
                continue
        raise RuntimeError("Failed to load sample.")

    def __len__(self) -> int:
        data_rate=1. if self.mode=="train" else 0.01
        # data_rate = 0.001
        return int(data_rate*self.num_chunks * self.chunk_size)

    def _safe_load(self, index):
        read_chunk_item_indices = []
        # Start searching from a random chunk
        read_chunk_idx = index // self.chunk_size
        # print("index:",index)

        while len(read_chunk_item_indices) == 0:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            # print("read_chunk_dir:",read_chunk_dir)

            try:
                read_chunk_item_indices = get_clean_item(read_chunk_dir)
                # print("read_chunk_item_indices",read_chunk_item_indices)

            except BaseException as e:
                # Print the error info
                print("Error catched when searching a clean chunk:", e)
                traceback.print_exc()
                read_chunk_item_indices = []
            
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks
        
        # read_chunk_item_index = random.choice(read_chunk_item_indices)
        # read_chunk_item_index = read_chunk_item_indices.pop()
        random_item_index = index % len(read_chunk_item_indices)
        read_chunk_item_index = read_chunk_item_indices[random_item_index]
        
        # 2024/12/3 zhy 
        # Modify the dirty bit
        # try:
        #     dirty_bit = read_dirty_bit(read_chunk_dir)
        #     dirty_bit[read_chunk_item_index] = 1
        #     save_dirty_bit(read_chunk_dir, dirty_bit)
        # except BaseException as e:
        #     # Print the error info
        #     print("Error catched when modifying the dirty bit:", e)
        #     traceback.print_exc()
        
        # load the sample
        try:
            # print("read_chunk_dir:",read_chunk_dir)
            # print("read_chunk_item_index",read_chunk_item_index)
            content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
            # print("content:",content)

            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            # Print the error info
            print("Error catched when loading sample:", e)
            traceback.print_exc()
            
            # If failed to load the data, return the last loaded data for robustness
            content, meta = self.last_content, self.last_meta

        return (content, *meta)
    
    def __getitem__(self, index):
        time_get_data_start = time.time()
        while True:
            data_dict = None
            try:
                if self.use_hdf5: # false
                    res = self.hdf5_dataset.get_item()
                    content = res['meta']
                    states = res['state']
                    actions = res['actions']
                    state_elem_mask = res['state_indicator']
                    image_metas = [
                        res['cam_high'], res['cam_high_mask'],
                        res['cam_right_wrist'], res['cam_right_wrist_mask'],
                        res['cam_left_wrist'], res['cam_left_wrist_mask'],
                    ]
                    state_std = res['state_std']
                    state_mean = res['state_mean']
                    state_norm = res['state_norm']
                    
                    image_goal = res['goal_rgb_data']
                    
                    if self.use_reward:
                        reward = res['reward']
                        episode_keypoints = res['episode_keypoints']
                else:
                    while True:
                        (content, step_id, states, _, actions, _, 
                        state_elem_mask, *image_metas, 
                        state_std, state_mean, state_norm,
                        reward,episode_keypoints,image_goal
                        ) = self._safe_load(index)

                        # print("actions.shape:",actions.shape)
                        # print("states.shape:",states.shape)

                        if self.mode==content["mode"]: break
                        else: 
                            index = (index + 1) % len(self)
                    
                    content['step_id']=step_id
                    actions = np.concatenate((states[1:],actions),axis=0)

                self.dataset_name = content['dataset_name']
                        
                if self.state_noise_snr is not None: # none
                    states += np.random.normal(
                        0.0, state_std / np.sqrt(10 ** (self.state_noise_snr / 10)), 
                        states.shape)
                    
                # ds_state_mean = np.array(self.dataset_stat[data_dict['dataset_name']]['state_mean'])
                # ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                # Randomly mask the states by the mean state
                returns_buffer=[]
                RTG_data=torch.zeros(self.seq_len,self.reward_dim).float()
                self.current_s_t = content['step_id']
                self.subgoal_dict = None
                
                action_buffer = []
                action_valid = []
                for k in range(0, self.seq_len + self.act_len - 1): # 1+1-1
                    temp_action = actions[k]
                    action_buffer += [temp_action]
                    action_valid += [1]

                    if self.use_reward: 
                        returns_single,RTG = self.get_reward(content['step_id']-self.seq_len+k,
                                                             episode_keypoints,
                                                             reward, 
                                                             labeled_reward=1)
                        returns_buffer += [returns_single]
                        if k<self.seq_len:
                            RTG_data[k] = RTG


                # Action and action mask
                actions = []
                action_masks = []
                returns = []
                for i in range(0, self.seq_len):
                    actions.append(action_buffer[i:(i+self.act_len)])
                    action_masks.append(action_valid[i:(i+self.act_len)])
                    if self.use_reward: returns.append(returns_buffer[i:(i+self.act_len)])

                actions = np.array(actions) # (len, act_len, act_dim)
                action_data = torch.from_numpy(actions)
                # action_data  = torch.zeros(self.seq_len, self.act_len, self.action_dim).float() # (len, act_len, action_dim)
                # action_data [:tlen] = actions
                action_masks = np.array(action_masks) # (len, act_len, act_dim)
                action_mask_data = torch.from_numpy(action_masks).long()
                # action_mask_data = torch.zeros(self.seq_len, self.act_len).long() # (len, act_len, action_dim)
                # action_mask_data[:tlen] = action_masks

                returns_data  = torch.zeros(self.seq_len, self.act_len, 1).float() # (len, act_len, 1)
                if self.use_reward:
                    returns = np.array(returns) # (len, act_len)
                    returns = torch.from_numpy(returns)
                    # print("returns",returns)
                    returns_data = returns


                # Attention mask (should be all 1 for full dataset)
                attention_mask = np.ones(self.seq_len, dtype=np.int32) # (len)
                # attention_mask[tlen:] = 0.0
                assert np.sum(attention_mask) == self.seq_len
                attention_mask_data = torch.from_numpy(attention_mask).long()

                state_data = torch.from_numpy(states[:self.seq_len]).float() # len,128
                
                # We replace the invalid images with the background image
                # and also randomly mask images by the background image
                # background_color = np.array([
                #     int(x*255) for x in self.image_processor.image_mean
                # ], dtype=np.uint8).reshape(1, 1, 3)

                # background_image = np.ones((
                #     self.image_processor.size["height"], 
                #     self.image_processor.size["width"], 3), dtype=np.uint8
                # ) * background_color
                
                image_metas = list(self.pairwise(image_metas))

                # print("states:",states.shape," image_metas:",image_metas[0][0].shape)

                # mask_probs = [self.cond_mask_prob] * self.num_cameras
                # if self.cam_ext_mask_prob >= 0.0:
                #     mask_probs[0] = self.cam_ext_mask_prob

                rearranged_images = []
                rearranged_images_hand = []
                for i in range(self.seq_len): # 2
                    for j in range(self.num_cameras): # 2
                        images, image_mask = image_metas[j] # (2, 224, 224, 3),  (2,)
                        image, valid = images[i], image_mask[i] # (224, 224, 3),  (1,)
                        if valid and (math.prod(image.shape) > 0):
                            image = Image.fromarray(image)
                            if j==0:
                                # Only apply image augmentation to 50% of the images
                                if  self.mode == 'train' and (random.random() > 0.5):
                                    aug_type = random.choice([
                                        "corrput_only", "color_only", "both"])
                                    
                                    if aug_type != "corrput_only":
                                        image = transforms.ColorJitter(
                                            brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                                    if aug_type != "color_only":
                                        image = image_corrupt(image)
                            
                            image = T.ToTensor()(image.convert("RGB")) 
                            if j==0: rearranged_images.append(image)
                            elif j==1: rearranged_images_hand.append(image)

                        
                            # if i==self.seq_len-1 and j==0:
                
                image_goal = Image.fromarray(image_goal)
                image_goal = T.ToTensor()(image_goal.convert("RGB")) 
                rearranged_images.append(image_goal) # goal
                
                # print("rearranged_images:",len(rearranged_images)) # 2
                if len(rearranged_images)!=(self.seq_len+1):
                    index = (index + 1) % len(self)
                    continue
                
                rearranged_images = torch.stack(rearranged_images, dim=0) # 2+1=3, 3, 224, 224
                # print("rearranged_images",torch.min(rearranged_images),torch.max(rearranged_images)) tensor(0.) tensor(1.)
                
                if self.num_cameras==1:
                    rearranged_images_hand = torch.ones((self.seq_len,3,self.input_size[0],self.input_size[1]))
                    for channel_ind in range(3):
                        rearranged_images_hand[:,channel_ind] *= self.clip_mean[channel_ind]
                elif  self.num_cameras==2:
                    rearranged_images_hand = torch.stack(rearranged_images_hand, dim=0) # 2, 3, 224, 224
                
                if self.mode == 'train':
                    rearranged_images = self.static_rgb_preprocess_train(rearranged_images)
                    # print("rearranged_images preprocess:",torch.min(rearranged_images),torch.max(rearranged_images)) tensor(-2.1179) tensor(2.6400)
                    if self.num_cameras==2: rearranged_images_hand = self.static_rgb_preprocess_train(rearranged_images_hand)
                else:
                    rearranged_images = self.static_rgb_preprocess_val(rearranged_images)
                    if self.num_cameras==2: rearranged_images_hand = self.static_rgb_preprocess_val(rearranged_images_hand)

                goal_rgb_data = rearranged_images[-1].float() # (C, H, W)
                rgb_data = rearranged_images[:-1].float() # (len, C, H, W)
                hand_rgb_data = rearranged_images_hand.float() # (len, C, H, W)

                # preprocessed_images = []
                # # processor = self.image_processor
                # for image, valid in rearranged_images:
                #     image = Image.fromarray(image)
                #     image = T.ToTensor()(image.convert("RGB"))

                    # if self.image_size is not None:
                    #     image = transforms.Resize(self.image_size)(image) # (1008, 336)
                    # assert image.height == 336, "We haven't prepare for training with images of different resolutions."
                    
                    # if valid and self.auto_adjust_image_brightness: # False
                    #     pixel_values = list(image.getdata())
                    #     average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                    #     if average_brightness <= 0.15:
                    #         image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
                    # Only apply image augmentation to 50% of the images
                    # if valid and self.image_aug and (random.random() > 0.5):
                    #     aug_type = random.choice([
                    #         "corrput_only", "color_only", "both"])
                    #     if aug_type != "corrput_only":
                    #         image = transforms.ColorJitter(
                    #             brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                    #     if aug_type != "color_only":
                    #         image = image_corrupt(image)
                    
                    # if self.image_aspect_ratio == 'pad':
                    #     def expand2square(pil_img, background_color):
                    #         width, height = pil_img.size
                    #         if width == height:
                    #             return pil_img
                    #         elif width > height:
                    #             result = Image.new(pil_img.mode, (width, width), background_color)
                    #             result.paste(pil_img, (0, (width - height) // 2))
                    #             return result
                    #         else:
                    #             result = Image.new(pil_img.mode, (height, height), background_color)
                    #             result.paste(pil_img, ((height - width) // 2, 0))
                    #             return result
                            
                    #     image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))

                    # image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                    # preprocessed_images.append(image)

                # data_dict["images"] = preprocessed_images
                
                #progress
                progress_data=torch.zeros(self.seq_len).float()
                # for i in range(self.seq_len):
                #     progress=(st+i-traj_st)/(traj_ed-traj_st)
                #     progress_data[i]=progress
                
                self.text = content["instruction"]
                # print(self.text)
                
                self.current_progress = content['step_id']/content["#steps"]
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
                
                data_dict = {}
                data_dict['dataset_name'] = self.dataset_name # 'asu_table_top_converted_externally_to_rlds'
                data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']] # 0
                data_dict['ctrl_freq'] = torch.tensor([self.control_freq[data_dict['dataset_name']]]).float() # 12.5
                

                data_dict['goal_rgb'] = goal_rgb_data # (C, H, W)
                data_dict['rgb'] = rgb_data # (len, C, H, W)
                data_dict['hand_rgb'] = hand_rgb_data # (len, C, H, W)
                data_dict["text"] = [self.text] # 'Pick green bottle.'

                data_dict["action"] = action_data.float() # len,act_len,128
                data_dict["state_elem_mask"] = torch.from_numpy(state_elem_mask).long() #(128,)
                
                data_dict["state"] = state_data # (64,128)-->1,128
                data_dict['rel_state'] = state_data # (len, state_dim)
                data_dict['attention_mask'] = attention_mask_data # (len,)
                data_dict['action_mask'] = action_mask_data # (len, act_len, action_dim)
                # Stat for the episode that the step belongs to 
                data_dict["state_norm"] = state_norm # (128,)
                
                data_dict["progress"]=progress_data #(len)
                data_dict["RTG"]=RTG_data #torch.zeros((self.seq_len,1)).float()  #(len,reward_dim)
                data_dict["returns"]=returns_data #torch.zeros((self.seq_len,self.act_len,1)).float() #(len, act_len, 1)
                
                data_dict["time_get_data"]=time_get_data_end-time_get_data_start


                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                # print("goal_rgb_data",goal_rgb_data.shape)
                # print("rgb_data",rgb_data.shape)
                # print("hand_rgb_data",hand_rgb_data.shape)
                # print("action_data:",action_data.shape)
                # print("state_data:",state_data.shape) 
                # print("attention_mask_data:",attention_mask_data.shape)
                # print("state_elem_mask:",state_elem_mask.shape)
                # print("action_mask_data:",action_mask_data.shape) # action_mask_data: torch.Size([10, 5])
                # print("state_norm:",state_norm.shape)
                # print("RTG_data:",RTG_data.shape)
                # print("returns_data:",returns_data.shape)
                # print("data index:",index)

                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"
                        # data_dict[k] = torch.from_numpy(v)

                return data_dict
            
            except BaseException as e:
                # Print the error info
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}:", e)
                else:
                    print(f"Error catched when processing sample:", e)
                traceback.print_exc()
                # Try incresing the index
                index = (index + 1) % len(self)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            
    def get_reward(self,ed,episode_keypoints,reward, labeled_reward=0):
        # traj_idx = str(traj_idx)
       
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

        # reward=np.array(data_reward[str(traj_idx)]["reward"])
        # reward=np.concatenate((reward[:,0:3],reward[:,6:16]),axis=-1)
        episode_length = np.shape(reward)[0]

        # print("reward",reward) 
        # TODO
        # if self.dataset_name=="custom_robot_ur5":

        #     reward_error_joint = np.mean(reward[:,6:8],axis=-1,keepdims=True) # 1
        #     reward_error_joint_vel = np.mean(
        #                             np.concatenate((reward[:,8:9],
        #                                            reward[:,10:11]),
        #                                            axis=-1,keepdims=True), # 1
        #                                            axis=-1,keepdims=True) # 1
        #     reward_error_joint_acc = np.mean(
        #                             np.concatenate((reward[:,9:10],
        #                                            reward[:,11:12]),
        #                                            axis=-1,keepdims=True), # 1
        #                                            axis=-1,keepdims=True) # 1
        #     reward = np.concatenate((reward[:,:6],
        #                              reward_error_joint,
        #                              reward_error_joint_vel,
        #                              reward_error_joint_acc,
        #                              reward[:,12:]),axis=-1)

        reward=reward[:,:-1]
        
        # reward[:,9:15] = 0.5*reward[:,9:15] # T, D

        if self.reward_dim==1:
            reward=np.sum(reward,axis=-1) # T
        else:
            reward_total=np.sum(reward,axis=-1,keepdims=True) # T
            reward = np.concatenate((reward, reward_total),axis=-1) # 17

        reward_min,reward_max = np.min(reward,axis=0),np.max(reward,axis=0)
        
        reward= (reward[ed:] - reward_min)/( reward_max - reward_min + 10**(-6)) # t:T

        reward = 0.1*(reward+labeled_reward)

        # discounting_factor=0.8
        if self.reward_dim==1:
            returns = np.sum(reward * self.discounting_factor ** np.arange(len(reward)),keepdims=True)
        else:
            returns = np.sum(reward[:,-1] * self.discounting_factor ** np.arange(len(reward)),keepdims=True)

        RTG = np.sum(reward, axis=0,keepdims=True) # t+t_1+...
        
        returns = returns #/episode_length
        RTG = torch.from_numpy(RTG) #/episode_length

        return returns, RTG