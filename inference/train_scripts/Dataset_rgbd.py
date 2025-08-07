import numpy as np
import torch
import os
import cv2
from torch.utils.data import Dataset
from numpy.random import RandomState
import random


class SimpleDataset(Dataset):
    def __init__(self, args, scale=1):
        self.file_dir = args.turbid_data
        self.file_paths = sorted(os.listdir(self.file_dir))
        self.clear_dir = args.clear_data + 'images'
        self.clear_list = sorted(os.listdir(self.clear_dir))
        self.dep_dir = args.clear_data + 'depths'
        self.dep_list = sorted(os.listdir(self.dep_dir))
        self.rand_state = RandomState(66)
        self.ratio = scale

    def __getitem__(self, item):
        idx0 = random.randint(200, len(self.file_paths) - 1)
        img_name = self.file_paths[idx0]
        img_path = os.path.join(self.file_dir, img_name)
        input_image = cv2.imread(img_path)
        height, width = input_image.shape[:2]
        input_image = input_image.astype(np.float32) / 255.0
        input_image = cv2.resize(input_image, (int(width//self.ratio), int(height//self.ratio)), interpolation=cv2.INTER_LINEAR)  # Use this if image scaling (resizing) is needed
        input_image = cv2.resize(input_image, (640, 480), interpolation=cv2.INTER_LINEAR)
        height, width = input_image.shape[:2]
        input_image = np.transpose(input_image, (2, 0, 1))

        idx = random.randint(0, len(self.clear_list) - 1)
        clear_name = self.clear_list[idx]
        clear_path = os.path.join(self.clear_dir, clear_name)
        clear_image = cv2.imread(clear_path)
        clear_image = clear_image.astype(np.float32) / 255.0
        edge = 8
        clear_image = clear_image[edge:-edge, edge:-edge]
        clear_image = cv2.resize(clear_image, (int(width//self.ratio), int(height//self.ratio)), interpolation=cv2.INTER_LINEAR)
        clear_image = cv2.resize(clear_image, (640, 480), interpolation=cv2.INTER_LINEAR)
        clear_image = np.transpose(clear_image, (2, 0, 1))

        dep_name = self.dep_list[idx]
        dep_path = os.path.join(self.dep_dir, dep_name)
        dep_image = cv2.imread(dep_path, -1)
        dep_image = dep_image[edge:-edge, edge:-edge]
        dep_image = dep_image.astype(np.float32)/1000
        dep_image = cv2.resize(dep_image, (int(width//self.ratio), int(height//self.ratio)), interpolation=cv2.INTER_LINEAR)
        dep_image = cv2.resize(dep_image, (640, 480), interpolation=cv2.INTER_LINEAR)
        dep_image = 0.5 + (dep_image - np.min(dep_image)) / (np.max(dep_image) - np.min(dep_image)) * 4.5

        net_img = input_image
        net_img = torch.from_numpy(net_img)

        net_img1 = clear_image
        net_img1 = torch.from_numpy(net_img1)

        dep_img = dep_image
        dep_img = torch.from_numpy(dep_img)
        dep_img = dep_img.unsqueeze(dim=0)

        return net_img, net_img1, dep_img

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def collate_fn(batch):
        input_img, clear_image, dep_img = tuple(zip(*batch))
        input_img = torch.stack(input_img, dim=0)
        clear_image = torch.stack(clear_image, dim=0)
        dep_img = torch.stack(dep_img, dim=0)
        return input_img, clear_image, dep_img
