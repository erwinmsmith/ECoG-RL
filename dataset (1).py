import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io

class ECoGSingleDataset(Dataset):
    def __init__(self, file_path, data_key, label_key=None, transform=None, normalize=False, shuffle=False, window_size=1000, stride=500):
        """
        初始化单个 ECoG 数据样本
        :param file_path: 单个 ECoG 数据文件的路径
        :param data_key: 数据的键名
        :param label_key: 标签的键名，如果为 None，则表示无监督预训练
        :param transform: 可选的数据变换
        :param normalize: 是否对数据进行归一化
        :param shuffle: 是否在训练时打乱数据
        :param window_size: 时间滑窗的大小
        :param stride: 滑窗的步长
        """
        self.file_path = file_path
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.normalize = normalize
        self.shuffle = shuffle
        self.window_size = window_size
        self.stride = stride
        
        # 加载数据和标签
        mat_file = scipy.io.loadmat(self.file_path)
        self.data = mat_file[data_key].astype(np.float32)
        
        # 确保数据形状为 (channels, timesteps)
        if len(self.data.shape) == 2:
            self.data = self.data.transpose((1, 0))
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}. Expected 2D array.")
        
        # 归一化处理
        if self.normalize:
            self.mean = np.mean(self.data, axis=1, keepdims=True)
            self.std = np.std(self.data, axis=1, keepdims=True)
            self.data = (self.data - self.mean) / (self.std + 1e-10)  # 避免除以零
        
        self.labels = None
        if label_key is not None:
            self.labels = mat_file[label_key].astype(np.float32)
            # 确保标签形状为 (timesteps, num_classes)
            if len(self.labels.shape) != 2:
                raise ValueError(f"Unexpected label shape: {self.labels.shape}. Expected 2D array.")
            
            # 打乱数据
            if self.shuffle:
                # 确保数据和标签的数量一致
                assert self.data.shape[1] == len(self.labels), "Data and labels must have the same number of samples"
                
                # 生成随机索引
                indices = np.arange(self.data.shape[1])
                np.random.shuffle(indices)
                
                # 重新排列数据和标签
                self.data = self.data[:, indices]
                self.labels = self.labels[indices]
        else:
            # 如果没有标签，仅打乱数据
            if self.shuffle:
                indices = np.arange(self.data.shape[1])
                np.random.shuffle(indices)
                self.data = self.data[:, indices]

    def __len__(self):
        # 计算滑窗后的样本数量
        return (self.data.shape[1] - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        # 计算滑窗的起始和结束索引
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # 获取滑窗内的数据和标签
        sample = self.data[:, start_idx:end_idx]
        
        # 将 NumPy 数组转换为 PyTorch 张量
        sample = torch.from_numpy(sample)
        
        # 确保 sample 的形状是 [channels, timesteps]
        if len(sample.shape) == 1:
            sample = sample.unsqueeze(0)  # 增加一个维度，使其成为 [1, channels]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
            # 获取对应时间窗口的最后一个时间步的标签
            label = torch.from_numpy(self.labels[end_idx-1].astype(np.float32))
            return sample, label
        else:
            return sample, sample  # 返回输入和目标（自编码器任务）