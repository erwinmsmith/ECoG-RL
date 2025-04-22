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
        
        
        mat_file = scipy.io.loadmat(self.file_path)
        self.data = mat_file[data_key].astype(np.float32)
        
        
        if len(self.data.shape) == 2:
            self.data = self.data.transpose((1, 0))
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}. Expected 2D array.")
        
      
        if self.normalize:
            self.mean = np.mean(self.data, axis=1, keepdims=True)
            self.std = np.std(self.data, axis=1, keepdims=True)
            self.data = (self.data - self.mean) / (self.std + 1e-10)  
        
        self.labels = None
        if label_key is not None:
            self.labels = mat_file[label_key].astype(np.float32)
            
            if len(self.labels.shape) != 2:
                raise ValueError(f"Unexpected label shape: {self.labels.shape}. Expected 2D array.")
            
            
            if self.shuffle:
               
                assert self.data.shape[1] == len(self.labels), "Data and labels must have the same number of samples"
                
                indices = np.arange(self.data.shape[1])
                np.random.shuffle(indices)
                
               
                self.data = self.data[:, indices]
                self.labels = self.labels[indices]
        else:
           
            if self.shuffle:
                indices = np.arange(self.data.shape[1])
                np.random.shuffle(indices)
                self.data = self.data[:, indices]

    def __len__(self):
        
        return (self.data.shape[1] - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        
        sample = self.data[:, start_idx:end_idx]
        
        
        sample = torch.from_numpy(sample)
        
       
        if len(sample.shape) == 1:
            sample = sample.unsqueeze(0)  
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is not None:
           
            label = torch.from_numpy(self.labels[end_idx-1].astype(np.float32))
            return sample, label
        else:
            return sample, sample  
