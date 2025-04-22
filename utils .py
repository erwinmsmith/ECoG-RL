import torch
import numpy as np
import math
import torch.nn as nn
from sklearn.metrics import mutual_info_score

class MutualInformationLoss(nn.Module):
    def __init__(self, n_bins=10):
        super(MutualInformationLoss, self).__init__()
        self.n_bins = n_bins

    def forward(self, x, y):
  
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        x_binned = np.digitize(x, np.linspace(x.min(), x.max(), self.n_bins))
        y_binned = np.digitize(y, np.linspace(y.min(), y.max(), self.n_bins))
        
        mi = 0.0
        for i in range(x.shape[0]):
            mi += mutual_info_score(x_binned[i], y_binned[i])
        
        return 1/(torch.tensor(mi / x.shape[0]))
        
def save_model(model, path):
    """
    保存模型
    :param model: 要保存的模型
    :param path: 保存路径
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    加载模型
    :param model: 要加载的模型
    :param path: 模型路径
    :return: 加载后的模型
    """
    model.load_state_dict(torch.load(path))
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device, mi_loss, mode='pre'):
    model.train()
    total_time_loss = 0
    total_freq_loss = 0
    total_mi_loss = 0
    total_regularization_loss = 0
    
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        if mode == 'pre':
   
            time_loss = criterion(outputs['time_reconstructed'], inputs)
            freq_loss = criterion(outputs['freq_reconstructed'], outputs['original_freq'])  # 频域重建损失
            
            mi = mi_loss(outputs['time_features'], outputs['freq_features'])

            time_reg = torch.mean(torch.abs(outputs['time_reconstructed']))
            freq_reg = torch.mean(torch.abs(outputs['freq_reconstructed']))
            regularization_loss = time_reg + freq_reg
            
            # 总损失 = 重建损失 + 互信息损失 + 正则化损失
            loss = time_loss + freq_loss + mi + regularization_loss
            
            total_time_loss += time_loss.item()
            total_freq_loss += freq_loss.item()
            total_mi_loss += mi.item()
            total_regularization_loss += regularization_loss.item()
            
            loss.backward()
            optimizer.step()
        else:
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            total_time_loss += loss.item()

    if mode == 'pre':
        return total_time_loss / len(dataloader), total_freq_loss / len(dataloader), total_mi_loss / len(dataloader), total_regularization_loss / len(dataloader)
    else:
        return total_time_loss / len(dataloader)

def evaluate(model, dataloader, criterion, mi_loss, device, mode='pre'):
    model.eval()
    total_time_loss = 0
    total_freq_loss = 0
    total_mi_loss = 0
    total_regularization_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            if mode == 'pre':
                time_loss = criterion(outputs['time_reconstructed'], inputs)
                freq_loss = criterion(outputs['freq_reconstructed'], outputs['original_freq'])
                mi = mi_loss(outputs['time_features'], outputs['freq_features'])
                
                # 计算正则化损失
                time_reg = torch.mean(torch.abs(outputs['time_reconstructed']))
                freq_reg = torch.mean(torch.abs(outputs['freq_reconstructed']))
                regularization_loss = time_reg + freq_reg
                
                total_time_loss += time_loss.item()
                total_freq_loss += freq_loss.item()
                total_mi_loss += mi.item()
                total_regularization_loss += regularization_loss.item()
            else:
                targets = targets.squeeze()
                loss = criterion(outputs, targets)
                total_time_loss += loss.item()

    if mode == 'pre':
        return total_time_loss / len(dataloader), total_freq_loss / len(dataloader), total_mi_loss / len(dataloader), total_regularization_loss / len(dataloader)
    else:
        return total_time_loss / len(dataloader)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_metrics(epoch, metrics, mode='train'):
    print(f'Epoch {epoch} {mode}:', end=' ')
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}', end=' ')
    print()

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def adjust_learning_rate(optimizer, epoch, initial_lr, total_epochs, warmup_epochs=0):
    """
    调整学习率，使用余弦退火策略
    :param optimizer: 优化器
    :param epoch: 当前训练的轮数
    :param initial_lr: 初始学习率
    :param total_epochs: 总训练轮数
    :param warmup_epochs: 热身训练的轮数，默认为0
    """
    if epoch < warmup_epochs:
        # 线性热身
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # 余弦退火
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=20, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.has_saved = False  

    def __call__(self, val_loss, model, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if not self.has_saved:
                    save_model(model, save_path)
                    self.has_saved = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.has_saved = False

class NegativePCC_loss(nn.Module):
    def __init__(self):
        super(NegativePCC_loss, self).__init__()

    def forward(self, inputs, targets):
        targets = targets.float()
        input_mean = torch.mean(inputs, dim=1, keepdim=True)
        target_mean = torch.mean(targets, dim=1, keepdim=True)
        
        cov = torch.mean((inputs - input_mean) * (targets - target_mean), dim=1)
        
        input_std = torch.std(inputs, dim=1)
        target_std = torch.std(targets, dim=1)
        
        pcc = cov / (input_std * target_std + 1e-8)
        
        return 1-pcc
