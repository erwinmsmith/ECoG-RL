import torch
import torch.nn as nn
import torch.optim as optim
from model import PretrainedModel, FineTuningModel
from dataset import ECoGSingleDataset
from utils import train_one_epoch, evaluate, save_model, load_model, EarlyStopping, adjust_learning_rate, NegativePCC_loss, MutualInformationLoss
import argparse
import scipy.io
import ast
import os
import math

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs('model', exist_ok=True)
    
    if args.mode == 'pre':
        model = PretrainedModel(
            input_channels=args.input_channels,
            tcn_channels=args.tcn_channels,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            d_model=args.d_model,
            use_transformer=args.use_transformer,
            share_weights=args.share_weights
        ).to(device)
        
        dataset = ECoGSingleDataset(
            file_path=args.file_path,
            data_key=args.data_key,
            label_key=None,
            normalize=True,
            shuffle=False,
            window_size=args.window_size,
            stride=args.stride
        )
        
        # 按时间顺序划分训练集和验证集（验证集占20%）
        train_size = int(len(dataset) * 0.8)
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        mi_loss = MutualInformationLoss(n_bins=10)  # 互信息损失
        
        # 学习率调度和早停
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=20, delta=0.001)
        
        for epoch in range(args.epochs):
            # 训练阶段
            train_time_loss, train_freq_loss, train_mi_loss, train_reg_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, mi_loss, mode='pre')
            
            # 验证阶段
            val_time_loss, val_freq_loss, val_mi_loss, val_reg_loss = evaluate(model, val_dataloader, criterion, mi_loss, device, mode='pre')
            
            # 计算总损失
            train_total_loss = train_time_loss + train_freq_loss + train_mi_loss + train_reg_loss
            val_total_loss = val_time_loss + val_freq_loss + val_mi_loss + val_reg_loss
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}, LR: {current_lr:.6f}, '
                  f'Train Total Loss: {train_total_loss:.4f}, '
                  f'Train Time Loss: {train_time_loss:.4f}, '
                  f'Train Freq Loss: {train_freq_loss:.4f}, '
                  f'Train MI Loss: {train_mi_loss:.4f}, '
                  f'Train Reg Loss: {train_reg_loss:.4f}, '
                  f'Val Total Loss: {val_total_loss:.4f}, '
                  f'Val Time Loss: {val_time_loss:.4f}, '
                  f'Val Freq Loss: {val_freq_loss:.4f}, '
                  f'Val MI Loss: {val_mi_loss:.4f}, '
                  f'Val Reg Loss: {val_reg_loss:.4f}')
            
            # 更新学习率和早停检查
            scheduler.step(val_total_loss)
            early_stopping(val_total_loss, model, 'model/best_pretrained_model.pth')
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break  
    if args.mode == 'fin':
        pretrained_model = PretrainedModel(
            input_channels=args.input_channels,
            tcn_channels=args.tcn_channels,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            d_model=args.d_model,
            use_transformer=args.use_transformer,
            share_weights=args.share_weights
        )
        load_model(pretrained_model, args.pretrained_model_path)
        
        dataset = ECoGSingleDataset(
            file_path=args.file_path,
            data_key=args.data_key,
            label_key=args.label_key,
            normalize=True,
            shuffle=False,
            window_size=args.window_size,
            stride=args.stride
        )
        
        train_size = int(len(dataset) * 0.8)
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = FineTuningModel(
            pretrained_model=pretrained_model,
            num_regression_targets=args.num_regression_targets,
            input_channels=args.input_channels,
            window_size=args.window_size,
            share_weights=args.share_weights
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=20, delta=0.001)
        
        for epoch in range(args.epochs):
            train_loss = 0.0
            val_loss = 0.0
            
            model.train()
            for batch in train_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 确保目标标签是 float32
                targets = targets.float()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 确保目标标签是 float32
                    targets = targets.float()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch + 1}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            scheduler.step(val_loss)
            early_stopping(val_loss, model, 'model/best_finetuned_model.pth')
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECoG Model Training')
    parser.add_argument('--mode', type=str, required=True, choices=['pre', 'fin'], help='pre for pretraining, fin for fine-tuning')
    parser.add_argument('--file_path', type=str, required=True, help='Path to a single ECoG data file')
    parser.add_argument('--data_key', type=str, default='train_data', help='Key for data in MATLAB file')
    parser.add_argument('--label_key', type=str, default='train_dg', help='Key for labels in MATLAB file')
    parser.add_argument('--input_channels', type=int, default=62, help='Number of input channels')
    parser.add_argument('--tcn_channels', type=ast.literal_eval, default=[62, 124, 62], help='TCN channel configuration')
    parser.add_argument('--transformer_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--transformer_heads', type=int, default=2, help='Number of transformer heads')
    parser.add_argument('--d_model', type=int, default=62, help='Transformer embedding dimension')
    parser.add_argument('--use_transformer', type=int, default=1, help='Whether to use transformer (1 for yes, 0 for no)')
    parser.add_argument('--share_weights', type=int, default=1, help='Whether to share weights between encoder and decoder (1 for yes, 0 for no)')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to pretrained model (required for fine-tuning)')
    parser.add_argument('--num_regression_targets', type=int, help='Number of regression targets (required for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--window_size', type=int, default=1000, help='Window size for time series data')
    parser.add_argument('--stride', type=int, default=500, help='Stride for time series data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weighted_pretraining', type=int, default=1, help='Whether to use weighted pretraining (1 for yes, 0 for no)')
    
    args = parser.parse_args()
    main(args)