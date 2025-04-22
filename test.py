import torch
from model import FineTuningModel, PretrainedModel  
from dataset import ECoGSingleDataset
import argparse
import scipy.io
from scipy.stats import pearsonr
import numpy as np
import ast

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    pretrained_model = PretrainedModel(
        input_channels=args.input_channels,
        tcn_channels=args.tcn_channels,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        d_model=args.d_model,
        use_transformer=args.use_transformer,
        share_weights=args.share_weights
    )
    

    model = FineTuningModel(
        pretrained_model=pretrained_model,
        num_regression_targets=args.num_regression_targets,
        input_channels=args.input_channels, 
        window_size=args.window_size,        
        share_weights=args.share_weights
    ).to(device)
    
  
    model.load_state_dict(torch.load(args.finetuned_model_path, map_location=device))  
  

    dataset = ECoGSingleDataset(
        file_path=args.file_path,
        data_key=args.data_key,
        normalize=True,
        shuffle=False,
        window_size=args.window_size,
        stride=args.stride
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    

    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch  
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = outputs.cpu().numpy()
            all_predictions.append(predictions)
    
  
    all_predictions = np.concatenate(all_predictions, axis=0)
    np.save(args.output_path, all_predictions)

    true_labels_mat = scipy.io.loadmat('test/sub3_testlabels.mat')
    true_labels = true_labels_mat['test_dg']
    

    windowed_true_labels = []
    for start_idx in range(0, true_labels.shape[0] - args.window_size + 1, args.stride):
        end_idx = start_idx + args.window_size
        windowed_label = true_labels[start_idx:end_idx]
        windowed_true_labels.append(windowed_label[-1])  
    windowed_true_labels = np.array(windowed_true_labels)
    

    pcc_values = []
    ignore_dimension = 5  
    

    for i in range(windowed_true_labels.shape[1]):
        if i != ignore_dimension:
            corr, _ = pearsonr(windowed_true_labels[:, i], all_predictions[:, i])
            pcc_values.append(corr)
    

    average_pcc = np.mean(pcc_values)
    
   
    print(f"PCC values for each dimension (excluding dimension {ignore_dimension}): {pcc_values}")
    print(f"Average PCC (excluding dimension {ignore_dimension}): {average_pcc:.4f}")
    
    print(f"Predictions saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECoG Model Testing')
    parser.add_argument('--file_path', type=str, required=True, help='Path to a single ECoG data file')
    parser.add_argument('--data_key', type=str, default='test_data', help='Key for data in MATLAB file')
    parser.add_argument('--input_channels', type=int, default=62, help='Number of input channels')
    parser.add_argument('--tcn_channels', type=ast.literal_eval, default=[62, 124, 62], help='TCN channel configuration')
    parser.add_argument('--transformer_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--transformer_heads', type=int, default=2, help='Number of transformer heads')
    parser.add_argument('--d_model', type=int, default=62, help='Transformer embedding dimension')
    parser.add_argument('--use_transformer', type=int, default=1, help='Whether to use transformer (1 for yes, 0 for no)')
    parser.add_argument('--finetuned_model_path', type=str, required=True, help='Path to finetuned model')
    parser.add_argument('--num_regression_targets', type=int, required=True, help='Number of regression targets')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--output_path', type=str, default='predictions.npy', help='Path to save predictions')
    parser.add_argument('--window_size', type=int, default=1000, help='Window size for time series data')
    parser.add_argument('--stride', type=int, default=500, help='Stride for time series data')
    parser.add_argument('--share_weights', type=int, default=1, help='Whether to share weights between encoder and decoder (1 for yes, 0 for no)')
    
    args = parser.parse_args()
    main(args)
