import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PretrainedModel(nn.Module):
    def __init__(self, input_channels, tcn_channels, transformer_layers, transformer_heads, d_model, use_transformer=True, share_weights=False):
        super(PretrainedModel, self).__init__()
        self.input_channels = input_channels
        self.use_transformer = use_transformer
        self.share_weights = share_weights
        
        # TCN Hyperparameters
        self.tcn_channels = tcn_channels
        self.kernel_size = 2
        self.tcn_dropout = 0.2
        
        # Transformer Hyperparameters
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.d_model = d_model
        
        # Encoder - Time Domain
        self.time_encoder_tcn1 = TemporalConvNet(input_channels, tcn_channels, kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        
        if use_transformer:
            self.time_encoder_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, transformer_heads, dropout=self.tcn_dropout, batch_first=True),
                transformer_layers
            )
        else:
            self.time_encoder_linear = nn.Linear(tcn_channels[-1], d_model)
        
        # 使用 tcn_channels[::-1] 来确保时间域解码器的通道数与编码器匹配
        decoder_tcn_channels = tcn_channels[::-1]
        self.time_encoder_tcn2 = TemporalConvNet(tcn_channels[-1], decoder_tcn_channels, kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        
        # Decoder - Time Domain
        if share_weights:
            self.time_decoder_tcn1 = self.time_encoder_tcn2
            if use_transformer:
                self.time_decoder_transformer = self.time_encoder_transformer
            else:
                self.time_decoder_linear = self.time_encoder_linear
            # 确保时间域解码器的最后一个TCN层级输出通道数与输入通道数一致
            self.time_decoder_tcn2 = TemporalConvNet(decoder_tcn_channels[-1], [input_channels], kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        else:
            self.time_decoder_tcn1 = TemporalConvNet(tcn_channels[-1], decoder_tcn_channels[:-1], kernel_size=self.kernel_size, dropout=self.tcn_dropout)
            if use_transformer:
                self.time_decoder_transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, transformer_heads, dropout=self.tcn_dropout, batch_first=True),
                    transformer_layers
                )
            else:
                self.time_decoder_linear = nn.Linear(tcn_channels[-1], d_model)
            self.time_decoder_tcn2 = TemporalConvNet(decoder_tcn_channels[-2], [input_channels], kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        
        # Encoder - Frequency Domain
        self.freq_encoder_tcn1 = TemporalConvNet(2 * input_channels, tcn_channels, kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        
        if use_transformer:
            self.freq_encoder_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, transformer_heads, dropout=self.tcn_dropout, batch_first=True),
                transformer_layers
            )
        else:
            self.freq_encoder_linear = nn.Linear(tcn_channels[-1], d_model)
        
        self.freq_encoder_tcn2 = TemporalConvNet(tcn_channels[-1], decoder_tcn_channels, kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        
        # Decoder - Frequency Domain
        if share_weights:
            self.freq_decoder_tcn1 = self.freq_encoder_tcn2
            if use_transformer:
                self.freq_decoder_transformer = self.freq_encoder_transformer
            else:
                self.freq_decoder_linear = self.freq_encoder_linear
            # 确保频域解码器的最后一个TCN层级输出通道数与频域输入通道数一致
            self.freq_decoder_tcn2 = TemporalConvNet(decoder_tcn_channels[-1], [2 * input_channels], kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        else:
            self.freq_decoder_tcn1 = TemporalConvNet(tcn_channels[-1], decoder_tcn_channels[:-1], kernel_size=self.kernel_size, dropout=self.tcn_dropout)
            if use_transformer:
                self.freq_decoder_transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, transformer_heads, dropout=self.tcn_dropout, batch_first=True),
                    transformer_layers
                )
            else:
                self.freq_decoder_linear = nn.Linear(tcn_channels[-1], d_model)
            self.freq_decoder_tcn2 = TemporalConvNet(decoder_tcn_channels[-2], [2 * input_channels], kernel_size=self.kernel_size, dropout=self.tcn_dropout)
        
        # Pooling for global features
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Frequency feature extraction layers
        self.freq_magnitude_fc = nn.Linear(input_channels, d_model)
        self.freq_phase_fc = nn.Linear(input_channels, d_model)
        
        # Final output layers
        self.time_output = nn.Conv1d(input_channels, input_channels, 1)
        self.freq_output = nn.Conv1d(2 * input_channels, 2 * input_channels, 1)  # 修改为输出频域特征的通道数
        
    def forward(self, x):
        # 确保输入的形状为 (batch_size, channels, timesteps)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_channels:
            x = x.permute(0, 2, 1)
        
        # 时间域正则化
        x_time_norm = (x - x.mean(dim=[1, 2], keepdim=True)) / (x.std(dim=[1, 2], keepdim=True) + 1e-10)
        
        # 转换到频域
        with torch.no_grad():
            fft = torch.fft.fft(x, dim=-1)
            fft_magnitude = torch.abs(fft)
            fft_phase = torch.angle(fft)
        
        # 提取频域特征
        magnitude_features = self.freq_magnitude_fc(fft_magnitude.permute(0, 2, 1)).permute(0, 2, 1)
        phase_features = self.freq_phase_fc(fft_phase.permute(0, 2, 1)).permute(0, 2, 1)
        x_freq = torch.cat([magnitude_features, phase_features], dim=1)
        
        # 频域正则化
        x_freq_norm = (x_freq - x_freq.mean(dim=[1, 2], keepdim=True)) / (x_freq.std(dim=[1, 2], keepdim=True) + 1e-10)
        
        # 时间域编码器
        time_enc1 = self.time_encoder_tcn1(x_time_norm)
        if self.use_transformer:
            time_enc2 = self.time_encoder_transformer(time_enc1.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            time_enc2 = self.time_encoder_linear(time_enc1.permute(0, 2, 1)).permute(0, 2, 1)
        
        time_enc3 = self.time_encoder_tcn2(time_enc2)
        time_features = self.avg_pool(time_enc3).squeeze(-1)  # 保存时间域特征
        
        # 频域编码器
        freq_enc1 = self.freq_encoder_tcn1(x_freq_norm)
        if self.use_transformer:
            freq_enc2 = self.freq_encoder_transformer(freq_enc1.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            freq_enc2 = self.freq_encoder_linear(freq_enc1.permute(0, 2, 1)).permute(0, 2, 1)
        
        freq_enc3 = self.freq_encoder_tcn2(freq_enc2)
        freq_features = self.avg_pool(freq_enc3).squeeze(-1)  # 保存频域特征
        
        # 时间域解码器
        if self.share_weights:
            time_dec1 = self.time_decoder_tcn1(time_enc3)
            if self.use_transformer:
                time_dec2 = self.time_decoder_transformer(time_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                time_dec2 = self.time_decoder_linear(time_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            time_dec3 = self.time_decoder_tcn2(time_dec2)
        else:
            time_dec1 = self.time_decoder_tcn1(time_enc3)
            if self.use_transformer:
                time_dec2 = self.time_decoder_transformer(time_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                time_dec2 = self.time_decoder_linear(time_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            time_dec3 = self.time_decoder_tcn2(time_dec2)
        
        time_reconstructed = self.time_output(time_dec3)
        
        # 频域解码器
        if self.share_weights:
            freq_dec1 = self.freq_decoder_tcn1(freq_enc3)
            if self.use_transformer:
                freq_dec2 = self.freq_decoder_transformer(freq_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                freq_dec2 = self.freq_decoder_linear(freq_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            freq_dec3 = self.freq_decoder_tcn2(freq_dec2)
        else:
            freq_dec1 = self.freq_decoder_tcn1(freq_enc3)
            if self.use_transformer:
                freq_dec2 = self.freq_decoder_transformer(freq_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                freq_dec2 = self.freq_decoder_linear(freq_dec1.permute(0, 2, 1)).permute(0, 2, 1)
            freq_dec3 = self.freq_decoder_tcn2(freq_dec2)
        
        # 返回频域重建信号
        freq_reconstructed = self.freq_output(freq_dec3)
        
        # 返回重建信号和特征
        return {
            'time_reconstructed': time_reconstructed,
            'freq_reconstructed': freq_reconstructed,
            'time_features': time_features,  # 返回时间域特征
            'freq_features': freq_features,   # 返回频域特征
            'original_freq': x_freq  # 返回原始频域信号
        }
        
class FineTuningModel(nn.Module):
    def __init__(self, pretrained_model, num_regression_targets, input_channels, window_size, share_weights=False):
        super(FineTuningModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.num_regression_targets = num_regression_targets
        self.input_channels = input_channels
        self.window_size = window_size
        self.share_weights = share_weights

        # 确保预训练模型中的所有层都使用 float32
        self.pretrained_model = self.pretrained_model.float()
        
        # GRU 层
        self.gru = nn.GRU(input_size=2 * self.pretrained_model.tcn_channels[-1], hidden_size=256, num_layers=200, batch_first=True)
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(256 + input_channels * window_size, 128),  
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.Linear(256, 64),                               
            #nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, num_regression_targets)
        )
    
    def forward(self, x):
        # 确保输入是 float32
        x = x.float()
        
        # 前向传播通过预训练模型
        outputs = self.pretrained_model(x)
        time_features = outputs['time_features']
        freq_features = outputs['freq_features']
        
        # 将时间域和频域特征拼接
        combined_features = torch.cat([time_features, freq_features], dim=1)
        
        # Reshape for GRU
        gru_input = combined_features.view(combined_features.size(0), -1, 2 * self.pretrained_model.tcn_channels[-1])
        
        # GRU
        gru_output, _ = self.gru(gru_input)
        gru_output = gru_output[:, -1, :]  # 取最后一个时间步的输出
        
        # 拼接原始归一化后的数据
        x_normalized = x.permute(0, 2, 1)  # 形状为 (batch_size, timesteps, channels)
        x_normalized = x_normalized.reshape(x_normalized.size(0), -1)  # 展平时间维度和通道维度
        
        combined_regressor_input = torch.cat([gru_output, x_normalized], dim=1)
        
        # 回归预测
        predictions = self.regressor(combined_regressor_input)
        
        return predictions
