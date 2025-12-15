import torch
import torch.nn as nn

class SpectralFeatureExtractor(nn.Module):
    """
    Extracts features in both time and frequency domains
    """
    def __init__(self, input_channels=19, time_length=200, embedding_dim=768):
        super().__init__()

        self.time_length = time_length

        # Time domain features
        self.time_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        # Frequency domain processing
        self.freq_projection = nn.Sequential(
            nn.Linear(time_length // 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 * 32 + 64 * input_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x):
        # x shape: (batch, 19, 1, 200)
        x = x.squeeze(2)  # (batch, 19, 200)

        # Time domain features
        time_features = self.time_conv(x)
        time_features_flat = time_features.flatten(1)

        # Frequency domain features (using FFT)
        freq_features = []
        for i in range(x.size(1)):  # For each channel
            channel_data = x[:, i, :]
            fft_result = torch.fft.rfft(channel_data, dim=-1)
            magnitude = torch.abs(fft_result)
            freq_feat = self.freq_projection(magnitude)
            freq_features.append(freq_feat)

        freq_features = torch.stack(freq_features, dim=1)  # (batch, channels, 64)
        freq_features_flat = freq_features.flatten(1)
        # print(time_features_flat.shape)
        # print(freq_features_flat.shape)
        # Concatenate and fuse
        combined = torch.cat([time_features_flat, freq_features_flat], dim=1)
        embedding = self.fusion(combined)

        return embedding
