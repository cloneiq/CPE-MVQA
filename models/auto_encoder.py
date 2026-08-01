"""
auto_encoder.py  (stub)
=======================
Minimal placeholder for the CDAE (Convolutional Denoising Auto-Encoder) module
originally from MEVF (MICCAI'19).

The current CausalVQAModel does NOT instantiate or use this module in its
__init__ or forward method. This file exists only to satisfy the legacy import
    from . import auto_encoder
in vqa_module.py and prevent ImportError.

If you need the full CDAE implementation in the future, refer to:
    https://github.com/aioz-ai/MICCAI19-MedVQA/blob/master/auto_encoder.py
"""

import torch
import torch.nn as nn


class Auto_Encoder_Model(nn.Module):
    """Convolutional Denoising Auto-Encoder for 128x128 grayscale medical images.

    This is a minimal stub. The encoder produces a 768-d feature vector.
    """

    def __init__(self):
        super().__init__()
        # Encoder: 1-channel 128x128 -> feature vector
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # -> 64x64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 32x32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> 16x16
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# -> 8x8
            nn.ReLU(True),
        )
        # 256 * 8 * 8 = 16384 -> 768
        self.fc = nn.Linear(256 * 8 * 8, 768)

        # Decoder (unused during inference, included for weight-loading compat)
        self.decoder_fc = nn.Linear(768, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, 1, 128, 128) -> features: (B, 768)"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        features = self.fc(h)
        return features
