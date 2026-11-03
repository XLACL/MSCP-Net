"""MSCP-Net: A Multi-Source Calibrated Prior Network for Semi-Supervised Medical Image Segmentation.

核心机制:
  1) 双分支架构 (Dual-branch Architecture)
  2) PASR (Pixel-Aware Scale Reconciliation): 像素级尺度调和
  3) CFCA (Clinical Frequency Calibration Augmentation): 临床频域校准增强
  4) GABC (Geometry-Anchored Boundary Consistency): 几何锚定边界一致性
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. 双分支架构 (Dual-branch Architecture)
# ----------------------------------------------------------------------
class DualBranchDecoder(nn.Module):
    def __init__(self, feature_channels, num_classes):
        super().__init__()
        self.low_decoder = nn.Sequential(
            nn.Conv2d(feature_channels[3], feature_channels[2], 3, padding=1),
            nn.BatchNorm2d(feature_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels[2], num_classes, 1)
        )
        self.high_decoder = nn.Sequential(
            nn.Conv2d(feature_channels[4], feature_channels[3], 3, padding=1),
            nn.BatchNorm2d(feature_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels[3], num_classes, 1)
        )

    def forward(self, features):
        low_out = self.low_decoder(features[3])
        high_out = self.high_decoder(features[4])
        return low_out, high_out


# ----------------------------------------------------------------------
# 2. PASR: 像素级尺度调和 (Pixel-Aware Scale Reconciliation)
# ----------------------------------------------------------------------
class PASRModule(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2) / 2.0)

    def forward(self, pred_low, pred_high, confidence_low, confidence_high):
        """
        Args:
            pred_low:        [B, C, H, W]
            pred_high:       [B, C, H, W]
            confidence_low:  [B, H, W] 或 [B, 1, H, W]
            confidence_high: [B, H, W] 或 [B, 1, H, W]
        Returns:
            ensemble_pred: [B, C, H, W]
            w_low, w_high: [B, 1, H, W]
        """
        # 把 confidence 补到 [B, 1, H, W] 以便与 pred 广播
        if confidence_low.dim() == 3:
            confidence_low = confidence_low.unsqueeze(1)
        if confidence_high.dim() == 3:
            confidence_high = confidence_high.unsqueeze(1)

        w_low = self.weights[0] * confidence_low     # [B, 1, H, W]
        w_high = self.weights[1] * confidence_high   # [B, 1, H, W]

        total = w_low + w_high + 1e-8
        w_low = w_low / total
        w_high = w_high / total

        ensemble_pred = w_low * pred_low + w_high * pred_high
        return ensemble_pred, w_low, w_high


# ----------------------------------------------------------------------
# 3. CFCA: 临床频域校准增强 (Clinical Frequency Calibration Augmentation)
# ----------------------------------------------------------------------
def cfca_augment(image, alpha=0.1):
    B, C, H, W = image.shape
    device = image.device

    # FFT
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = torch.abs(fft_shifted)
    phase = torch.angle(fft_shifted)

    # ---- 构造低频 mask (只扰动中心低频区) ----
    y = torch.arange(H, device=device).float() - H // 2
    x = torch.arange(W, device=device).float() - W // 2
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(yy ** 2 + xx ** 2)
    low_mask = (dist < min(H, W) * 0.25).float()  # 低频区域

    # ---- 按相对幅度缩放噪声 ----
    rel_scale = magnitude / (magnitude.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    noise = alpha * torch.randn_like(magnitude) * rel_scale * low_mask
    
    # 防止幅度谱出现负值 (论文 Eq.14 的物理约束)
    magnitude_perturbed = magnitude * torch.clamp(1.0 + noise, min=0.0)

    # ---- 重构 ----
    fft_perturbed = magnitude_perturbed * torch.exp(1j * phase)
    fft_perturbed = torch.fft.ifftshift(fft_perturbed, dim=(-2, -1))
    enhanced = torch.fft.ifft2(fft_perturbed, dim=(-2, -1)).real

    # ---- 重归一化到原图范围 ----
    img_min = image.amin(dim=(-2, -1), keepdim=True)
    img_max = image.amax(dim=(-2, -1), keepdim=True)
    enh_min = enhanced.amin(dim=(-2, -1), keepdim=True)
    enh_max = enhanced.amax(dim=(-2, -1), keepdim=True)
    enhanced = (enhanced - enh_min) / (enh_max - enh_min + 1e-8) * (img_max - img_min) + img_min

    return enhanced


# ----------------------------------------------------------------------
# 4. GABC: 几何锚定边界一致性 (Geometry-Anchored Boundary Consistency)
# ----------------------------------------------------------------------
def compute_sdm(mask, num_classes, normalize=True):
    """计算符号距离图 (Signed Distance Map).

    Args:
        mask: [B, H, W] 分割掩码 (long)
        num_classes: 类别数
        normalize: 是否归一化到 [-1, 1]
    Returns:
        sdm: [B, C, H, W]
    """
    from scipy.ndimage import distance_transform_edt
    B, H, W = mask.shape
    device = mask.device
    sdm = torch.zeros(B, num_classes, H, W, device=device)

    mask_np = mask.detach().cpu().numpy()
    for b in range(B):
        for c in range(num_classes):
            binary = (mask_np[b] == c)
            if binary.sum() == 0:
                continue
            dist_out = distance_transform_edt(~binary)
            dist_in = distance_transform_edt(binary)
            signed_dist = dist_out - dist_in
            if normalize:
                max_dist = np.abs(signed_dist).max() + 1e-8
                signed_dist = signed_dist / max_dist
            sdm[b, c] = torch.from_numpy(signed_dist).to(device)

    return sdm


def gabc_loss(sdm_pred, sdm_target):
    return F.l1_loss(sdm_pred, sdm_target.detach())
