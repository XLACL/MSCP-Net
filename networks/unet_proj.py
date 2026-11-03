from torch import nn
from networks.unet import Encoder, Decoder


class UNetProj(nn.Module):
    """UNet with projection head + dual-branch + SDM regression (for SASNet)."""

    def __init__(self, in_channels, num_classes):
        super(UNetProj, self).__init__()

        params = {
            'in_chns': in_channels,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': num_classes,
            'acti_func': 'relu'
        }

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)  # 保留原始解码器用于兼容

        # 双分支输出头 (SASNet)
        self.low_head = nn.Conv2d(128, num_classes, 1)
        self.high_head = nn.Conv2d(256, num_classes, 1)
        
        # SDM 回归头
        self.sdm_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
            nn.Tanh()
        )

        dim = params['feature_chns'][0]
        self.proj_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim, affine=False)
        )

    def forward(self, x):
        feature = self.encoder(x)
        output, feat = self.decoder(feature)

        # 双分支输出
        low_out = self.low_head(feature[3])   # 第4层特征
        high_out = self.high_head(feature[4]) # 第5层特征

        # SDM 回归
        sdm_out = self.sdm_head(feat)

        # 返回字典以兼容旧代码 (旧代码用 _ 忽略第二返回值)
        return output, {
            'feat': feat,
            'low_out': low_out,
            'high_out': high_out,
            'sdm': sdm_out,
        }
