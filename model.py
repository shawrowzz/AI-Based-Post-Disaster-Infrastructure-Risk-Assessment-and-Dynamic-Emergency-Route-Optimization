import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SiameseFPN(nn.Module):
    """
    Siamese Feature Pyramid Network (FPN) for Building Damage Assessment.
    Supports ResNet-50, ConvNeXt-Tiny, and Swin-Tiny.
    Standardizes lateral projection to d_fpn=256 so decoder parameters are matched.
    """
    def __init__(self, backbone_name='resnet50.a1_in1k', num_classes=5, d_fpn=256, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.d_fpn = d_fpn

        # Load backbone with multi-stage features
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True
        )
        feature_info = self.encoder.feature_info.channels()
        # feature_info contains channels for the 4 stages (strides 4, 8, 16, 32)
        assert len(feature_info) >= 4, f"Backbone {backbone_name} must provide at least 4 feature stages."
        self.stage_channels = feature_info[-4:]  # C2, C3, C4, C5

        # 1. Standardized 1x1 lateral projections to d_fpn
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, d_fpn, kernel_size=1) for ch in self.stage_channels
        ])

        # 2. Siamese Fusion modules: concatenates [f_pre, f_post, |f_post - f_pre|] -> 3 * d_fpn -> d_fpn
        self.fusion_convs = nn.ModuleList([
            ConvBNReLU(3 * d_fpn, d_fpn, kernel_size=3, padding=1) for _ in range(4)
        ])

        # 3. Top-down FPN smooth convs
        self.smooth_convs = nn.ModuleList([
            ConvBNReLU(d_fpn, d_fpn, kernel_size=3, padding=1) for _ in range(4)
        ])

        # 4. Final classification head
        self.head = nn.Sequential(
            ConvBNReLU(4 * d_fpn, d_fpn, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(d_fpn, num_classes, kernel_size=1)
        )

    def extract_features(self, x):
        """Extract multi-stage features and apply uniform 1x1 lateral projection to d_fpn."""
        raw_feats = self.encoder(x)[-4:]
        projected = [lat(f) for lat, f in zip(self.lateral_convs, raw_feats)]
        return projected

    def forward(self, pre_img, post_img):
        B, _, H, W = pre_img.shape

        # Extract features for pre and post disaster
        feats_pre = self.extract_features(pre_img)
        feats_post = self.extract_features(post_img)

        # Siamese Difference Fusion at each stage
        fused = []
        for i in range(4):
            f_pre = feats_pre[i]
            f_post = feats_post[i]
            diff = torch.abs(f_post - f_pre)
            cat = torch.cat([f_pre, f_post, diff], dim=1) # [B, 3*d_fpn, H_i, W_i]
            fused.append(self.fusion_convs[i](cat))

        # Top-down FPN pathway (from Stage 4 down to Stage 1)
        # fused[3] is stride 32, fused[0] is stride 4
        m5 = fused[3]
        m4 = fused[2] + F.interpolate(m5, size=fused[2].shape[-2:], mode='bilinear', align_corners=False)
        m3 = fused[1] + F.interpolate(m4, size=fused[1].shape[-2:], mode='bilinear', align_corners=False)
        m2 = fused[0] + F.interpolate(m3, size=fused[0].shape[-2:], mode='bilinear', align_corners=False)

        # Smooth features
        p5 = self.smooth_convs[3](m5)
        p4 = self.smooth_convs[2](m4)
        p3 = self.smooth_convs[1](m3)
        p2 = self.smooth_convs[0](m2)

        # Upsample all pyramid levels to stride 4 (p2 scale)
        p5_up = F.interpolate(p5, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)

        fused_all = torch.cat([p2, p3_up, p4_up, p5_up], dim=1) # [B, 4*d_fpn, H/4, W/4]

        # Classification logits
        logits_stride4 = self.head(fused_all)
        logits = F.interpolate(logits_stride4, size=(H, W), mode='bilinear', align_corners=False)

        return logits
