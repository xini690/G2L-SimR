import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class TextAttentionFusion(nn.Module):
    def __init__(self, feat_dim, text_dim):
        super().__init__()
        self.query_proj = nn.Conv2d(feat_dim, feat_dim, 1)
        self.key_proj = nn.Linear(text_dim, feat_dim)
        self.value_proj = nn.Linear(text_dim, feat_dim)
        self.scale = feat_dim ** -0.5

    def forward(self, feat_map, text_feat):
        B, C, H, W = feat_map.shape
        query = self.query_proj(feat_map).flatten(2).permute(0, 2, 1)
        key = self.key_proj(text_feat).unsqueeze(1)
        value = self.value_proj(text_feat).unsqueeze(1)
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-2)
        out = (attn @ value).permute(0, 2, 1).view(B, C, H, W)
        return feat_map + out


class G2L(nn.Module):
    def __init__(self,
                 encoder_name="resnet50",
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=1,
                 text_dim=512,
                 fusion_mode="attention"):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )

        self.encoder_channels = self.unet.encoder.out_channels
        self.text_dim = text_dim
        self.fusion_mode = fusion_mode

        self.text_projs = nn.ModuleList([
            nn.Linear(text_dim, c) for c in self.encoder_channels
        ])

        self.skip_projs = nn.ModuleList([
            nn.Conv2d(self.encoder_channels[i-1], c, kernel_size=1) if i > 0 else nn.Identity()
            for i, c in enumerate(self.encoder_channels)
        ])

        if fusion_mode == "attention":
            self.fusions = nn.ModuleList([
                TextAttentionFusion(c, text_dim) for c in self.encoder_channels
            ])
        elif fusion_mode == "concat":
            self.fusions_conv = nn.ModuleList([
                nn.Conv2d(c * 2, c, kernel_size=3, padding=1) for c in self.encoder_channels
            ])

    def forward(self, image, clip_feats):
        features = self.unet.encoder(image)
        fused_features = []
        skip = None

        for i, feat in enumerate(features):

            text_proj = self.text_projs[i](clip_feats).unsqueeze(-1).unsqueeze(-1)
            text_proj = text_proj.expand(-1, -1, feat.size(2), feat.size(3))

            if skip is not None:
                skip_proj = self.skip_projs[i](skip)
                skip_proj = F.interpolate(skip_proj, size=(feat.size(2), feat.size(3)), mode='bilinear', align_corners=False)
                feat = feat + skip_proj


            if self.fusion_mode == "add":
                fused = feat + text_proj
            elif self.fusion_mode == "concat":
                fused = torch.cat([feat, text_proj], dim=1)
                fused = self.fusions_conv[i](fused)
            elif self.fusion_mode == "attention":
                fused = self.fusions[i](feat, clip_feats)
            else:
                raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

            fused_features.append(fused)
            skip = fused

        decoder_output = self.unet.decoder(*fused_features)
        masks = self.unet.segmentation_head(decoder_output)
        return masks
