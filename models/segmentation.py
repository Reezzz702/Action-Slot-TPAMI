import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPixelDecoder3D(nn.Module):
    def __init__(self, in_channels=512, num_classes=65):
        super(ConvPixelDecoder3D, self).__init__()

        self.stage0 = self._block(in_channels, in_channels)
        self.stage1 = self._block(in_channels, 256) # 1/16
        self.stage2 = self._block(256, 128) # 1/8
        self.stage3 = self._block(128, num_classes) # 1/4
        # self.stage4 = self._block(64, 32) # 1/2
        # self.stage5 = self._block(32, num_classes) # 1/1

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stage0(x)
        for stage in [self.stage1, self.stage2, self.stage3]:
            # Upsample only spatial dimensions (H, W), keep T the same
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
            x = stage(x)
        x = F.softmax(x, dim=1)
        return x
    
class ActionSegmentation(nn.Module):
    def __init__(self, num_classes=64):
        super().__init__()
        self.resnet = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        self.resnet = self.resnet.blocks[:-1]
        for t in self.resnet.parameters():
            t.requires_grad=False
        for t in self.resnet[-1].parameters():
            t.requires_grad=True
        for t in self.resnet[-2].parameters():
            t.requires_grad=True
            
        self.in_c = 192
        self.drop = nn.Dropout(p=0.5)
        
        self.resolution = (8, 24)
        self.resolution3d = (16, 8, 24)
        self.conv3d = nn.Sequential(
                    nn.ReLU(),
                    nn.BatchNorm3d(self.in_c),
                    nn.Conv3d(self.in_c, 256, (1, 1, 1), stride=1),
                    nn.ReLU(),)
        
        self.decoder = ConvPixelDecoder3D(in_channels=256, num_classes=num_classes)
        # self.decoder = ViViTSegmentationDecoder(in_channels=256, num_classes=num_classes)
        
    def forward(self, x):
        batch_size, _, seq_len, height, width = x.shape
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)
            print(x.shape)
            
        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]
        
        # [b, c, n, w, h]
        x = self.conv3d(x)
        
        x = x.permute((0, 2, 3, 4, 1))
        # [bs, n, w, h, c]
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))
        x = x.permute((0, 4, 1, 2, 3))
        pred_mask = self.decoder(x)
        return pred_mask
    
    
class FPN_3DDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=64):
        super(FPN_3DDecoder, self).__init__()

        self.conv_l3 = self._block(in_channels_list[0], 256)  # input from last encoder block
        self.conv_l2 = self._block(in_channels_list[1], 128)  # second last
        self.conv_l1 = self._block(in_channels_list[2], 64)   # third last
        self.conv_l0 = self._block(in_channels_list[3], 64)   # third last

        self.lateral3 = nn.Conv3d(256, 128, kernel_size=1)
        self.lateral2 = nn.Conv3d(128, 128, kernel_size=1)
        self.lateral1 = nn.Conv3d(64, 128, kernel_size=1)
        self.lateral0 = nn.Conv3d(64, 128, kernel_size=1)

        self.final_conv = nn.Conv3d(128, out_channels, kernel_size=3, padding=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        l3, l2, l1, l0 = features  # from deeper to shallower

        # Transform features
        l3 = self.conv_l3(l3)  # 8x24 → 16x48
        l2 = self.conv_l2(l2)  # 16x48 → 32x96
        l1 = self.conv_l1(l1)  # 32x96 → 64x192
        l0 = self.conv_l0(l0)  # 32x96 → 64x192

        # Laterals
        l3_up = F.interpolate(self.lateral3(l3), size=l2.shape[-3:], mode='trilinear', align_corners=False)
        l2 = self.lateral2(l2) + l3_up

        l2_up = F.interpolate(l2, size=l1.shape[-3:], mode='trilinear', align_corners=False)
        l1 = self.lateral1(l1) + l2_up

        l1_up = F.interpolate(l1, size=l0.shape[-3:], mode='trilinear', align_corners=False)
        l0 = self.lateral0(l0) + l1_up

        # Final output
        out = self.final_conv(l0)  # shape: [B, 64, 16, 64, 192]
        out = F.softmax(out, dim=1)
        return out


class ActionSegmentation_FPN(nn.Module):
    def __init__(self, num_classes=65):
        super(ActionSegmentation_FPN, self).__init__()
        self.num_classes = num_classes

        # Load pretrained encoder
        self.resnet = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        self.resnet = self.resnet.blocks[:-1]

        # Freeze all except last two blocks
        for t in self.resnet.parameters():
            t.requires_grad = False
        for t in self.resnet[-1].parameters():
            t.requires_grad = True
        for t in self.resnet[-2].parameters():
            t.requires_grad = True
        for t in self.resnet[-3].parameters():
            t.requires_grad = True
        for t in self.resnet[-4].parameters():
            t.requires_grad = True

        self.encoder_out_channels = [192, 96, 48, 24]  # last three encoder layers

        # 1x1 conv to reduce to fixed dimensions
        self.reduce_c3 = nn.Conv3d(192, 256, kernel_size=1)
        self.reduce_c2 = nn.Conv3d(96, 128, kernel_size=1)
        self.reduce_c1 = nn.Conv3d(48, 64, kernel_size=1)
        self.reduce_c0 = nn.Conv3d(24, 64, kernel_size=1)

        self.decoder = FPN_3DDecoder(
            in_channels_list=[256, 128, 64, 64],
            out_channels=num_classes
        )

    def forward(self, x):
        batch_size, _, seq_len, h, w = x.shape
        features = []
        for i in range(len(self.resnet)):
            x = self.resnet[i](x)
            if i in [1, 2, 3, 4]:  # save features from these 3 blocks
                features.append(x)

        # Reduce channels
        c0 = self.reduce_c0(features[0])  # 48 → 64
        c1 = self.reduce_c1(features[1])  # 48 → 64
        c2 = self.reduce_c2(features[2])  # 96 → 128
        c3 = self.reduce_c3(features[3])  # 192 → 256

        out = self.decoder([c3, c2, c1, c0])  # Output shape: [B, 64, 16, 64, 192]
        return out
    

# class ConvPixelDecoder3D(nn.Module):
#     def __init__(self, in_channels=256, out_channels=65):
#         super().__init__()

#         self.relu = nn.ReLU(inplace=True)

#         self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv3d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv3d(in_channels // 8, out_channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         # x: [B, C, T, H, W]
#         x = self.relu(self.conv1(x))
#         x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)

#         x = self.relu(self.conv2(x))
#         x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)

#         x = self.relu(self.conv3(x))
#         x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)

#         x = self.conv4(x)
#         x = F.softmax(x, dim=1)
#         return x  # [B, N, T, 8H, 8W]

class ReferringDecoder3D(nn.Module):
    def __init__(self, in_channels, num_classes=64, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels

        self.class_proj = nn.Linear(64, 256)
        self.fuse_conv = nn.Conv3d(in_channels + 1, hidden_dim, kernel_size=3, padding=1)

        self.up1 = self._up_block(hidden_dim, hidden_dim // 2)  # 2x
        self.up2 = self._up_block(hidden_dim // 2, hidden_dim // 4)  # 4x
        self.up3 = self._up_block(hidden_dim // 4, 1)  # 8x output (binary) (16, 64, 192)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, class_onehot, attn_map):
        """
        feat:        (B, D, T, H, W)
        class_onehot:(B, C)
        attn_map:    (B, T, H, W)
        """
        B, D, T, H, W = feat.shape


        # Class-guided feature weighting
        class_embed = self.class_proj(class_onehot)  # (B, D)
        class_embed = class_embed.view(B, D, 1, 1, 1)  # (B, D, 1, 1, 1)
        feat = feat * class_embed  # (B, D, T, H, W)

        # Add attention map as extra channel
        x = torch.cat([feat, attn_map], dim=1)  # (B, D+1, T, H, W)

        # Fuse and upsample
        x = self.fuse_conv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = x.squeeze(1)  # (B, T, H', W')
        return x