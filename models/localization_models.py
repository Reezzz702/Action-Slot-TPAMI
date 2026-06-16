import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from collections import OrderedDict

class FasterRCNN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		# Load pretrained VGG16 backbone
		backbone = torchvision.models.vgg16(weights="IMAGENET1K_V1").features
		backbone.out_channels = 512

		# Anchor generator
		anchor_generator = AnchorGenerator(
			sizes=((32, 64, 128, 256, 512),),
			aspect_ratios=((0.5, 1.0, 2.0),)
		)
		roi_pooler = torchvision.ops.MultiScaleRoIAlign(
			featmap_names=['0'],
			output_size=7,
			sampling_ratio=2
		)

		# Build FasterRCNN
		self.model = FasterRCNN(
			backbone,
			num_classes=91,  # dummy for Visual Genome
			rpn_anchor_generator=anchor_generator,
			box_roi_pool=roi_pooler
		)

		# Top 100 proposals only
		self.model.rpn.post_nms_top_n_test = 100
		self.model.rpn.post_nms_top_n_train = 100

	@torch.no_grad()
	def forward(self, images):
		"""
		Returns:
		- proposals: Tensor[B, 100, 5] (x1, y1, x2, y2, area)
		- features: Tensor[B, 100, feature_dim]
		"""
		self.model.eval()

		original_image_sizes = [img.shape[-2:] for img in images]
		images, _ = self.model.transform(images)

		features = self.model.backbone(images.tensors)
		if isinstance(features, torch.Tensor):
			features = OrderedDict([('0', features)])

		proposals, _ = self.model.rpn(images, features, None)

		all_props = []
		all_feats = []

		for props, img_size in zip(proposals, images.image_sizes):
			# ROI Align for this image
			box_features = self.model.roi_heads.box_roi_pool(features, [props], [img_size])  # (N, 256, 7, 7)
			flattened_features = box_features.flatten(start_dim=1)  # (N, 12544)

			# Compute area
			x1, y1, x2, y2 = props[:, 0], props[:, 1], props[:, 2], props[:, 3]
			area = (x2 - x1) * (y2 - y1)
			proposals_with_area = torch.stack([x1, y1, x2, y2, area], dim=1)  # (N, 5)

			# Pad if fewer than 100 proposals
			if proposals_with_area.shape[0] < 100:
				pad_size = 100 - proposals_with_area.shape[0]
				proposals_with_area = torch.cat([
					proposals_with_area,
					torch.zeros((pad_size, 5), device=proposals_with_area.device)
				], dim=0)
				flattened_features = torch.cat([
					flattened_features,
					torch.zeros((pad_size, flattened_features.shape[1]), device=flattened_features.device)
				], dim=0)
			else:
				proposals_with_area = proposals_with_area[:100]
				flattened_features = flattened_features[:100]

			all_props.append(proposals_with_area.unsqueeze(0))
			all_feats.append(flattened_features.unsqueeze(0))

		proposals_out = torch.cat(all_props, dim=0)   # (B, 100, 5)
		features_out = torch.cat(all_feats, dim=0)    # (B, 100, 12544)

		return proposals_out, features_out
	
	
class MIL_baseline(nn.Module):
	def __init__(self, feat_dim=192, hidden_dim=128, num_classes=64):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.obj_mlp = nn.Sequential(
			nn.Conv3d(feat_dim, hidden_dim, kernel_size=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep T, pool over H, W
		)
		self.cls_mlp = nn.Sequential(
			nn.Linear(num_classes, hidden_dim),
			nn.ReLU()
		)

	def forward(self, feat_map, obj_mask, onehot, obj_num, mode='train'):
		"""
		Args:
			feat_map:   [B, D, T, H, W]
			obj_mask:   [B, N_obj, T, H, W]
			onehot:  [B, C]
			obj_num:    [B, T]  -> number of objects in each frame
		Returns:
			obj_feat:      [B, 64, T, D]
			all_cls_embed: [B, C, T, D]
		"""
		B, D, T, H, W = feat_map.shape
		N_obj = obj_mask.shape[1]  # max possible objects (should be ≥ 64)
		device = feat_map.device

		# Process spatial features
		feat_map_proj = self.obj_mlp(feat_map)  # [B, D', T, H, W]
		D_ = feat_map_proj.shape[1]

		# Initialize padded output: [B, 100, T, D]
		max_obj = 100
		obj_feat_padded = torch.zeros(B, max_obj, T, D_, device=device)

		# Valid mask: 1 where obj is valid
		obj_valid_mask = torch.zeros(B, max_obj, T, dtype=torch.bool, device=device)

		for b in range(B):
			for t in range(T):
				valid_obj_count = obj_num[b, t]
				obj_valid_mask[b, :valid_obj_count, t] = True
				for i in range(valid_obj_count):
					mask = obj_mask[b, i, t]  # [H, W]
					if mask.sum() == 0:
						continue  # skip empty masks
					mask = mask[None, None, :, :]  # [1, 1, H, W]
					feat = feat_map_proj[b, :, t]  # [D', H, W]
					feat = feat[None] * mask.float()  # [1, D', H, W]
					avg_feat = feat.view(D_, -1).sum(dim=1) / (mask.sum() + 1e-6)  # [D']
					obj_feat_padded[b, i, t] = avg_feat  # fill into padded output
	 

		cls_emb = self.cls_mlp(onehot)  # [B, D']
		cls_feat = cls_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, D']

		return obj_feat_padded, cls_feat, obj_valid_mask
	
	def inference(self, feat_map, obj_mask, ori_obj_mask, query_onehot, obj_num):
		"""
		Args:
			feat_map:   [B, D, T, H, W]
			obj_mask:   [B, N_obj, T, H, W]
			ori_obj_mask:   [B, N_obj, T, H', W']
			query_onehot:  [B, C]  # one-hot or multi-hot class vector
			obj_num:    [B, T]     # number of objects in each frame
		Returns:
			pred_mask: [B, 16, 64, 192]
		"""
		B, D, T, H, W = feat_map.shape
		N_obj = obj_mask.shape[1]
		device = feat_map.device

		# === Feature Projection ===
		feat_map_proj = self.obj_mlp(feat_map)  # [B, D', T, H, W]
		D_ = feat_map_proj.shape[1]

		# === Init padded outputs ===
		max_obj = 100
		obj_feat_padded = torch.zeros(B, max_obj, T, D_, device=device)
		obj_valid_mask = torch.zeros(B, max_obj, T, dtype=torch.bool, device=device)

		for b in range(B):
			for t in range(T):
				valid_obj_count = min(obj_num[b, t].item(), max_obj)
				obj_valid_mask[b, :valid_obj_count, t] = True
				for i in range(valid_obj_count):
					mask = obj_mask[b, i, t]  # [H, W]
					if mask.sum() == 0:
						continue
					mask = mask[None, None, :, :]
					feat = feat_map_proj[b, :, t]
					feat = feat[None] * mask.float()
					avg_feat = feat.view(D_, -1).sum(dim=1) / (mask.sum() + 1e-6)
					obj_feat_padded[b, i, t] = avg_feat

		# === Class feature embedding ===
		cls_emb = self.cls_mlp(query_onehot)  # [B, D']
		cls_feat = cls_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, D']

		# === Normalize features ===
		# obj_feat_padded = F.normalize(obj_feat_padded, dim=-1)
		# cls_feat = F.normalize(cls_feat, dim=-1)

		# === Cosine similarity: A(V_t, q) ===
		sim = torch.einsum('bntd,btd->bnt', obj_feat_padded, cls_feat)  # [B, N, T]

		# === NEW: Use only object with max similarity per frame ===
		# Get the index of the object with max sim at each frame
		max_idx = sim.argmax(dim=1)  # [B, T]

		# Convert to one-hot selection: [B, N, T]
		selected = torch.zeros_like(sim, dtype=torch.bool)
		for b in range(B):
			for t in range(T):
				selected[b, max_idx[b, t], t] = True

		selected = selected.unsqueeze(-1).unsqueeze(-1)  # [B, N, T, 1, 1]

		# === Mask the original object masks ===
		masked_ori_obj = ori_obj_mask * selected  # [B, N, T, H', W']
		pred_mask = masked_ori_obj.max(dim=1).values  # [B, T, H', W']
		pred_mask = pred_mask.unsqueeze(1).float()

		# === Upsample to desired resolution ===
		pred_mask = F.interpolate(pred_mask, size=(16, 64, 192), mode='trilinear', align_corners=False)
		pred_mask = pred_mask.squeeze(1)

		return pred_mask  # [B, 16, 64, 192]


class MIL_baseline_attn(nn.Module):
	def __init__(self, feat_dim=192, hidden_dim=128):
		super().__init__()
		self.hidden_dim = hidden_dim

		# Project video feature map to object-level features
		self.obj_mlp = nn.Sequential(
			nn.Conv3d(feat_dim, hidden_dim, kernel_size=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool3d((None, 1, 1))  # [B, D', T, 1, 1]
		)

		# Project attention maps to frame-level cls_feat
		self.cls_mlp = nn.Sequential(
			nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),  # [B*T, 1, H, W] -> [B*T, D', H, W]
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((1, 1))  # [B*T, D', 1, 1]
		)

	def forward(self, feat_map, obj_mask, attn_map, obj_num):
		"""
		Args:
			feat_map:   [B, D, T, H, W]
			obj_mask:   [B, N_obj, T, H, W]
			attn_map:   [B, T, H, W] - attention map per frame
			obj_num:    [B, T] - number of objects in each frame

		Returns:
			obj_feat:        [B, N_obj, T, D']
			cls_feat:        [B, T, D']
			obj_valid_mask:  [B, N_obj, T]
		"""
		B, D, T, H, W = feat_map.shape
		N_obj = obj_mask.shape[1]
		device = feat_map.device

				# Process spatial features
		feat_map_proj = self.obj_mlp(feat_map)  # [B, D', T, H, W]
		D_ = feat_map_proj.shape[1]

		# Initialize padded output: [B, 100, T, D]
		max_obj = 100
		obj_feat_padded = torch.zeros(B, max_obj, T, D_, device=device)

		# Valid mask: 1 where obj is valid
		obj_valid_mask = torch.zeros(B, max_obj, T, dtype=torch.bool, device=device)

		for b in range(B):
			for t in range(T):
				valid_obj_count = obj_num[b, t]
				obj_valid_mask[b, :valid_obj_count, t] = True
				for i in range(valid_obj_count):
					mask = obj_mask[b, i, t]  # [H, W]
					if mask.sum() == 0:
						continue  # skip empty masks
					mask = mask[None, None, :, :]  # [1, 1, H, W]
					feat = feat_map_proj[b, :, t]  # [D', H, W]
					feat = feat[None] * mask.float()  # [1, D', H, W]
					avg_feat = feat.view(D_, -1).sum(dim=1) / (mask.sum() + 1e-6)  # [D']
					obj_feat_padded[b, i, t] = avg_feat  # fill into padded output


		# === Project attention map to class feature ===
		attn_map = attn_map.unsqueeze(2)  # [B, T, 1, H, W]
		attn_map = attn_map.reshape(B * T, 1, H, W)  # [B*T, 1, H, W]

		cls_feat = self.cls_mlp(attn_map)  # [B*T, D', 1, 1]
		cls_feat = cls_feat.view(B, T, D_)  # [B, T, D']

		return obj_feat_padded, cls_feat, obj_valid_mask
	
	def inference(self, feat_map, obj_mask, ori_obj_mask, attn_map, obj_num):
		"""
		Args:
			feat_map:   [B, D, T, H, W]
			obj_mask:   [B, N_obj, T, H, W]
			ori_obj_mask:   [B, N_obj, T, H', W']
			query_onehot:  [B, C]  # one-hot or multi-hot class vector
			obj_num:    [B, T]     # number of objects in each frame
		Returns:
			pred_mask: [B, 16, 64, 192]
		"""
		B, D, T, H, W = feat_map.shape
		N_obj = obj_mask.shape[1]
		device = feat_map.device

		# === Feature Projection ===
		feat_map_proj = self.obj_mlp(feat_map)  # [B, D', T, H, W]
		D_ = feat_map_proj.shape[1]

		# === Init padded outputs ===
		max_obj = 100
		obj_feat_padded = torch.zeros(B, max_obj, T, D_, device=device)
		obj_valid_mask = torch.zeros(B, max_obj, T, dtype=torch.bool, device=device)

		for b in range(B):
			for t in range(T):
				valid_obj_count = min(obj_num[b, t].item(), max_obj)
				obj_valid_mask[b, :valid_obj_count, t] = True
				for i in range(valid_obj_count):
					mask = obj_mask[b, i, t]  # [H, W]
					if mask.sum() == 0:
						continue
					mask = mask[None, None, :, :]
					feat = feat_map_proj[b, :, t]
					feat = feat[None] * mask.float()
					avg_feat = feat.view(D_, -1).sum(dim=1) / (mask.sum() + 1e-6)
					obj_feat_padded[b, i, t] = avg_feat

		# === Class feature embedding ===
		attn_map = attn_map.unsqueeze(2)  # [B, T, 1, H, W]
		attn_map = attn_map.reshape(B * T, 1, H, W)  # [B*T, 1, H, W]

		cls_feat = self.cls_mlp(attn_map)  # [B*T, D', 1, 1]
		cls_feat = cls_feat.view(B, T, D_)  # [B, T, D']

		# === Normalize features ===
		# obj_feat_padded = F.normalize(obj_feat_padded, dim=-1)
		# cls_feat = F.normalize(cls_feat, dim=-1)

		# === Cosine similarity: A(V_t, q) ===
		sim = torch.einsum('bntd,btd->bnt', obj_feat_padded, cls_feat)  # [B, N, T]

		# === NEW: Use only object with max similarity per frame ===
		# Get the index of the object with max sim at each frame
		max_idx = sim.argmax(dim=1)  # [B, T]

		# Convert to one-hot selection: [B, N, T]
		selected = torch.zeros_like(sim, dtype=torch.bool)
		for b in range(B):
			for t in range(T):
				selected[b, max_idx[b, t], t] = True

		selected = selected.unsqueeze(-1).unsqueeze(-1)  # [B, N, T, 1, 1]

		# === Mask the original object masks ===
		masked_ori_obj = ori_obj_mask * selected  # [B, N, T, H', W']
		pred_mask = masked_ori_obj.max(dim=1).values  # [B, T, H', W']
		pred_mask = pred_mask.unsqueeze(1).float()

		# === Upsample to desired resolution ===
		pred_mask = F.interpolate(pred_mask, size=(16, 64, 192), mode='trilinear', align_corners=False)
		pred_mask = pred_mask.squeeze(1)

		return pred_mask  # [B, 16, 64, 192]

class ActionSegmentationRefiner(nn.Module):
	def __init__(self, low_dim=192, high_dim=48):
		super().__init__()
		self.attn_proj = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, low_dim, kernel_size=3, padding=1),
			nn.Sigmoid(),  # attention modulator
		)

		self.low_res_fuser = nn.Sequential(
			nn.Conv3d(low_dim, low_dim, kernel_size=3, padding=1),
			nn.BatchNorm3d(low_dim),
			nn.ReLU(),
		)

		self.high_res_decoder = nn.Sequential(
			nn.Conv3d(low_dim + high_dim, 128, kernel_size=3, padding=1),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(64, 1, kernel_size=1),
		)

	def forward(self, feat_low, feat_high, attn):
		B, D1, T, H, W = feat_low.shape
		_, D2, _, H4, W4 = feat_high.shape

		# 1. Process attention map to modulate low-res feature
		attn = attn.unsqueeze(2)  # B, T, 1, H, W
		attn_2d = attn.reshape(B * T, 1, H, W)
		attn_weight = self.attn_proj(attn_2d)  # [B*T, D1, H, W]
		attn_weight = attn_weight.view(B, T, D1, H, W).permute(0, 2, 1, 3, 4)  # [B, D1, T, H, W]

		feat_low = feat_low * attn_weight  # modulate low-res features
		feat_low = self.low_res_fuser(feat_low)  # contextualized

		# 2. Upsample low-res features and mask
		feat_low_up = F.interpolate(feat_low, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
		# fg_mask_up = F.interpolate(fg_mask.float().unsqueeze(1), mode='nearest')  # B,1,T,4H,4W

		# 3. Fuse high-res features, upsampled low-res, and mask
		x = torch.cat([feat_low_up, feat_high], dim=1)  # [B, D1+D2+1, T, 4H, 4W]

		pred = self.high_res_decoder(x)  # B, 1, T, 4H, 4W
		pred = pred.squeeze(1)  # B, T, 4H, 4W
		return pred
	
class CrossAttentionFusion(nn.Module):
	def __init__(self, feat_dim=192, high_dim=48, embed_dim=256, heads=4):
		super().__init__()
		# feat_dim = 192
		# high_dim = 48
		feat_dim=2048
		high_dim = 512

		self.embed_dim = embed_dim

		# Project spatiotemporal features and attention into embedding space
		self.proj_feat = nn.Conv3d(feat_dim, embed_dim, kernel_size=1, bias=True)
		self.proj_attn = nn.Conv2d(1, embed_dim, kernel_size=1, bias=True)

		self.to_q = nn.Linear(embed_dim, embed_dim)
		self.to_k = nn.Linear(embed_dim, embed_dim)
		self.to_v = nn.Linear(embed_dim, embed_dim)


		# Cross-attention mechanism
		self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)

		# Decoder head
		self.high_res_decoder = nn.Sequential(
			nn.Conv3d(embed_dim + high_dim, 128, kernel_size=3, padding=1, bias=True),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=True),
			nn.ReLU(),
			nn.Conv3d(64, 1, kernel_size=1, bias=True),  # Add bias here
		)

		# Initialize final conv bias to encourage low initial predictions
		nn.init.constant_(self.high_res_decoder[-1].bias, 1.0)

	def forward(self, feat_low, feat_high, attn):
		B, C1, T, H, W = feat_low.shape
		C2 = feat_high.shape[1]

		
		attn = attn.view(B * T, 1, H, W) 
		attn_proj = self.proj_attn(attn)  # [B*T, D, H, W]
		attn_proj = attn_proj.flatten(2).transpose(1, 2)  # [B*T, HW, D]

		feat_proj = self.proj_feat(feat_low)  # [B, D, T, H, W]
  
		# Flatten spatial dimensions for attention map and low-res features
		attn = attn.view(B * T, 1, H, W)
		attn_proj = self.proj_attn(attn)  # [B*T, D, H, W]
		attn_flat = attn_proj.flatten(2).transpose(1, 2)  # [B*T, HW, D]
  
		feat_proj = self.proj_feat(feat_low)  # [B, D, T, H, W]
		feat_flat = feat_proj.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, self.embed_dim)  # [B*T, HW, D]

		# === 3. Map to QKV ===
		Q = self.to_q(attn_flat)  # [B*T, HW, D]
		K = self.to_k(feat_flat)  # [B*T, HW, D]
		V = self.to_v(feat_flat)  # [B*T, HW, D]

		# === 4. Cross-attention ===
		fused, _ = self.attn(query=Q, key=K, value=V)  # [B*T, HW, D]
		fused = fused.transpose(1, 2).view(B, T, self.embed_dim, H, W).permute(0, 2, 1, 3, 4)  # [B, D, T, H, W]

		fused_up = F.interpolate(fused, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
		x = torch.cat([fused_up, feat_high], dim=1)  # [B, D+C2, T, 4H, 4W]

		pred = self.high_res_decoder(x).squeeze(1)  # [B, T, 4H, 4W]
		return pred

	# def forward(self, feat_low, feat_high, attn):
	# 	B, C1, T, H, W = feat_low.shape
	# 	C2 = feat_high.shape[1]

	# 	attn = attn.view(B * T, 1, H, W)
	# 	attn_proj = self.proj_attn(attn)  # [B*T, D, H, W]
	# 	attn_proj = attn_proj.flatten(2).transpose(1, 2)  # [B*T, HW, D]

	# 	feat_proj = self.proj_feat(feat_low)  # [B, D, T, H, W]
	# 	feat_proj = feat_proj.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, self.embed_dim)  # [B*T, HW, D]

	# 	# Cross-attention: Q=attn, K/V=features
	# 	fused, _ = self.attn(query=attn_proj, key=feat_proj, value=feat_proj)  # [B*T, HW, D]
	# 	fused = fused.transpose(1, 2).view(B, T, self.embed_dim, H, W).permute(0, 2, 1, 3, 4)  # [B, D, T, H, W]

	# 	fused_up = F.interpolate(fused, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
	# 	x = torch.cat([fused_up, feat_high], dim=1)  # [B, D+C2, T, 4H, 4W]

	# 	pred = self.high_res_decoder(x).squeeze(1)  # [B, T, 4H, 4W]
	# 	return pred
	
class CrossAttentionFusion_OneHot(nn.Module):
	def __init__(self, feat_dim=192, high_dim=48, embed_dim=256, heads=4, num_classes=64):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_classes = num_classes

		self.to_q = nn.Linear(embed_dim, embed_dim)
		self.to_k = nn.Linear(feat_dim, embed_dim)
		self.to_v = nn.Linear(feat_dim, embed_dim)
  
		# Embed one-hot input into the same space as attention map would be
		self.class_embed = nn.Linear(num_classes, embed_dim)

		# Cross-attention mechanism
		self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)

		# Decoder head
		self.high_res_decoder = nn.Sequential(
			nn.Conv3d(embed_dim + high_dim, 128, kernel_size=3, padding=1, bias=True),
			nn.BatchNorm3d(128),
			nn.ReLU(),
			nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=True),
			nn.ReLU(),
			nn.Conv3d(64, 1, kernel_size=1, bias=True),
		)

		# Bias init to favor background at start
		nn.init.constant_(self.high_res_decoder[-1].bias, 1.0)

	# def forward(self, feat_low, feat_high, onehot):
	# 	"""
	# 	Args:
	# 		feat_low: [B, C1, T, H, W]
	# 		feat_high: [B, C2, T, H*4, W*4]
	# 		onehot: [B, num_classes]  - binary vector of class presence
	# 	Returns:
	# 		pred: [B, T, H*4, W*4]
	# 	"""
	# 	B, C1, T, H, W = feat_low.shape
	# 	C2 = feat_high.shape[1]

	# 	# === 1. Embed One-Hot ===
	# 	class_query = self.class_embed(onehot.float())  # [B, D]
	# 	query = class_query.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
	# 	query = query.reshape(B * T, 1, self.embed_dim)     # [B*T, 1, D]

	# 	# === 2. Project video features ===
	# 	feat_proj = self.proj_feat(feat_low)  # [B, D, T, H, W]
	# 	feat_proj = feat_proj.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, self.embed_dim)  # [B*T, HW, D]

	# 	# === 3. Cross-attention: query from class onehot, key/value from features ===
	# 	fused, _ = self.attn(query=query, key=feat_proj, value=feat_proj)  # [B*T, 1, D]
	# 	fused = fused.view(B, T, self.embed_dim).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D, T, 1, 1]
	# 	fused = fused.expand(-1, -1, -1, H, W)  # Broadcast to spatial map [B, D, T, H, W]

	# 	# === 4. Fuse with high-res features ===
	# 	fused_up = F.interpolate(fused, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)  # [B, D, T, 4H, 4W]
	# 	x = torch.cat([fused_up, feat_high], dim=1)  # [B, D+C2, T, 4H, 4W]

	# 	pred = self.high_res_decoder(x).squeeze(1)  # [B, T, 4H, 4W]
	# 	return pred
	def forward(self, feat_low, feat_high, onehot):
		"""
		Args:
			feat_low: [B, C1, T, H, W]
			feat_high: [B, C2, T, H*4, W*4]
			onehot: [B, num_classes]  - binary vector of class presence
		Returns:
			pred: [B, T, H*4, W*4]
		"""
		B, C1, T, H, W = feat_low.shape
		C2 = feat_high.shape[1]

		# === 1. Prepare query from one-hot ===
		class_query = self.class_embed(onehot.float())       # [B, D]
		query = class_query.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
		query = query.reshape(B * T, 1, self.embed_dim)     # [B*T, 1, D]
		Q = self.to_q(query)                                # [B*T, 1, D]

		# === 2. Flatten video features and project to K/V ===
		feat_flat = feat_low.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C1)  # [B*T, HW, C1]
		K = self.to_k(feat_flat)  # [B*T, HW, D]
		V = self.to_v(feat_flat)  # [B*T, HW, D]

		# === 3. Cross-attention ===
		fused, _ = self.attn(query=Q, key=K, value=V)  # [B*T, 1, D]
		fused = fused.view(B, T, self.embed_dim).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D, T, 1, 1]
		fused = fused.expand(-1, -1, -1, H, W)  # Broadcast to spatial map [B, D, T, H, W]

		# === 4. Fuse with high-res features ===
		fused_up = F.interpolate(fused, scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)  # [B, D, T, 4H, 4W]
		x = torch.cat([fused_up, feat_high], dim=1)  # [B, D+C2, T, 4H, 4W]

		# === 5. Decode ===
		pred = self.high_res_decoder(x).squeeze(1)  # [B, T, 4H, 4W]
		return pred