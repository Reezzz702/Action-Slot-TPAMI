import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import os
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import center_of_mass, maximum_filter, label, generate_binary_structure
from skimage.measure import regionprops, find_contours
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb, Normalize
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


actor_table = ['c:z1-z2', 'c:z1-z3', 'c:z1-z4',
				'c:z2-z1', 'c:z2-z3', 'c:z2-z4',
				'c:z3-z1', 'c:z3-z2', 'c:z3-z4',
				'c:z4-z1', 'c:z4-z2', 'c:z4-z3',
				'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
				'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
				'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
				'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
				'b:z1-z2', 'b:z1-z3', 'b:z1-z4',
				'b:z2-z1', 'b:z2-z3', 'b:z2-z4',
				'b:z3-z1', 'b:z3-z2', 'b:z3-z4',
				'b:z4-z1', 'b:z4-z2', 'b:z4-z3',
				'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
				'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
				'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
				'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
				'p:c1-c2', 'p:c1-c4', 
				'p:c2-c1', 'p:c2-c3', 
				'p:c3-c2', 'p:c3-c4', 
				'p:c4-c1', 'p:c4-c3', 
				'p+:c1-c2', 'p+:c1-c4', 
				'p+:c2-c1', 'p+:c2-c3', 
				'p+:c3-c2', 'p+:c3-c4', 
				'p+:c4-c1', 'p+:c4-c3',
				'bg']

def get_mean(norm_value=255, dataset='activitynet'):
		assert dataset in ['activitynet', 'kinetics']

		if dataset == 'activitynet':
				return [
						114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
				]
		elif dataset == 'kinetics':
				# Kinetics (10 videos for each class)
				return [
						110.63666788 / norm_value, 103.16065604 / norm_value,
						96.29023126 / norm_value
				]


def get_std(norm_value=255):
		# Kinetics (10 videos for each class)
		return [
				38.7568578 / norm_value, 37.88248729 / norm_value,
				40.02898126 / norm_value
		]

class AverageMeter(object):
	def __init__(self):
		self.val = None
		self.sum = None
		self.cnt = None
		self.avg = None
		self.ema = None
		self.initialized = False

	def update(self, val, n=1):
		if not self.initialized:
			self.initialize(val, n)
		else:
			self.add(val, n)

	def initialize(self, val, n):
		self.val = val
		self.sum = val * n
		self.cnt = n
		self.avg = val
		self.ema = val
		self.initialized = True

	def add(self, val, n):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt
		self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class=1, start_class=0):
		pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
		mask = mask.data.cpu().numpy().astype(np.uint8)
		pred = np.asarray(pred, dtype=np.uint8).copy()
		mask = np.asarray(mask, dtype=np.uint8).copy()
		inter = pred * (pred == mask)
		(area_inter, _) = np.histogram(inter, bins=num_class, range=(start_class, num_class))
		(area_pred, _) = np.histogram(pred, bins=num_class, range=(start_class, num_class))
		(area_mask, _) = np.histogram(mask, bins=num_class, range=(start_class, num_class))
		area_union = area_pred + area_mask - area_inter
		return (area_inter, area_union)


def plot_wp(gt_wp, pred_wp, target_point):
	# generate all white bev image
	bev_image = np.ones((512,1024,3), dtype=np.uint8)
	bev_image *= 255
	origin = (512, 512)
	loc_pixels_per_meter = 16
 
	if gt_wp is not None:
		gt_wp_color = (255, 255, 0)
		for wp in gt_wp.detach().cpu().numpy()[0]:
			wp_x = wp[1] * loc_pixels_per_meter + origin[0]
			wp_y = - wp[0] * loc_pixels_per_meter + origin[1]
			cv2.circle(bev_image, (int(wp_x), int(wp_y)), radius=10, color=gt_wp_color, thickness=-1)
				
	if pred_wp is not None:
		pred_wps = pred_wp.detach().cpu().numpy()[0]
		num_wp = len(pred_wps)
		for idx, wp in enumerate(pred_wps):
			color_weight = 0.5 + 0.5 * float(idx) / num_wp
			wp_x = wp[1] * loc_pixels_per_meter + origin[0]
			wp_y = - wp[0] * loc_pixels_per_meter + origin[1]
			cv2.circle(bev_image, (int(wp_x), int(wp_y)),
						radius=8,
						lineType=cv2.LINE_AA,
						color=(0, 0, int(color_weight * 255)),
						thickness=-1)
	
	if target_point is not None:
		x_tp = target_point[0][1] * loc_pixels_per_meter + origin[0]
		y_tp = - target_point[0][0] * loc_pixels_per_meter + origin[1]
		cv2.circle(bev_image, (int(x_tp), int(y_tp)), radius=12, lineType=cv2.LINE_AA, color=(255, 0, 0), thickness=-1)

	# bev_image = np.rot90(bev_image, k=1)
	return bev_image

def batch_binary_dilation(mask_batch, dilation_iters=1):
	"""
	Performs binary dilation on a batch of masks using PyTorch.
	
	Args:
		mask_batch (torch.Tensor): Input masks of shape (B, 1, H, W), dtype=torch.uint8 (binary masks).
		dilation_iters (int): Number of dilation iterations.
	
	Returns:
		torch.Tensor: Dilated masks of shape (B, 1, H, W), dtype=torch.uint8.
	"""
	device = mask_batch.device  # Maintain device (CPU/GPU)
	
	# Define a 3x3 cross-shaped structuring element (matches scipy default)
	kernel = torch.tensor([[0, 1, 0],
						   [1, 1, 1],
						   [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1,1,3,3)
	
	# Convert mask to float (conv2d requires float input)
	mask = mask_batch.float()

	# Perform dilation iteratively
	for _ in range(dilation_iters):
		mask = F.conv2d(mask, kernel, padding=1)  # 3x3 dilation (same padding)
		mask = (mask > 0).float()  # Convert back to binary (thresholding)
	
	return mask.to(torch.uint8)  # Convert back to uint8 (binary mask)

def binary_mask(mask, th=0.1):
	mask[mask>th] = 1
	mask[mask<=th] = 0
	return mask

 
def vis_inpaint(image_list, logdir, batch_num):
	image_list = image_list.permute(0,1,3,4,2).mul(255)
	gif_path = os.path.join(logdir, 'gif/inpaint')
	os.makedirs(gif_path, exist_ok=True)
	image_list = image_list[0].cpu().numpy()
	frames = [Image.fromarray(np.uint8(img)) for img in image_list]
	output_gif = os.path.join(gif_path, f"{batch_num}.gif")
	frames[0].save(
				output_gif,
				save_all=True,
				append_images=frames[1:],  # Add the remaining frames
				optimize=True,
				duration=200,  # Duration per frame in milliseconds
				loop=0         # Loop forever (set loop=1 for one loop only)
				)


def print_map_results(results, class_names=None, max_class_display=64):
	"""
	Nicely format and print the mAP results.

	Args:
		results (dict): Output from compute_batch_mask_map().
		class_names (list[str], optional): List of class names. Defaults to class indices.
		max_class_display (int): Max number of classes to display in the table.
	"""
	from tabulate import tabulate
	import math
 
	thresholds = torch.arange(0.5, 0.95 + 1e-6, 0.05)
	iou_thresholds = [round(t.item(), 2) for t in thresholds]
	ap_by_threshold = results['ap_by_threshold']
	per_class_AP = results['per_class_AP']
	mAP = results['mAP']

	print("📊 **Mask AP Evaluation**")
	print(f"➡️  mAP@[0.5:0.95]: **{mAP:.4f}**")
	print()

	print("🔍 AP per IoU threshold:")
	rows = [(f"{t:.2f}", f"{ap:.4f}" if not math.isnan(ap) else "NaN")
			for t, ap in zip(iou_thresholds, ap_by_threshold)]
	print(tabulate(rows, headers=["IoU Threshold", "mAP"], tablefmt="github"))
	print()

	print("🧩 Per-class AP (mean over thresholds):")
	cls_headers = ["Class", "AP"]
	cls_table = []
	print(np.array(per_class_AP).shape)
	for cls, ap in enumerate(per_class_AP):
		name = class_names[cls] if class_names else f"Class {cls}"
		cls_table.append((name, f"{ap:.4f}" if not math.isnan(ap) else "NaN"))

	print(tabulate(cls_table[:max_class_display], headers=cls_headers, tablefmt="github"))
	if len(cls_table) > max_class_display:
		print(f"... ({len(cls_table) - max_class_display} more classes hidden)")

	print()
 
 
def match_objects_to_actions(obj_masks, act_masks, num_objs, action_present, threshold=0.0):
	"""
	Match per-frame object masks to the closest action masks (among valid actions only).

	Args:
		obj_masks:      [B, N, T, h, w] — object binary masks
		act_masks:      [B, C, T, H, W] — predicted action masks
		num_objs:       [B, T] — number of valid objects in each frame
		action_present: [B, C] — 0/1 indicating actions present
		threshold:      float — optional IoU threshold for valid assignment

	Returns:
		assigned_masks: [B, C, T, H, W] — aggregated action masks
	"""
	B, N, T, h, w = obj_masks.shape
	_, C, _, H, W = act_masks.shape
	act_masks = act_masks > 0.5
 
	device = obj_masks.device

	# Upsample object masks to match action resolution
	obj_masks_up = F.interpolate(
		obj_masks.float(), size=(T, H, W), mode='trilinear', align_corners=False
	) > 0.5  # [B, N, T, H, W]

	class_map = torch.full((B, T, H, W), fill_value=64, dtype=torch.long, device=device)  # 64 = background

	for b in range(B):
		present_actions = (action_present[b] > 0).nonzero(as_tuple=False).squeeze(1)  # [n_present]
		if present_actions.numel() == 0:
			continue

		act_mask_b = act_masks[b]  # [C, T, H, W]

		for t in range(T):
			n_obj = num_objs[b, t].item()
			if n_obj == 0:
				continue

			obj_t = obj_masks_up[b, :n_obj, t]  # [n_obj, H, W]
			act_t = act_mask_b[present_actions, t]  # [n_present, H, W]

			obj_flat = obj_t.view(n_obj, -1).float()
			act_flat = act_t.view(len(present_actions), -1).float()

			inter = torch.matmul(obj_flat, act_flat.T)
			union = obj_flat.sum(-1, keepdim=True) + act_flat.sum(-1) - inter
			ious = inter / (union + 1e-6)

			best_idx = torch.argmax(ious, dim=1)
			best_ious = ious[torch.arange(n_obj), best_idx]

			for i in range(n_obj):
				if best_ious[i] > threshold:
					act_id = present_actions[best_idx[i]].item()
					mask = obj_t[i]
					class_map[b, t][mask] = act_id  # class labels: 0 ~ C-1

	return class_map  # [B, T, H, W], values in 0 ~ C-1 or 64 for background


def vis_masks(mask, raw_image, label, logdir, batch_num, alpha=0.5, color_map=None):
	"""
	Overlay multiple binary masks on raw video frames, using different colors per class.
	
	Args:
		mask:       [B, C, T, h, w] — binary masks for each class
		raw_image:  [B, T, H, W, 3] — raw RGB video frames
		label:      [B, C]          — 0/1 indicating active classes per video
		alpha:      float           — blending strength
		color_map:  Optional[List[Tuple[int, int, int]]] — RGB color per class
		
	Returns:
		overlaid:   [B, T, H, W, 3] — video with color mask overlays
	"""
	B, C, T, h, w = mask.shape

	_, _, H, W, _ = raw_image.shape
	device = mask.device

	# Default color map (distinct RGB for up to 10+ classes)
	if color_map is None:
		color_map = [
			(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
			(255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
			(0, 255, 128), (255, 64, 64)
		]

	# === Step 1: Upsample all class masks ===
	upsampled = F.interpolate(
		mask.float(), size=(T, H, W), mode="trilinear", align_corners=False
	) > 0.5  # [B, C, T, H, W]

	# === Step 2: Overlay all masks for active classes ===
	overlaid = raw_image.clone().float()

	found_action = -1
	for b in range(B):
		for c in range(64):
			if label[b, c] == 1:
				found_action += 1
				mask_bc = upsampled[b, c]  # [T, H, W]
				match_indices = torch.where(mask_bc==1)
				color = torch.tensor(color_map[found_action], device=device).view(1, 1, 3)
				color_rgb = color.expand(T, H, W, 3)  # [T, H, W, 3]

				overlaid[b][match_indices] = (
					alpha * color_rgb[match_indices] + (1 - alpha) * overlaid[b][match_indices]
				)
	
	# gif_path = os.path.join(logdir, 'gif')
	gif_path = os.path.join('../gif/attn_obj')
	os.makedirs(gif_path, exist_ok=True)
	overlaid = overlaid.clamp(0, 255).cpu().numpy()
	frames = [Image.fromarray(np.uint8(img)) for img in overlaid[0]]
	output_gif = os.path.join(gif_path, f"{batch_num}.gif")
	frames[0].save(
				output_gif,
				save_all=True,
				append_images=frames[1:],  # Add the remaining frames
				optimize=True,
				duration=200,  # Duration per frame in milliseconds
				loop=0         # Loop forever (set loop=1 for one loop only)
				)
 
 

def multi_assign_objects_to_actions(obj_masks, act_masks, num_objs, action_present, threshold=0.0):
	"""
	Multi-assign object masks to active action masks based on IoU or distance.

	Args:
		obj_masks:      [B, N, T, h, w] — object binary masks
		act_masks:      [B, C, T, H, W] — predicted action masks
		num_objs:       [B, T] — number of valid objects in each frame
		action_present: [B, C] — 0/1 indicating actions present
		threshold:      float — IoU threshold

	Returns:
		assigned_masks: [B, C+2, T, H, W] — binary masks with class channels, 64 = unassigned, 65 = full background
	"""
	B, N, T, h, w = obj_masks.shape
	_, C, _, H, W = act_masks.shape
	device = obj_masks.device

	act_masks = act_masks > 0.5
	obj_masks_up = F.interpolate(
		obj_masks.float(), size=(T, H, W), mode='trilinear', align_corners=False
	) > 0.5  # [B, N, T, H, W]

	assigned_masks = torch.zeros((B, C, T, H, W), dtype=torch.bool, device=device)

	y_grid = torch.arange(H, device=device).view(1, H, 1).expand(1, H, W)
	x_grid = torch.arange(W, device=device).view(1, 1, W).expand(1, H, W)

	for b in range(B):
		present_actions = (action_present[b] > 0).nonzero(as_tuple=False).squeeze(1)
		if present_actions.numel() == 0:
			continue

		act_mask_b = act_masks[b]  # [C, T, H, W]
		obj_up_b = obj_masks_up[b]  # [N, T, H, W]

		for t in range(T):
			if not (act_mask_b[present_actions, t].any()):
				continue

			n_obj = num_objs[b, t].item()
			if n_obj == 0:
				continue

			obj_t = obj_up_b[:n_obj, t]  # [n_obj, H, W]
			obj_flat = obj_t.view(n_obj, -1).float()

		 # Compute centroids for objects
			obj_cy = (y_grid * obj_t).view(n_obj, -1).sum(dim=1) / (obj_flat.sum(dim=1) + 1e-6)
			obj_cx = (x_grid * obj_t).view(n_obj, -1).sum(dim=1) / (obj_flat.sum(dim=1) + 1e-6)

			for c_idx, action_id in enumerate(present_actions):
				act_mask = act_mask_b[action_id, t]  # [H, W]
				if not act_mask.any():
					continue

				act_flat = act_mask.view(-1).float()
				act_area = act_flat.sum()

				# IoU check
				act_rep = act_mask.expand(n_obj, -1, -1)
				inter = (obj_t & act_rep).view(n_obj, -1).sum(dim=1).float()
				union = (obj_t | act_rep).view(n_obj, -1).sum(dim=1).float()
				ious = inter / (union + 1e-6)

				assigned = (ious > threshold)

				# Fallback: find closest unmatched object
				if not assigned.any():
					act_cy = (y_grid * act_mask).view(-1).sum() / (act_area + 1e-6)
					act_cx = (x_grid * act_mask).view(-1).sum() / (act_area + 1e-6)

					dists = torch.sqrt((obj_cy - act_cy) ** 2 + (obj_cx - act_cx) ** 2)
					closest_idx = torch.argmin(dists)
					assigned[closest_idx] = True

				# Assign all matched object masks to action class
				for i in range(n_obj):
					if assigned[i]:
						assigned_masks[b, action_id, t] |= obj_t[i]

		# Class 64: background (not covered by any action or remainder)
		assigned_masks[b, 64] = ~assigned_masks[b, :64].any(dim=0)

	return assigned_masks.float()  # [B, C, T, H, W]


def vis_masks_ref(mask, raw_image, mask_index, logdir, batch_num, attn=None, alpha=0.5, color_map=None):
	"""
	Overlay multiple binary masks on raw video frames, using different colors per class.
	
	Args:
		mask:       [B, C, T, h, w] — binary masks for each class
		raw_image:  [B, T, H, W, 3] — raw RGB video frames
		label:      [B, C]          — 0/1 indicating active classes per video
		alpha:      float           — blending strength
		color_map:  Optional[List[Tuple[int, int, int]]] — RGB color per class
		
	Returns:
		overlaid:   [B, T, H, W, 3] — video with color mask overlays
	"""
	B, T, h, w = mask.shape

	_, _, H, W, _ = raw_image.shape
	device = mask.device

	# Default color map (distinct RGB for up to 10+ classes)
	if color_map is None:
		color_map = [
			(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
			(255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
			(0, 255, 128), (255, 64, 64)
		]

	# === Step 1: Upsample all class masks ===
	upsampled = F.interpolate(
		mask.float(), size=(H, W), mode="bilinear", align_corners=False
	)  # [B, T, H, W]
	upsampled = torch.sigmoid(upsampled)  # [B, T, H, W]
	upsampled = (upsampled > 0.5).float()  # Binarize to [0, 1]
 
	# === Step 2: Overlay all masks for active classes ===
	overlaid = raw_image.clone().float()
  
	# === Step 3: Add attention overlay ===
	if attn is not None:
		attn = attn.to(overlaid.device)
		attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)

		attn_thresh = 0.2  # << Set your attention threshold here
		cmap = cm.get_cmap('jet')

		for b in range(B):
			for t in range(T):
				attn_img = attn[b, t].cpu().numpy()

				# Normalize attention to [0, 1]
				attn_min = attn_img.min()
				attn_max = attn_img.max()
				if attn_max - attn_min > 1e-5:
					attn_img = (attn_img - attn_min) / (attn_max - attn_min)
				else:
					attn_img = np.zeros_like(attn_img)

				# Get heatmap colors
				attn_rgb = cmap(attn_img)[..., :3] * 255  # (H, W, 3)
				attn_rgb = torch.from_numpy(attn_rgb).to(overlaid.device).float()  # [H, W, 3]

				# Create attention mask (1 where attn >= threshold, 0 elsewhere)
				alpha_mask = (attn_img >= attn_thresh).astype(np.float32)  # (H, W)
				alpha_mask = torch.from_numpy(alpha_mask).to(overlaid.device).unsqueeze(-1)  # [H, W, 1]

				# Blend only the attended region
				overlaid[b, t] = overlaid[b, t] * (1 - alpha * alpha_mask) + attn_rgb * (alpha * alpha_mask)
	
	for b in range(B):
		mask_bc = upsampled[b]  # [T, H, W]
		match_indices = torch.where(mask_bc==1)
		color = torch.tensor(color_map[0], device=device).view(1, 1, 3)
		color_rgb = color.expand(T, H, W, 3)  # [T, H, W, 3]

		overlaid[b][match_indices] = (
			alpha * color_rgb[match_indices] + (1 - alpha) * overlaid[b][match_indices]
		)
	
	gif_path = os.path.join(logdir, 'gif/ori_rgb')
	# gif_path = os.path.join('../gif/attn_obj')
	os.makedirs(gif_path, exist_ok=True)
	overlaid = overlaid.clamp(0, 255).cpu().numpy()
	frames = [Image.fromarray(np.uint8(img)) for img in overlaid[0]]
	output_gif = os.path.join(gif_path, f"{batch_num}_{actor_table[mask_index]}.gif")
	frames[0].save(
				output_gif,
				save_all=True,
				append_images=frames[1:],  # Add the remaining frames
				optimize=True,
				duration=200,  # Duration per frame in milliseconds
				loop=0         # Loop forever (set loop=1 for one loop only)
				)
 
 
def assign_objects_to_single_class(obj_mask, gt_mask, obj_num, iou_threshold=0.1):
	"""
	Assign objects to a single action class per frame to maximize IoU.

	Args:
		obj_mask:   [B, N, T, H, W] - binary object masks.
		gt_mask:    [B, T, H, W]    - binary ground truth mask for the target class.
		obj_num:    [B, T]          - number of valid objects per frame.
		iou_threshold: float        - minimum IoU to assign object to the class.

	Returns:
		pred_mask: [B, T, H, W] - predicted mask for the target class.
	"""
	B, N, T, H, W = obj_mask.shape
	device = obj_mask.device
	obj_mask = (obj_mask == 1)
	
	gt_mask = (gt_mask == 1)
	pred_mask = torch.zeros(B, T, H, W, device=device, dtype=torch.bool)

	def compute_iou_per_frame(objs, gt):
		# objs: [N, H, W], gt: [H, W]
		N, H, W = objs.shape
		objs = objs.view(N, -1)  # [N, HW]
		gt = gt.view(-1)         # [HW]
		intersection = (objs & gt).float().sum(dim=1)     # [N]
		union = (objs | gt).float().sum(dim=1)            # [N]
		iou = intersection / (union + 1e-6)                # [N]
		return iou

	for b in range(B):
		for t in range(T):
			valid_n = obj_num[b, t].item()
			if valid_n == 0:
				continue

			obj_masks = obj_mask[b, :valid_n, t]      # [N, H, W]
			gt = gt_mask[b, t]                        # [H, W]
			ious = compute_iou_per_frame(obj_masks, gt)  # [N]

			for n in range(valid_n):
				if ious[n] >= iou_threshold:
					pred_mask[b, t] |= obj_masks[n]

	return pred_mask

def get_upper_bound_mask(obj_mask, gt_mask, obj_num):
	"""
	Compute the upper bound mask by intersecting ground truth mask and the union of object masks.

	Args:
		obj_mask:  [B, N, T, H, W] - binary object masks
		gt_mask:   [B, T, H, W]    - ground truth mask for one action class
		obj_num:   [B, T]          - number of valid objects per frame

	Returns:
		upper_bound_mask: [B, T, H, W] - intersection of gt with union of object masks
	"""
	B, N, T, H, W = obj_mask.shape
	device = obj_mask.device
	upper_bound_mask = torch.zeros(B, T, H, W, device=device, dtype=torch.bool)

	for b in range(B):
		for t in range(T):
			valid_n = obj_num[b, t].item()
			if valid_n == 0:
				continue
			union_mask = obj_mask[b, :valid_n, t].any(dim=0)  # [H, W]
			upper_bound_mask[b, t] = gt_mask[b, t].bool() & union_mask

	return upper_bound_mask


def get_mask_attn(obj_mask, attn, query_onehot, ori_obj_mask, cls_present, obj_num):
	"""
	Args:
		obj_mask:      [B, N, T, H, W]
		attn:          [B, C, T, H, W]
		query_onehot:  [B, C]
		ori_obj_mask:  [B, N, T, H', W']
		cls_present:   [B, C]
		obj_num:       [B, T] — number of valid objects per frame
		threshold:     scalar, minimum normalized attention to assign
	Returns:
		pred_mask:     [B, 16, 32, 96] — predicted mask for the target class
	"""
	B, C, T, H, W = attn.shape
	N = obj_mask.shape[1]
	cls_count = cls_present.sum(dim=1)
	device = attn.device

	# Multiply attention with object masks
	attn_exp = attn.unsqueeze(2)         # [B, C, 1, T, H, W]
	obj_mask_exp = obj_mask.unsqueeze(1) # [B, 1, N, T, H, W]
	sim_map = attn_exp * obj_mask_exp    # [B, C, N, T, H, W]

	# Mean similarity per class per object per frame
	sim_score = sim_map.view(B, C, N, T, -1).max(dim=-1).values  # [B, C, N, T]

	# Mask absent classes (not in query label)
	cls_present_mask = cls_present.unsqueeze(2).unsqueeze(-1)  # [B, C, 1, 1]
	sim_score = sim_score.masked_fill(cls_present_mask == 0, float('-inf'))

	threshold = 0.001  # or any other small positive value
	# Mask low similarity scores
	sim_score_masked = sim_score.clone()  # [B, C, N, T]
	sim_score_masked[sim_score_masked < threshold] = float('-inf')

	# Apply softmax (ignoring low values)
	sim_score_norm = F.softmax(sim_score_masked, dim=1)

	# Handle NaNs: wherever softmax returns NaN (i.e., all -inf), replace with zeros
	sim_score_norm = torch.where(
		sim_score_norm.isnan(),
		torch.zeros_like(sim_score_norm),
		sim_score_norm
	)

	# Assignment and score thresholding
	assigned_class = sim_score_norm.argmax(dim=1)  # [B, N, T]
	max_score = sim_score_norm.max(dim=1).values   # [B, N, T]

	# Final segmentation map
	_, _, T, H_, W_ = ori_obj_mask.shape
	seg_mask_highres = torch.zeros(B, C, T, H_, W_, device=device)

	for b in range(B):
		assign_count = torch.zeros(C, T, device=device)
		threshold = 1/cls_count[b]
		for t in range(T):
			valid_n = obj_num[b, t].item()
			for n in range(valid_n):
				if max_score[b, n, t] < threshold:
					continue
				c = assigned_class[b, n, t].item()
				seg_mask_highres[b, c, t] += ori_obj_mask[b, n, t]
				assign_count[c, t] += 1

	# # Select predicted mask for the queried class
	pred_mask = torch.einsum('bc,bcthw->bthw', query_onehot.float(), seg_mask_highres)
	pred_mask = pred_mask.unsqueeze(1).float()
	# pred_mask = seg_mask_highres
 
	# Upsample to final resolution
	pred_mask = F.interpolate(pred_mask, size=(16, 32, 96), mode='trilinear', align_corners=False)
	pred_mask = pred_mask.squeeze(1)

	return pred_mask


def vis_attn_dif(attn_diff, rgb_video, mask_index, logdir, batch_num):
	"""
	Overlay attention difference map onto RGB video frames and save as GIF.

	Args:
		attn_diff:    [B, T, H, W] attention map difference
		rgb_video:    [B, T, 3, H, W] tensor with pixel values in [0, 1] or [0, 255]
		mask_index:   int, index to label the gif
		logdir:       str, directory to store the gif
		batch_num:    int, for naming
	"""
	attn_diff = attn_diff[0]

	# Prepare RGB video
	rgb_video = rgb_video[0]  # [T, 3, H, W]
	if rgb_video.max() <= 1.0:
		rgb_video = rgb_video * 255.0
	rgb_video = rgb_video.cpu().numpy()  # [T, H, W, 3]

	frames = []
	colormap = cm.get_cmap('jet')

	for t in range(attn_diff.shape[0]):
		# Get attention heatmap
		heat = colormap(attn_diff[t].cpu().numpy())[..., :3]  # [H, W, 3], RGB only
		heat = (heat * 255).astype(np.uint8)

		# Blend with RGB frame
		rgb = rgb_video[t]
		rgb_img = Image.fromarray(rgb)
		heat_img = Image.fromarray(heat).resize(rgb_img.size, Image.BILINEAR)

		# Blend heatmap with original frame
		blended = Image.blend(rgb_img.convert("RGB"), heat_img.convert("RGB"), alpha=0.5)
		frames.append(blended)

	# Save as GIF
	gif_path = os.path.join(logdir, 'attn_dif')
	os.makedirs(gif_path, exist_ok=True)
	output_gif = os.path.join(gif_path, f"{batch_num}_{actor_table[mask_index]}.gif")

	frames[0].save(
		output_gif,
		save_all=True,
		append_images=frames[1:],
		optimize=True,
		duration=200,
		loop=0
	)
 
def get_attn_dif(attn_1, attn_2):
	# attn_1, attn_2: [B, T, H, W]
	attn_diff = torch.clamp(attn_1 - attn_2, min=0)  # [B, T, H, W]

	# Min and max per sample
	B = attn_diff.shape[0]
	attn_diff_flat = attn_diff.view(B, -1)  # [B, T*H*W]
	min_vals = attn_diff_flat.min(dim=1, keepdim=True)[0]  # [B, 1]
	max_vals = attn_diff_flat.max(dim=1, keepdim=True)[0]  # [B, 1]

	# Normalize per sample
	attn_diff_norm = (attn_diff_flat - min_vals) / (max_vals - min_vals + 1e-6)  # [B, T*H*W]
	attn_diff = attn_diff_norm.view_as(attn_diff)  # [B, T, H, W]
	return attn_diff

def weighted_dbscan(attn_thresh, eps=3, min_samples=10):
    """
    Args:
        attn_thresh: [H, W] attention map (float values, not binary)
        eps: neighborhood radius for DBSCAN
        min_samples: minimum samples for a cluster
    Returns:
        labeled: [H, W] array with cluster labels (0 = background)
        num_features: number of clusters
    """
    H, W = attn_thresh.shape
    mask_bin = attn_thresh > 0  # same as before
    coords = np.argwhere(mask_bin)  # [N, 2]
    if coords.shape[0] == 0:
        return np.zeros_like(attn_thresh, dtype=int), 0

    # Add attention values as an extra dimension
    values = attn_thresh[mask_bin][:, None]  # [N, 1]
    coords_with_val = np.hstack([coords, values])

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_with_val)
    labels = db.labels_  # -1 = noise

    # Map back to full image
    labeled = np.zeros((H, W), dtype=int)
    valid_coords = tuple(coords.T)
    # Shift labels to be >= 1 (like scipy.ndimage.label)
    cluster_ids = np.unique(labels[labels >= 0])
    label_map = {cid: i+1 for i, cid in enumerate(cluster_ids)}
    mapped_labels = np.array([label_map.get(l, 0) for l in labels])
    labeled[valid_coords] = mapped_labels

    num_features = len(cluster_ids)
    return labeled, num_features

def weighted_label(attn_map, threshold=0, value_eps=0.1, connectivity=1):
    """
    Weighted connected component labeling.

    Args:
        attn_map: [H, W] attention map (float values)
        threshold: binarization threshold (only > threshold are considered foreground)
        value_eps: maximum allowed difference in attention value between connected pixels
        connectivity: 1 (4-connectivity in 2D), 2 (8-connectivity in 2D)

    Returns:
        labeled: [H, W] array with cluster labels (0 = background)
        num_features: number of clusters
    """
    H, W = attn_map.shape
    mask = attn_map > threshold
    labeled = np.zeros((H, W), dtype=int)

    # Connectivity kernel (like ndimage.label)
    structure = generate_binary_structure(2, connectivity)

    cluster_id = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and labeled[i, j] == 0:
                cluster_id += 1
                # BFS flood fill
                queue = [(i, j)]
                labeled[i, j] = cluster_id
                base_val = attn_map[i, j]

                while queue:
                    x, y = queue.pop()
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if (dx, dy) == (0, 0): 
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if mask[nx, ny] and labeled[nx, ny] == 0:
                                    # check attention value similarity
                                    if abs(attn_map[nx, ny] - attn_map[x, y]) <= value_eps:
                                        labeled[nx, ny] = cluster_id
                                        queue.append((nx, ny))

    return labeled, cluster_id

def refine_pseudo_mask(attn_map, obj_masks, frames_rgb=None, vis_dir=None, batch_num=None, mask_index=None, max_clusters=2, distance_thresh=100, mode='gravity'):
	"""
	attn_map: [B, T, H, W] - attention maps
	obj_masks: [B, N, T, H, W] - binary object masks
	frames_rgb: [B, T, H, W, 3] - input RGB frames (0-255)
	"""
	B, T, H, W = attn_map.shape
	N = obj_masks.shape[1]
	pseudo_masks = torch.zeros_like(attn_map)

	if vis_dir:
		gif_dir = os.path.join(vis_dir, 'refine')
		os.makedirs(gif_dir, exist_ok=True)

	for b in range(B):
		if mask_index[b] < 12 or (mask_index[b]>=24 and mask_index[b]<36) or (mask_index[b]>=48 and mask_index[b]<56):
			max_clusters = 1
		else:
			max_clusters = 3
   
		if mask_index[b] < 24:
			mode = 'gravity'
		else:
			mode = 'distance'
   
		frame_imgs = [] if vis_dir else None

		for t in range(T):
			attn_t = attn_map[b, t].detach().cpu().numpy()

			# === Threshold and Normalize ===
			attn_thresh = np.where(attn_t > 0.2, attn_t, 0)
			# attn_norm = (attn_thresh - attn_thresh.min()) / (attn_thresh.max() - attn_thresh.min() + 1e-6)

			# === Connected Component Clustering ===
			mask_bin = attn_thresh > 0
			labeled, num_features = label(mask_bin)
			# labeled, num_features = weighted_label(attn_thresh)
			# labeled, num_features = weighted_dbscan(attn_thresh)

			cluster_ids = np.unique(labeled)[1:]  # exclude 0

			if len(cluster_ids) > max_clusters:
				cluster_sizes = [(labeled == cid).sum() for cid in cluster_ids]
				sorted_ids = [cid for _, cid in sorted(zip(cluster_sizes, cluster_ids), reverse=True)]
				cluster_ids = sorted_ids[:max_clusters]

			matched_mask = np.zeros((H, W), dtype=np.uint8)

			if vis_dir:
				rgb_frame = (frames_rgb[b, t].detach().cpu().numpy()).astype(np.uint8)
				base_img = Image.fromarray(rgb_frame).convert("RGBA")
				overlay = base_img.copy()
				draw = ImageDraw.Draw(overlay)

			# === Assign random colors for clusters ===
			if vis_dir:
				cluster_colors = {}
				for i, cid in enumerate(cluster_ids):
					hue = i / max(len(cluster_ids), 1)
					red, green, blue = hsv_to_rgb([[hue, 1, 1]])[0]
					cluster_colors[cid] = tuple(int(255 * x) for x in (red, green, blue))

			for cid in cluster_ids:
				cluster_mask = (labeled == cid)
				if cluster_mask.sum() == 0:
					continue

				cy, cx = center_of_mass(cluster_mask)

				# Visualize cluster mask
				if vis_dir:
					cluster_img = np.zeros((H, W, 4), dtype=np.uint8)
					cluster_img[..., :3] = cluster_colors[cid]
					cluster_img[..., 3] = (cluster_mask * 180).astype(np.uint8)  # transparency
					cluster_overlay = Image.fromarray(cluster_img)
					overlay = Image.alpha_composite(overlay, cluster_overlay)
					draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill='red', outline='black', width=1)
	 
				if mode == 'gravity':
					best_score = -1
					best_obj = -1
					best_center = None
					for n in range(N):
						obj_mask = obj_masks[b, n, t].detach().cpu().numpy()
						if obj_mask.sum() == 0:
							continue
						yx = np.argwhere(obj_mask)
						ocy, ocx = yx.mean(axis=0)
						dist = np.linalg.norm([cy - ocy, cx - ocx])
						area = obj_mask.sum()
						gravity = area / (dist ** 2 + 1e-6)

						if gravity > best_score:
							best_score = gravity
							best_obj = n
							best_center = (ocx, ocy)

					if best_obj != -1:
						matched_mask |= obj_masks[b, best_obj, t].detach().cpu().numpy().astype(np.uint8)

						if vis_dir and best_center is not None:
							draw.ellipse((best_center[0]-4, best_center[1]-4, best_center[0]+4, best_center[1]+4), fill='blue')
							draw.line([(cx, cy), best_center], fill='blue', width=2)

				else:  # distance-based matching
					min_dist = float('inf')
					best_obj = -1
					best_center = None
					for n in range(N):
						obj_mask = obj_masks[b, n, t].detach().cpu().numpy()
						if obj_mask.sum() == 0:
							continue
						yx = np.argwhere(obj_mask)
						ocy, ocx = yx.mean(axis=0)
						dist = np.linalg.norm([cy - ocy, cx - ocx])
						if dist < min_dist:
							min_dist = dist
							best_obj = n
							best_center = (ocx, ocy)

					if best_obj != -1 and min_dist < distance_thresh:
						matched_mask |= obj_masks[b, best_obj, t].detach().cpu().numpy().astype(np.uint8)

						if vis_dir and best_center is not None:
							draw.ellipse((best_center[0]-4, best_center[1]-4, best_center[0]+4, best_center[1]+4), fill='blue')
							draw.line([(cx, cy), best_center], fill='blue', width=2)

			pseudo_masks[b, t] = torch.from_numpy(matched_mask)

			# === Draw matched object contours ===
			if vis_dir:
				for n in range(N):
					mask = obj_masks[b, n, t].detach().cpu().numpy()
					if mask.sum() == 0:
						continue
					contours = find_contours(mask, 0.5)
					for contour in contours:
						contour = [(int(x[1]), int(x[0])) for x in contour]
						draw.line(contour, fill='green', width=2)

				frame_imgs.append(overlay.convert("RGB"))

		if vis_dir and frame_imgs:
			gif_path = os.path.join(gif_dir, f"{batch_num}_{actor_table[mask_index]}.gif")
			frame_imgs[0].save(
				gif_path,
				save_all=True,
				append_images=frame_imgs[1:],
				optimize=True,
				duration=200,
				loop=0
			)

	return pseudo_masks  # [B, T, H, W]


def refine_pseudo_mask_multi(
	attn_maps: torch.Tensor,        # [B, C, T, H, W]
	obj_masks: torch.Tensor,        # [B, N, T, H, W]
	valid_classes: torch.Tensor,    # [B, C] — binary class presence
	distance_thresh: float = 100.0
) -> torch.Tensor:
	"""
	Refine pseudo masks per class based on attention clusters and object masks.

	Args:
		attn_maps:      [B, C, T, H, W] — attention maps per class
		obj_masks:      [B, N, T, H, W] — binary object masks
		valid_classes:  [B, C] — binary indicators for present classes
		distance_thresh: float — threshold for distance-based matching

	Returns:
		pseudo_masks:   [B, C, T, H, W] — refined masks
	"""
	B, C, T, H, W = attn_maps.shape
	N = obj_masks.shape[1]
	device = attn_maps.device

	pseudo_masks = torch.zeros((B, C, T, H, W), dtype=torch.uint8, device=device)

	for b in range(B):
		for cls in range(C):
			if valid_classes[b, cls] == 0:
				continue

			# Class-specific clustering rules
			if cls < 12 or (24 <= cls < 36) or (48 <= cls < 56):
				max_clusters = 1
			else:
				max_clusters = 3

			mode = 'gravity' if cls < 24 else 'distance'

			for t in range(T):
				attn_t = attn_maps[b, cls, t].detach().cpu().numpy()
				attn_thresh = np.where(attn_t > 0.2, attn_t, 0)

				mask_bin = attn_thresh > 0
				labeled, num_features = label(mask_bin)
				cluster_ids = np.unique(labeled)[1:]  # exclude background

				if len(cluster_ids) > max_clusters:
					cluster_sizes = [(labeled == cid).sum() for cid in cluster_ids]
					cluster_ids = [cid for _, cid in sorted(zip(cluster_sizes, cluster_ids), reverse=True)[:max_clusters]]

				matched_mask = np.zeros((H, W), dtype=np.uint8)

				for cid in cluster_ids:
					cluster_mask = (labeled == cid)
					if cluster_mask.sum() == 0:
						continue
					cy, cx = center_of_mass(cluster_mask)

					best_score = -1
					best_obj = -1

					for n in range(N):
						obj_mask = obj_masks[b, n, t].detach().cpu().numpy()
						if obj_mask.sum() == 0:
							continue

						yx = np.argwhere(obj_mask)
						ocy, ocx = yx.mean(axis=0)
						dist = np.linalg.norm([cy - ocy, cx - ocx])

						if mode == 'gravity':
							area = obj_mask.sum()
							gravity = area / (dist**2 + 1e-6)
							if gravity > best_score:
								best_score = gravity
								best_obj = n
						else:  # distance mode
							if best_score == -1 or dist < best_score:
								best_score = dist
								best_obj = n

					if best_obj != -1 and (mode == 'gravity' or best_score < distance_thresh):
						matched_mask |= obj_masks[b, best_obj, t].detach().cpu().numpy().astype(np.uint8)

				pseudo_masks[b, cls, t] = torch.from_numpy(matched_mask).to(device)

	return pseudo_masks


def plot_recognition_mask_quality(S_b, S_a, IOU, pseudo_IOU=None, save_path='../results/recog_mask_diff_val.png', bins=25):
	"""
	Heatmap of recognition score change vs. pseudo mask quality (ΔIoU),
	with bin density-aware visualization and per-type ΔIoU stats.

	Args:
		S_b (list of tensors): Recognition scores before inpaint.
		S_a (list of tensors): Recognition scores after inpaint.
		IOU (list of tensors): Ground-truth IoU.
		pseudo_IOU (list of tensors): Pseudo mask IoU (optional).
		save_path (str): Output path.
		bins (int): Number of bins for heatmap.
	"""
	def tensor_list_to_numpy(lst):
		return torch.cat([x.detach().cpu().flatten() for x in lst]).numpy()

	S_b = tensor_list_to_numpy(S_b)
	S_a = tensor_list_to_numpy(S_a)
	IOU = tensor_list_to_numpy(IOU)

	if pseudo_IOU is not None:
		pseudo_IOU = tensor_list_to_numpy(pseudo_IOU)
		IOU_diff = pseudo_IOU - IOU
	else:
		IOU_diff = IOU.copy()

	# 2D binning for ΔIoU (heatmap) and count (density)
	statistic, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(
		S_b, S_a, IOU_diff, statistic='mean', bins=bins
	)
	counts, _, _, _ = scipy.stats.binned_statistic_2d(
		S_b, S_a, None, statistic='count', bins=[xedges, yedges]
	)

	# Log-norm count to [0, 1] to use as alpha
	alpha = np.log1p(counts)
	alpha = alpha / np.nanmax(alpha)

	# Plot
	plt.figure(figsize=(7, 6))
	if len(pseudo_IOU) == 0:
		vmax = 1.0
		vmin = 0.0
	else:
		vmax = np.nanmax(np.abs(statistic))
		vmin = -vmax

	for i in range(bins):
		for j in range(bins):
			color = plt.cm.seismic((statistic[i, j] - vmin) / (vmax - vmin)) if not np.isnan(statistic[i, j]) else (1, 1, 1, 0)
			color = list(color)
			color[-1] = alpha[i, j]  # Use normalized count as alpha
			plt.gca().add_patch(plt.Rectangle(
				(xedges[i], yedges[j]),
				xedges[i + 1] - xedges[i],
				yedges[j + 1] - yedges[j],
				color=color
			))

	plt.axhline(0.5, color='black', linestyle=':', linewidth=1.2)
	plt.axvline(0.5, color='black', linestyle=':', linewidth=1.2)
	plt.xlabel('Recognition Score Before (S_b)')
	plt.ylabel('Recognition Score After (S_a)')
	plt.title('Recognition vs. ΔIoU (pseudo - GT)')

	# Custom colorbar
	import matplotlib.colors as mcolors
	import matplotlib.cm as cm
	sm = plt.cm.ScalarMappable(cmap='seismic', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
	cbar = plt.colorbar(sm, label='Δ IoU (pseudo - GT)')
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()

	# Per-type stats
	S_b_tensor = torch.tensor(S_b)
	S_a_tensor = torch.tensor(S_a)
	IOU_diff_tensor = torch.tensor(IOU_diff)

	def type_stats(mask, label):
		if not mask.any(): return {f'{label}': 'N/A'}

		subset = IOU_diff_tensor[mask]
		return {
			f'{label} (Avg ΔIoU)': round(subset.mean().item(), 4),
			f'{label} (Count +)': round(100*((subset > 0).sum().item())/int(mask.sum().item()), 2),
			f'{label} (Count -)': round(100*((subset < 0).sum().item())/int(mask.sum().item()), 2),
		}

	type1 = (S_b_tensor >= 0.5) & (S_a_tensor < 0.5)
	type2 = (S_b_tensor >= 0.5) & (S_a_tensor >= 0.5)
	type3 = (S_b_tensor < 0.5) & (S_a_tensor >= 0.5)
	type4 = (S_b_tensor < 0.5) & (S_a_tensor < 0.5)

	summary = {}
	summary.update(type_stats(type1, 'Type 1: ✓→✗'))
	summary.update(type_stats(type2, 'Type 2: ✓→✓'))
	summary.update(type_stats(type3, 'Type 3: ✗→✓'))
	summary.update(type_stats(type4, 'Type 4: ✗→✗'))

	print("\n[ΔIoU Summary by Recognition Transition Type]")
	for k, v in summary.items():
		print(f"{k}: {v}")
  


class GradCAM:
	def __init__(self, model, target_layer):
		self.model = model
		self.target_layer = target_layer
		self.gradients = None
		self.activations = None
		self._register_hooks()

	def _register_hooks(self):
		def forward_hook(module, input, output):
			self.activations = output.detach()

		def backward_hook(module, grad_input, grad_output):
			self.gradients = grad_output[0].detach()

		self.target_layer.register_forward_hook(forward_hook)
		self.target_layer.register_full_backward_hook(backward_hook)

	def get_cam(self, input_tensor):
		self.model.to(input_tensor.device)
		input_tensor = input_tensor.requires_grad_(True)
		self.model.eval()
		self.model.zero_grad()

		with torch.enable_grad():
			out = self.model(input_tensor)
			if isinstance(out, tuple):
				out = out[1]

		B, C = out.shape
		cam_per_class = []

		for class_idx in range(C):
			self.model.zero_grad()
			score = out[:, class_idx].sum()
			score.backward(retain_graph=True)

			weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
			cam = (weights * self.activations).sum(dim=1)  # [B, T, H, W]
			cam = F.relu(cam)

			# Normalize
			cam_flat = cam.view(B, -1)
			cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
			cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
			cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

			cam_per_class.append(cam)

		return torch.stack(cam_per_class, dim=1)  # [B, C, T, H, W]

def vis_masks_ref_new(mask, raw_image, mask_index, logdir, batch_num, attn=None, alpha=0.5):
	B, T, h, w = mask.shape
	_, _, H, W, _ = raw_image.shape
	device = mask.device

	# === Step 1: Upsample and binarize masks ===
	upsampled = F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False)
	upsampled = torch.sigmoid(upsampled)
	upsampled = (upsampled > 0.5).float()  # [B, T, H, W]

	# === Step 2: Initialize black backgrounds ===
	black_background = torch.zeros_like(raw_image).float()
	mask_on_black = black_background.clone()
	attn_on_black = black_background.clone()
	overlay_rgb = raw_image.clone().float()  # for RGB+mask+attn overlay
	ori_rgb = raw_image.clone().float()

	# === Step 3: Attention overlay on black ===
	if attn is not None:
		attn = attn.to(device)
		attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
		attn_thresh = 0.2
		cmap = cm.get_cmap('jet')

		for b in range(B):
			for t in range(T):
				attn_img = attn[b, t].cpu().numpy()

				# Do NOT normalize, just use raw values
				cmap = cm.get_cmap('jet')
				attn_rgb = cmap(attn_img)[..., :3] * 255  # [H, W, 3]
				attn_rgb = torch.from_numpy(attn_rgb).to(device).float()

				# Threshold directly at 0.2
				alpha_mask = (attn_img >= 0.2).astype(np.float32)
				alpha_mask = torch.from_numpy(alpha_mask).to(device).unsqueeze(-1)  # [H, W, 1]

				# Apply attention map only on thresholded region over black
				attn_on_black[b, t] = attn_rgb * alpha_mask

				# === Overlay attention (after mask) onto RGB ===
				overlay_rgb[b, t] = overlay_rgb[b, t] * (1 - alpha * alpha_mask) + attn_rgb * (alpha * alpha_mask)

	# === Step 4: Mask overlay (red) ===
	red = torch.tensor([255, 0, 0], device=device).view(1, 1, 3).float()

	for b in range(B):
		for t in range(T):
			mask_bc = upsampled[b, t]  # [H, W]
			mask_bool = mask_bc == 1
			# overlay mask on black
			mask_on_black[b, t][mask_bool] = red
			# overlay mask on RGB (before attention)
			overlay_rgb[b, t][mask_bool] = (1 - alpha) * overlay_rgb[b, t][mask_bool] + alpha * red
 
	# # === Step 4: Mask overlay (keep original RGB inside mask) ===
	# for b in range(B):
	# 	for t in range(T):
	# 		mask_bc = upsampled[b, t]  # [H, W]
	# 		mask_bool = mask_bc == 1

	# 		# show original RGB inside mask region (on black background)
	# 		mask_on_black[b, t][mask_bool] = ori_rgb[b, t][mask_bool]

	# 		# overlay mask onto RGB before attention (keep original RGB in mask region)
	# 		overlay_rgb[b, t][mask_bool] = (1 - alpha) * overlay_rgb[b, t][mask_bool] + alpha * ori_rgb[b, t][mask_bool]

	# === Step 5: Save GIFs ===
	def save_gif(frames, path):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		frames = [Image.fromarray(np.uint8(f)) for f in frames]
		frames[0].save(
			path,
			save_all=True,
			append_images=frames[1:],
			optimize=True,
			duration=200,
			loop=0
		)

	# Mask
	mask_path = os.path.join(logdir, 'gif/val_mask_g', f"{batch_num}_{actor_table[mask_index]}.gif")
	save_gif(mask_on_black[0].clamp(0, 255).cpu().numpy(), mask_path)

	# Attention only
	if attn is not None:
		attn_path = os.path.join(logdir, 'gif/val_attn', f"{batch_num}_{actor_table[mask_index]}.gif")
		save_gif(attn_on_black[0].clamp(0, 255).cpu().numpy(), attn_path)

		# RGB + Mask + Attention overlay
		overlay_path = os.path.join(logdir, 'gif/val_overlay_g', f"{batch_num}_{actor_table[mask_index]}.gif")
		save_gif(overlay_rgb[0].clamp(0, 255).cpu().numpy(), overlay_path)
 

def otsu_threshold_batch(attn_maps: torch.Tensor) -> torch.Tensor:
	"""
	Apply Otsu thresholding to a batch of attention maps.
	
	Args:
		attn_maps (torch.Tensor): (B, C, T, H, W) attention maps.
	
	Returns:
		torch.Tensor: (B, C, T, H, W) binary masks (0/1).
	"""
	attn_maps_np = attn_maps.detach().cpu().numpy()
	B, C, T, H, W = attn_maps_np.shape
	masks = np.zeros((B, C, T, H, W), dtype=np.uint8)

	for b in range(B):
		for c in range(C):
			for t in range(T):
				attn = attn_maps_np[b, c, t]
				# normalize to [0, 255]
				norm = ((attn - attn.min()) / (attn.max() - attn.min() + 1e-8) * 255).astype(np.uint8)
				_, mask = cv2.threshold(norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				masks[b, c, t] = mask

	return torch.from_numpy(masks).to(attn_maps.device)


def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
	"""
	Compute IoU between two binary masks (H, W).
	"""
	inter = (mask1 & mask2).sum().float()
	union = (mask1 | mask2).sum().float()
	return (inter / (union + 1e-6)).item()


def generate_pseudo_masks(attn_maps: torch.Tensor, obj_masks: torch.Tensor, mask_index: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
	"""
	Generate pseudo masks by selecting object masks with IoU >= alpha with Otsu-thresholded attention.
	
	Args:
		attn_maps (torch.Tensor): (B, C, T, H, W) attention maps.
		obj_masks (torch.Tensor): (B, T, N, H, W) object masks (binary 0/1).
		alpha (float): IoU threshold.
	
	Returns:
		torch.Tensor: (B, C, T, H, W) pseudo masks (union of matched object masks).
	"""
	bin_attn = otsu_threshold_batch(attn_maps)  # (B, C, T, H, W)
	B, C, T, H, W = bin_attn.shape
	bin_attn_query = []
	for b in range(B):
		bin_attn_query.append(bin_attn[b, mask_index[b]])

	bin_attn_query = torch.stack(bin_attn_query, dim=0)
	N = obj_masks.size(2)

	pseudo_masks = torch.zeros((B, T, H, W), dtype=torch.uint8, device=attn_maps.device)

	for b in range(B):
		for t in range(T):
			attn_mask = bin_attn_query[b, t]
			union_mask = torch.zeros((H, W), dtype=torch.uint8, device=attn_maps.device)
			for n in range(N):
				obj_mask = obj_masks[b, t, n]
				iou = compute_iou(attn_mask, obj_mask)
				if iou >= alpha:
					union_mask |= obj_mask.byte()
			pseudo_masks[b, t] = union_mask

	return pseudo_masks