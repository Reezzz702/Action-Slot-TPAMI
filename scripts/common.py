from typing import Optional
"""
common.py
---------
Shared constants, utilities, the LightningModule Trainer class, and model
builders used by both train.py and eval.py.
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import average_precision_score

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from generate_model import generate_model
from action_slot_utils import *
from segmentation import *
from localization_models import *
from loss import *
from inference_propainter import ProPainter


# ====================================================================== #
#  Constants                                                               #
# ====================================================================== #

# fmt: off
actor_table = [
	'c:z1-z2',  'c:z1-z3',  'c:z1-z4',
	'c:z2-z1',  'c:z2-z3',  'c:z2-z4',
	'c:z3-z1',  'c:z3-z2',  'c:z3-z4',
	'c:z4-z1',  'c:z4-z2',  'c:z4-z3',
	'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
	'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
	'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
	'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
	'b:z1-z2',  'b:z1-z3',  'b:z1-z4',
	'b:z2-z1',  'b:z2-z3',  'b:z2-z4',
	'b:z3-z1',  'b:z3-z2',  'b:z3-z4',
	'b:z4-z1',  'b:z4-z2',  'b:z4-z3',
	'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
	'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
	'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
	'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
	'p:c1-c2',  'p:c1-c4',
	'p:c2-c1',  'p:c2-c3',
	'p:c3-c2',  'p:c3-c4',
	'p:c4-c1',  'p:c4-c3',
	'p+:c1-c2', 'p+:c1-c4',
	'p+:c2-c1', 'p+:c2-c3',
	'p+:c3-c2', 'p+:c3-c4',
	'p+:c4-c1', 'p+:c4-c3',
	'bg',
]
# fmt: on

NUM_ACTOR_CLASSES = 64

# Models that expose slot attention maps natively.
# All other models (x3d, csn, i3d) use GradCAM instead.
SLOT_ATTN_MODELS = {'action_slot', 'slot_vps'}

# GradCAM target layers, keyed by model_name.
GRADCAM_TARGET_LAYERS = {
	'x3d': lambda m: m.model.blocks[-3],
	'csn': lambda m: m.model.blocks[-2],
	'i3d': lambda m: m.model.blocks[-2],
}


# ====================================================================== #
#  Utilities                                                               #
# ====================================================================== #

def uses_slot_attention(model_name: str) -> bool:
	"""Return True if *model_name* produces attention maps natively."""
	return model_name in SLOT_ATTN_MODELS


def binary_dilation_torch(input_mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
	"""GPU binary dilation with a 3x3 cross structuring element.

	Args:
		input_mask: Boolean or float tensor of shape (B, T, H, W).
		iterations: Number of dilation passes.

	Returns:
		Dilated mask of shape (B, T, 1, H, W).
	"""
	B, T, H, W = input_mask.shape
	x = input_mask.view(B * T, 1, H, W).float()

	kernel = torch.tensor(
		[[0, 1, 0],
		 [1, 1, 1],
		 [0, 1, 0]],
		dtype=torch.float32,
		device=input_mask.device,
	).unsqueeze(0).unsqueeze(0)

	for _ in range(iterations):
		x = F.conv2d(x, kernel, padding=1)
		x = (x > 0).float()

	return x.view(B, T, H, W).unsqueeze(2)


def _gather_by_index(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
	"""Select one slot per batch item.

	Args:
		tensor: Shape (B, S, ...).
		index:  Shape (B,).

	Returns:
		Shape (B, ...).
	"""
	return torch.stack([tensor[b, index[b]] for b in range(tensor.shape[0])], dim=0)


# ====================================================================== #
#  Model builders                                                          #
# ====================================================================== #

def load_recog_model_from_training_ckpt(ckpt_path: str, args) -> nn.Module:
	"""Extract recog_model weights from a LocalizationModule Lightning checkpoint.

	During training, LocalizationModule saves all three models together inside
	one .ckpt file under keys prefixed with 'recog_model.'.  This function
	strips that prefix and loads only the recog_model weights, so the model
	can be used standalone as an attention source in eval_pseudo_mask.py.

	Args:
		ckpt_path: Path to a .ckpt file saved by train_localization.py.
		args:      Parsed args — uses args.model_name and args.backbone.

	Returns:
		The recog_model with trained weights loaded, moved to eval mode and frozen.
	"""
	ckpt = torch.load(ckpt_path, map_location='cpu')
	full_sd = ckpt['state_dict']

	prefix = 'recog_model.'
	recog_sd = {
		k[len(prefix):]: v
		for k, v in full_sd.items()
		if k.startswith(prefix)
	}

	if not recog_sd:
		raise KeyError(
			f"No keys with prefix '{prefix}' found in {ckpt_path}. "
			"Make sure this checkpoint was saved by LocalizationModule "
			"(train_localization.py)."
		)

	model = generate_model(args, 4, NUM_ACTOR_CLASSES)
	missing, unexpected = model.load_state_dict(recog_sd, strict=False)

	# slots_mu/slots_sigma may be missing in old checkpoints (see on_load_checkpoint)
	slot_params = {'slot_attention.slots_mu', 'slot_attention.slots_sigma'}
	unexpected_missing = set(missing) - slot_params
	if unexpected_missing:
		raise RuntimeError(
			f"Unexpected missing keys when loading recog_model from checkpoint: "
			f"{unexpected_missing}"
		)
	if unexpected:
		raise RuntimeError(
			f"Unexpected extra keys in recog_model state dict: {unexpected}"
		)

	model.eval()
	for p in model.parameters():
		p.requires_grad = False

	return model


def build_attn_model(args) -> nn.Module:
	"""Load the model used to produce attention maps / GradCAM.

	Controlled by args:
		args.attn_model_name  -- model architecture (e.g. 'action_slot', 'x3d')
		args.attn_backbone    -- backbone for that architecture
		args.attn_cp          -- path to its checkpoint
		args.freeze_attn      -- if True, freeze all parameters after loading

	NOTE: when freeze_attn=False, this function is NOT called — the attn_model
	is shared with recog_model directly in build_common_kwargs.
	"""
	attn_args = copy.deepcopy(args)
	attn_args.model_name = args.attn_model_name
	attn_args.backbone   = args.attn_backbone

	model = generate_model(attn_args, 4, NUM_ACTOR_CLASSES)
	model.load_state_dict(torch.load(args.attn_cp), strict=False)

	# freeze_attn=True is guaranteed by the caller when this function is used
	model.eval()
	for p in model.parameters():
		p.requires_grad = False

	return model


def build_recog_model(args) -> nn.Module:
	"""Load the recognition backbone used to generate localization features.

	Controlled by args:
		args.cp          -- checkpoint path (raw .pth or Lightning .ckpt)
		args.freeze_loc  -- if True, freeze all parameters after loading

	Handles two checkpoint formats:
		Raw weights file (.pth):   direct state_dict
		Lightning checkpoint (.ckpt): nested under 'state_dict' key, with
			keys prefixed by 'recog_model.' which are stripped automatically.
	"""
	ckpt = torch.load(args.cp, map_location='cpu')

	# Lightning .ckpt files wrap the state_dict and prefix submodule names
	if isinstance(ckpt, dict) and 'state_dict' in ckpt:
		full_sd = ckpt['state_dict']
		prefix  = 'recog_model.'
		sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
		if not sd:
			raise KeyError(
				f"No keys with prefix '{prefix}' found in {args.cp}. "
				"If this is a raw weights file, make sure it does not have a "
				"'state_dict' wrapper."
			)
	else:
		# Raw weights file — use as-is
		sd = ckpt

	model = generate_model(args, 4, NUM_ACTOR_CLASSES)

	# slots_mu / slots_sigma may be absent in old raw checkpoints or present
	# as unexpected keys in environment mismatches — handle both gracefully.
	slot_keys = {'slot_attention.slots_mu', 'slot_attention.slots_sigma'}
	missing    = slot_keys - sd.keys()
	unexpected = {k for k in slot_keys if k in sd and not hasattr(model, k.split('.')[0])}
	
	# Remove unexpected keys before loading
	# for k in unexpected:
	#     del sd[k]

	missing_non_slot = set(model.state_dict().keys()) - sd.keys() - slot_keys
	if missing_non_slot:
		import warnings
		warnings.warn(
			f"build_recog_model: {len(missing_non_slot)} non-slot keys missing "
			f"from checkpoint — loading with strict=False. "
			f"First few: {list(missing_non_slot)[:5]}",
			UserWarning,
		)
		model.load_state_dict(sd, strict=False)
	else:
		model.load_state_dict(sd, strict=False)

	if getattr(args, 'freeze_loc', False):
		model.eval()
		for p in model.parameters():
			p.requires_grad = False

	return model


def _build_decoder(args) -> nn.Module:
	"""Instantiate the localization decoder from args.decoder.

	Supported values for --decoder:
		'cross_attn'       CrossAttentionFusion  -- uses attn map as query (default)
		'cross_attn_onehot' CrossAttentionFusion_OneHot -- uses class one-hot as query,
							no attention model needed
	"""
	decoder_name = getattr(args, 'decoder', 'cross_attn')
	if decoder_name == 'cross_attn_onehot':
		return CrossAttentionFusion_OneHot()
	return CrossAttentionFusion()


def build_common_kwargs(args) -> dict:
	"""Instantiate all models and return as kwargs for LocalizationModule.

	Three models are involved:
		attn_model   -- provides attention maps (checkpoint: args.attn_cp)
		recog_model  -- recognition backbone for localization features (args.cp)
		loc_decoder  -- CrossAttentionFusion segmentation head (no pretrain)

	When freeze_attn=False, attn_model and recog_model are the SAME object.
	A single forward pass is run and its attention maps + features are both
	used, and the weights are updated together by one optimizer entry.

	When freeze_attn=True, attn_model is a separate frozen model loaded from
	args.attn_cp, while recog_model is loaded from args.cp independently.
	"""
	recog_model = build_recog_model(args)

	if getattr(args, 'freeze_attn', True):
		# Separate frozen attention model loaded from its own checkpoint
		attn_model = build_attn_model(args)
	else:
		# Shared model: attn_model IS recog_model — same object, same weights
		attn_model = recog_model

	return dict(
		args=args,
		attn_model=attn_model,
		recog_model=recog_model,
		loc_decoder=_build_decoder(args),
		num_actor_class=NUM_ACTOR_CLASSES,
	)


def build_pl_trainer(args, logdir: str, callbacks: list, logger, resume_path=None):
	"""Construct a pl.Trainer with the project-standard settings."""
	return pl.Trainer(
		logger=logger,
		max_epochs=args.epochs,
		log_every_n_steps=1,
		gpus=args.gpus,
		strategy=DDPStrategy(find_unused_parameters=False),
		check_val_every_n_epoch=args.val_every,
		callbacks=callbacks,
		resume_from_checkpoint=resume_path,
		inference_mode=False,
	)


# ====================================================================== #
#  LightningModule                                                         #
# ====================================================================== #

class LocalizationModule(pl.LightningModule):
	"""PyTorch Lightning module for the activity localization task.

	Three models are wired together:

		attn_model  -- produces attention maps used to build pseudo masks.
					   When freeze_attn=True this is a separate frozen model.
					   When freeze_attn=False this IS recog_model (same object).

		recog_model -- recognition backbone whose intermediate features feed
					   the loc_decoder. Trained jointly with loc_decoder unless
					   freeze_loc=True.

		loc_decoder -- CrossAttentionFusion segmentation head. Always trained.
	"""

	def __init__(
		self,
		args,
		attn_model:      nn.Module,
		recog_model:     nn.Module,
		loc_decoder:     nn.Module,
		num_actor_class: int,
	):
		super().__init__()
		self.args        = args
		self.attn_model  = attn_model
		self.recog_model = recog_model
		self.loc_decoder = loc_decoder

		# True when attn_model and recog_model are the same object
		self._shared_model = attn_model is recog_model

		self.criterion = SegmentationLoss(args, num_actor_class)

		attention_res = None
		if hasattr(self.recog_model, 'resolution'):
			h = self.recog_model.resolution[0] * args.bg_upsample
			w = self.recog_model.resolution[1] * args.bg_upsample
			attention_res = (h, w)
		self.action_criterion = ActionSlotLoss(args, num_actor_class, attention_res)

		self.normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		)

		self.action_loss_weight = {
			'action_slot_loss': 1.0,
			'seg_loss':         1.0,
			'ad_loss':          0.01,
			# cam_loss is only included when --cam_loss is set; weight from
			# --cam_loss_weight (default 1.0).
			'cam_loss': getattr(args, 'cam_loss_weight', 1.0),
		}
		if not getattr(self.args, 'refine', False):
			self.action_loss_weight['seg_loss'] = 1.0

		# ── GradCAM setup for non-slot attn_model ─────────────────────
		# Slot-based models (action_slot, slot_vps) expose attention maps
		# directly from their forward pass.  All other backbones (x3d, csn,
		# i3d) produce attention via GradCAM.
		#
		# GradCAM always runs on an isolated deepcopy so its backward pass
		# never contaminates the DDP training graph, regardless of whether
		# the model is frozen.  The copy is built once here and reused.
		self.gradcam = None
		if not uses_slot_attention(self.args.attn_model_name):
			get_target_layer = GRADCAM_TARGET_LAYERS.get(self.args.attn_model_name)
			if get_target_layer is None:
				raise ValueError(
					f"No GradCAM target layer defined for attn_model "
					f"'{self.args.attn_model_name}'. Add it to "
					f"GRADCAM_TARGET_LAYERS or SLOT_ATTN_MODELS."
				)
			gradcam_copy = copy.deepcopy(self.attn_model).eval()
			for p in gradcam_copy.parameters():
				p.requires_grad_(True)
			self.gradcam = GradCAM(gradcam_copy, get_target_layer(gradcam_copy))

		self._reset_train_state()
		self._reset_val_state()

		self.propainter = ProPainter(self.device)

		# Freeze the GRU inside recog_model's slot attention — it tends to
		# destabilise training if left unfrozen.
		if 'action_slot' in self.args.model_name:
			for t in self.recog_model.slot_attention.gru.parameters():
				t.requires_grad = False
			slots_mu = self.recog_model.slot_attention.slots_mu
			if isinstance(slots_mu, nn.Parameter):
				slots_mu.requires_grad = False

	# ------------------------------------------------------------------ #
	#  Checkpoint compatibility                                             #
	# ------------------------------------------------------------------ #

	def on_load_checkpoint(self, checkpoint: dict) -> None:
		"""Reconcile checkpoint state_dict with the current model.

		Handles two opposite version mismatches for slots_mu / slots_sigma:

		Case 1 — keys MISSING from checkpoint (old checkpoint, new model):
			The old code called .cuda() before nn.Parameter(), which silently
			dropped these from the state_dict.  Inject them from the current
			model's random init so the rest of the load proceeds strictly.

		Case 2 — keys UNEXPECTED in checkpoint (new checkpoint, old model):
			Python 3.7 / older PyTorch may not register slots_mu/slots_sigma
			as nn.Parameters (they end up as plain tensors or buffers), so
			the current model has no matching key.  Remove them from the
			checkpoint so the strict load doesn't raise on the extra keys.
			The model's own __init__ already initialises these correctly.
		"""
		import warnings
		sd = checkpoint['state_dict']
		slot_keys = {
			'recog_model.slot_attention.slots_mu',
			'recog_model.slot_attention.slots_sigma',
		}

		# ── Case 1: keys missing from checkpoint ──────────────────────
		missing_in_ckpt = slot_keys - sd.keys()
		if missing_in_ckpt:
			warnings.warn(
				f"Checkpoint is missing {missing_in_ckpt}. "
				"Falling back to random initialisation for these keys — "
				"all other weights are loaded normally.",
				UserWarning,
			)
			for key in missing_in_ckpt:
				obj = self
				for attr in key.split('.'):
					obj = getattr(obj, attr)
				sd[key] = obj.data.clone()

		# ── Case 2: keys present in checkpoint but absent from model ──
		# Check which slot_keys the current model does NOT have as parameters
		# or buffers — these must be removed so strict loading doesn't fail.
		unexpected_in_model = set()
		for key in slot_keys & sd.keys():
			obj = self
			try:
				for attr in key.split('.'):
					obj = getattr(obj, attr)
				# Key exists in model — no action needed
			except AttributeError:
				unexpected_in_model.add(key)

		if unexpected_in_model:
			warnings.warn(
				f"Checkpoint contains {unexpected_in_model} but the current "
				"model does not register them (likely a Python/PyTorch version "
				"difference). Removing from checkpoint so loading can proceed.",
				UserWarning,
			)
			for key in unexpected_in_model:
				del sd[key]

	# ------------------------------------------------------------------ #
	#  State helpers                                                        #
	# ------------------------------------------------------------------ #

	def _reset_train_state(self):
		self.batch_num = 0

	def _reset_val_state(self):
		self.example_batch       = []
		self.correct_ego         = 0
		self.total_ego           = 0
		self.map_pred_actor_list = []
		self.label_actor_list    = []

		self.iou_metrics = {
			'iou':                  0.0,
			'overall_intersection': 0.0,
			'overall_union':        0.0,
			'temporal_iou':         0.0,
			'mAP@tIoU':             0.0,
			'precision':            0.0,
			'recall':               0.0,
			'f1':                   0.0,
			'fp_ratio':             0.0,
		}
		self.iou_metrics_per_class = {
			k: [[] for _ in range(NUM_ACTOR_CLASSES)]
			for k in self.iou_metrics
		}

	# ------------------------------------------------------------------ #
	#  Lightning hooks                                                      #
	# ------------------------------------------------------------------ #

	def training_step(self, batch, batch_idx):
		loss_dict = self.shared_step(batch, mode='train')
		loss = torch.tensor(0.0, device=self.device)
		for k, v in loss_dict.items():
			self.log(k, v, prog_bar=True)
			loss += self.action_loss_weight[k] * v
		self.log('total_loss', loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		iou_metrics = self.shared_step(batch, mode='val')
		self.example_batch.append(
			(self.batch_num - 1, iou_metrics['iou'], actor_table[batch['mask_index'][0]])
		)
		mask_index = batch['mask_index']
		for k, v in iou_metrics.items():
			self.iou_metrics[k] += v
			self.iou_metrics_per_class[k][mask_index[0]].append(v)

	def on_validation_epoch_end(self):
		for k in self.iou_metrics:
			self.iou_metrics[k] = self.all_gather(self.iou_metrics[k]).mean()

		n           = num_val / num_gpus
		iou         = self.iou_metrics['iou']               / n
		tiou        = self.iou_metrics['temporal_iou']      / n
		mAP_tiou    = self.iou_metrics['mAP@tIoU']          / n
		precision   = self.iou_metrics['precision']         / n
		recall      = self.iou_metrics['recall']            / n
		f1          = self.iou_metrics['f1']                / n
		fp_ratio    = self.iou_metrics['fp_ratio']          / n
		overall_iou = (
			self.iou_metrics['overall_intersection']
			/ self.iou_metrics['overall_union']
		)

		self.log('val_IOU',         iou,         prog_bar=True)
		self.log('val_overall_IOU', overall_iou, prog_bar=False)
		self.log('val_tIOU',        tiou,        prog_bar=False)
		self.log('val_mAP_tIOU',    mAP_tiou,    prog_bar=False)
		self.log('val_precision',   precision,   prog_bar=False)
		self.log('val_recall',      recall,      prog_bar=False)
		self.log('val_f1',          f1,          prog_bar=False)
		self.log('val_fp_ratio',    fp_ratio,    prog_bar=False)

		if not getattr(self.args, 'pseudo_mask', False):
			self._log_actor_metrics()

		if getattr(self.args, 'val', False):
			self._print_per_group_recall()
			if getattr(self.args, 'per_class_iou', False):
				self._print_per_class_iou()

		self._reset_val_state()

	# ------------------------------------------------------------------ #
	#  Metric helpers                                                       #
	# ------------------------------------------------------------------ #

	def _log_actor_metrics(self):
		preds  = np.stack(self.map_pred_actor_list, axis=0).reshape(-1, NUM_ACTOR_CLASSES)
		labels = np.stack(self.label_actor_list,    axis=0).reshape(-1, NUM_ACTOR_CLASSES)
		mAP    = average_precision_score(labels, preds.astype(np.float32))
		ego    = self.correct_ego / self.total_ego
		self.log('val_mAP', mAP, prog_bar=True)
		self.log('val_ego', ego, prog_bar=False)

	def _print_per_group_recall(self):
		"""Print mean IoU grouped by actor category (c, c+, k, k+, p, p+)."""
		groups = [
			('c',  0,  12),
			('c+', 12, 24),
			('k',  24, 36),
			('k+', 36, 48),
			('p',  48, 56),
			('p+', 56, 64),
		]
		for name, lo, hi in groups:
			all_vals = [
				v
				for cls in range(lo, hi)
				for v in self.iou_metrics_per_class['iou'][cls]
			]
			if not all_vals:
				continue
			mean = torch.stack(all_vals, dim=0).mean()
			print(f'(val) recall of {name}: {mean:.4f}')

	def _print_per_class_iou(self):
		"""Print per-class IoU breakdown across all 64 actor classes (--per_class_iou)."""
		iou_per_class = []
		for cls in range(NUM_ACTOR_CLASSES):
			vals = self.iou_metrics_per_class['iou'][cls]
			if vals:
				mean = torch.stack(vals, dim=0).mean()
				iou_per_class.append(round(float(mean.cpu().numpy()) * 100, 1))
			else:
				iou_per_class.append(float('nan'))

		groups = [
			('c',  iou_per_class[0:12]),
			('c+', iou_per_class[12:24]),
			('k',  iou_per_class[24:36]),
			('k+', iou_per_class[36:48]),
			('p',  iou_per_class[48:56]),
			('p+', iou_per_class[56:64]),
		]
		for name, vals in groups:
			valid = [v for v in vals if not np.isnan(v)]
			mean  = np.mean(valid) if valid else float('nan')
			print(f'(val) per-class IOU [{name}]: {vals}  mean={mean:.1f}')

	# ------------------------------------------------------------------ #
	#  Shared forward / loss step                                           #
	# ------------------------------------------------------------------ #

	def shared_step(self, batch, mode='train'):
		B, T, C, H, W = batch['videos'].shape

		videos     = batch['videos'].to(self.device, dtype=torch.float32).permute(0, 2, 1, 3, 4)
		obj_masks  = batch['obj_masks'].to(self.device)
		mask_index = batch['mask_index'].to(self.device)

		bin_obj_mask = (
			(obj_masks + 1).unsqueeze(1)
			== torch.arange(101, device=self.device).view(1, 101, 1, 1, 1)
		)[:, 1:, ...]

		# ── Run recog_model to get classification outputs + features ──
		pred_ego, pred_actor, feat_low, feat_high = self._run_recog_model(videos)

		# Upsample features to (16, 8, 24) for backbones that temporally downsample.
		# i3d outputs T=4; csn outputs T=1. x3d/action_slot/slot_vps keep T=16.
		if self.args.backbone in ('i3d', 'csn'):
			feat_low  = F.interpolate(feat_low,  (16,  8, 24), mode='trilinear')
			feat_high = F.interpolate(feat_high, (16, 32, 96), mode='trilinear')

		# ── Get attention maps ────────────────────────────────────────
		# When _shared_model=True, attn_model IS recog_model — we reuse
		# the attention maps already produced by _run_recog_model instead
		# of running a second forward pass.
		attn_before = self._get_attention(videos, recog_attn=self._shared_attn)

		self._shared_attn = None  # consumed — clear for next step

		attn_before_query      = _gather_by_index(attn_before, mask_index)
		action_slot_attn_query = attn_before_query

		# CrossAttentionFusion_OneHot uses a class one-hot vector as the query
		# instead of an attention map — no attn_model forward pass is needed.
		if isinstance(self.loc_decoder, CrossAttentionFusion_OneHot):
			mask_index_one_hot = F.one_hot(
				mask_index, num_classes=NUM_ACTOR_CLASSES
			).float()                                    # (B, C)
			pred_mask = self.loc_decoder(feat_low, feat_high, mask_index_one_hot)
		else:
			pred_mask = self.loc_decoder(feat_low, feat_high, attn_before_query)

		attn_query_upsampled = F.interpolate(action_slot_attn_query, (256, 768), mode='bilinear')
		pseudo_mask = refine_pseudo_mask(attn_query_upsampled, bin_obj_mask, mask_index=mask_index)
		pseudo_mask = F.interpolate(pseudo_mask, (32,96), mode='bilinear')
		pred = {
			'pred_mask':   pred_mask,
			'pseudo_mask': pseudo_mask,
			'actor':       pred_actor,
			'ego':         pred_ego,
			# Detach attn_before from the computation graph. ActionSlotLoss uses
			# it purely as a supervision signal (no gradients flow through it).
			# Without detach, when --refine is on, criterion.backward() frees
			# graph A and _compute_action_slot_loss then tries to use the freed
			# graph through pred['attn'] — causing the double-backward error.
			'attn':        attn_before,
			# feat_high is passed so SegmentationLoss.cam_loss can access the
			# feature map when --cam_loss is enabled. It is ignored otherwise.
			'feat':        feat_high,
		}

		if getattr(self.args, 'refine', False) and mode == 'train':
			pred = self._apply_ad_refinement(
				batch, pred, bin_obj_mask, mask_index, action_slot_attn_query
			)

		loss_dict = self.criterion(pred, batch, mode=mode)

		if mode == 'train':
			loss_dict.update(self._compute_action_slot_loss(pred, batch))

		if getattr(self.args, 'val', False) and getattr(self.args, 'vis', False):
			vis_masks_ref_new(
				pseudo_mask, batch['frames_inp'], mask_index,
				logdir, self.batch_num, attn=attn_before_query,
			)

		self.batch_num += 1
		return loss_dict

	def _run_recog_model(self, videos: torch.Tensor):
		"""Forward through recog_model; stash attention if model is shared.

		When attn_model is recog_model (freeze_attn=False), the attention
		maps come for free from this single forward pass.  We stash them in
		self._shared_attn so _get_attention can return them without a second
		forward pass.
		"""
		if uses_slot_attention(self.args.model_name):
			pred_ego, pred_actor, attn, feat_low, feat_high = self.recog_model(videos)
			if self._shared_model:
				self._shared_attn = attn   # reuse in _get_attention
		else:
			pred_ego, pred_actor, feat_low, feat_high = self.recog_model(videos)
			if self._shared_model:
				self._shared_attn = None   # GradCAM path — computed separately

		return pred_ego, pred_actor, feat_low, feat_high

	def _get_attention(
		self,
		videos: torch.Tensor,
		recog_attn: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Return spatial attention / GradCAM maps from attn_model.

		Args:
			videos:      Input tensor (B, C, T, H, W).
			recog_attn:  Pre-computed attention from recog_model's forward pass.
						 Only non-None when _shared_model=True and the model is
						 slot-based — avoids a redundant second forward pass.

		Slot-based models (action_slot, slot_vps):
			- Shared model: returns recog_attn directly (no extra forward pass).
			- Separate frozen model: runs attn_model under no_grad.

		GradCAM-based models (x3d, csn, i3d):
			Always runs GradCAM on an isolated deepcopy under enable_grad,
			regardless of freeze state, to keep the backward pass out of DDP.
		"""
		if uses_slot_attention(self.args.attn_model_name):
			if recog_attn is not None:
				# Shared model path — attention already computed, reuse it
				return recog_attn
			# Separate frozen attn_model
			with torch.no_grad():
				_, _, attn, _, _ = self.attn_model(videos)
			return attn
		else:
			with torch.enable_grad():
				return self.gradcam.get_cam(videos)

	def _apply_ad_refinement(self, batch, pred, bin_obj_mask, mask_index, action_slot_attn_query):
		"""Refine pseudo mask via ProPainter inpainting + attention difference."""
		frames      = batch['frames'].to(self.device)
		frames_inp  = batch['frames_inp'].to(self.device)
		pseudo_mask = pred['pseudo_mask']
		B, T, C, H, W = batch['videos'].shape

		inpaint_mask  = (F.interpolate(pseudo_mask, (256, 768), mode='bilinear') > 0.5).float()
		masks_dilated = binary_dilation_torch(inpaint_mask, iterations=self.args.mask_dilation)

		# ProPainter requires masks at the original video temporal resolution T.
		# Backbones that downsample time (e.g. CSN: T->1, i3d: T->4) produce
		# pseudo masks with fewer frames than the video, causing a size mismatch
		# in the flow completion module (which operates on T-1 frame pairs).
		# Upsample masks_dilated to T if its temporal dim doesn't match.
		# Shape: (B, T_feat, 1, H, W) -> (B, T, 1, H, W)
		if masks_dilated.shape[1] != T:
			masks_dilated = F.interpolate(
				masks_dilated.squeeze(2),          # (B, T_feat, H, W)
				size=(T, 256, 768),
				mode='trilinear',
				align_corners=False,
			).unsqueeze(2)                         # (B, T, 1, H, W)
			masks_dilated = (masks_dilated > 0.5).float()

		comp_frames   = self.propainter.process_video(frames, frames_inp, masks_dilated, masks_dilated)
		inpaint_input = self.normalize(comp_frames).permute(0, 2, 1, 3, 4)

		# Always run a fresh forward pass on the inpainted input — no caching
		attn_after       = self._get_attention(inpaint_input)
		attn_after_query = _gather_by_index(attn_after, mask_index)

		attn_query_up = F.interpolate(action_slot_attn_query, (256, 768), mode='bilinear')
		attn_after_up = F.interpolate(attn_after_query,       (256, 768), mode='bilinear')

		attn_diff  = get_attn_dif(attn_query_up, attn_after_up)
		new_pseudo = refine_pseudo_mask(attn_diff, bin_obj_mask, mask_index=mask_index)
		new_pseudo = F.interpolate(new_pseudo, (32, 96), mode='bilinear')

		pred['pseudo_mask'] = new_pseudo
		return pred

	def _compute_action_slot_loss(self, pred, batch):
		"""Compute action-slot classification + attention losses."""
		action_loss_dict = self.action_criterion(pred, batch)

		ego_loss   = action_loss_dict['ego'] or torch.zeros(1, device=self.device)
		actor_loss = action_loss_dict['actor']

		if uses_slot_attention(self.args.model_name):
			attn_info        = action_loss_dict['attn']
			attn_loss        = (
				self.args.action_attn_weight * attn_info['attn_loss']
				+ self.args.bg_attn_weight   * attn_info['bg_attn_loss']
			)
			action_slot_loss = actor_loss + self.args.ego_loss_weight * ego_loss + attn_loss
		else:
			action_slot_loss = actor_loss + self.args.ego_loss_weight * ego_loss

		return {'action_slot_loss': action_slot_loss}

	# ------------------------------------------------------------------ #
	#  Optimiser                                                            #
	# ------------------------------------------------------------------ #

	def configure_optimizers(self):
		# loc_decoder is always trained.
		param_groups = [{'params': self.loc_decoder.parameters(), 'lr': 1e-5}]

		# recog_model is trained unless freeze_loc=True.
		if not getattr(self.args, 'freeze_loc', False):
			param_groups.append({'params': self.recog_model.parameters(), 'lr': 1e-4})

		# attn_model is NEVER added to the optimizer:
		#   - freeze_attn=True  -> separate frozen model, no grad
		#   - freeze_attn=False -> same object as recog_model, already covered above

		optimizer = optim.Adam(param_groups)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		return [optimizer], [scheduler]


# ====================================================================== #
#  MIL baseline helpers                                                    #
# ====================================================================== #

def expand_obj_masks(
	obj_masks: torch.Tensor,
	max_obj:   int = 100,
) -> torch.Tensor:
	"""Convert integer instance-ID mask to per-object binary masks.

	Args:
		obj_masks: (B, T, H, W) int, values -1=background, 0..N-1=instances.
		max_obj:   Maximum number of objects to expand (pads / clips to this).

	Returns:
		(B, max_obj, T, H, W) float binary tensor.
	"""
	ids = torch.arange(max_obj, device=obj_masks.device)
	return (obj_masks.unsqueeze(1) == ids.view(1, max_obj, 1, 1, 1)).float()


class MILModule(pl.LightningModule):
	"""PyTorch Lightning module for MIL_baseline training and evaluation.

	Training:  RankingLoss on (obj_feat, cls_feat) pairs.
	Validation: MIL_baseline.inference() selects the best-matching object
				mask per frame; metrics computed against action_seg GT.
	"""

	def __init__(
		self,
		args,
		recog_model: nn.Module,
		mil_model:   nn.Module,
		num_actor_class: int,
	):
		super().__init__()
		self.args        = args
		self.recog_model = recog_model
		self.mil_model   = mil_model
		self.criterion   = RankingLoss(args, num_actor_class)
		self._reset_val_state()

	# ------------------------------------------------------------------ #
	#  State helpers                                                        #
	# ------------------------------------------------------------------ #

	def _reset_val_state(self):
		self.iou_metrics = {
			'iou':                  0.0,
			'overall_intersection': 0.0,
			'overall_union':        0.0,
			'temporal_iou':         0.0,
			'mAP@tIoU':             0.0,
			'precision':            0.0,
			'recall':               0.0,
			'f1':                   0.0,
			'fp_ratio':             0.0,
		}
		self.iou_metrics_per_class = {
			k: [[] for _ in range(NUM_ACTOR_CLASSES)]
			for k in self.iou_metrics
		}

	# ------------------------------------------------------------------ #
	#  Data preparation                                                     #
	# ------------------------------------------------------------------ #

	def _prepare_inputs(self, batch):
		"""Extract and prepare all tensors needed by MIL_baseline."""
		device = self.device

		videos     = batch['videos'].to(device, dtype=torch.float32).permute(0, 2, 1, 3, 4)
		mask_index = batch['mask_index'].to(device)
		obj_num    = batch['obj_num'].to(device, dtype=torch.int32)
		onehot     = F.one_hot(mask_index, num_classes=NUM_ACTOR_CLASSES).float()

		obj_masks_raw = batch['obj_masks'].to(device)          # (B, T, H, W)
		H_full, W_full = obj_masks_raw.shape[-2], obj_masks_raw.shape[-1]

		# Full-res binary: (B, 100, T, H_full, W_full)
		ori_obj_mask = expand_obj_masks(obj_masks_raw, max_obj=100)

		return videos, mask_index, onehot, obj_num, ori_obj_mask, H_full, W_full

	def _get_feat_low(self, videos: torch.Tensor) -> torch.Tensor:
		"""Run recog_model forward and return feat_low.

		Return-value unpacking depends on the model family:

			action_slot / slot_vps  (slot models):
				forward() -> (ego, actor, attn, feat_low, feat_high)

			x3d / csn / i3d  (non-slot backbones):
				forward() -> (ego, actor, feat_low, feat_high)

		For action_slot, feat_low is the output of conv3d BEFORE the slot
		attention permute, shape (B, C, T, H, W) — exactly what
		MIL_baseline.obj_mlp expects.
		"""
		if uses_slot_attention(self.args.model_name):
			_, _, _, feat_low, _ = self.recog_model(videos)
		else:
			_, _, feat_low, _ = self.recog_model(videos)

		# Upsample temporally-downsampled backbones to canonical (16, 8, 24)
		if self.args.backbone in ('csn', 'i3d'):
			feat_low = F.interpolate(
				feat_low, (16, 8, 24), mode='trilinear', align_corners=False
			)
		return feat_low

	def _downsample_obj_masks(
		self,
		ori_obj_mask: torch.Tensor,
		feat_shape:   tuple,
	) -> torch.Tensor:
		"""Downsample full-res binary object masks to feature resolution.

		Args:
			ori_obj_mask: (B, 100, T, H_full, W_full)
			feat_shape:   (T_feat, H_feat, W_feat)
		Returns:
			(B, 100, T_feat, H_feat, W_feat) binary float
		"""
		B, N, T, H, W = ori_obj_mask.shape
		T_f, H_f, W_f = feat_shape
		x = ori_obj_mask.reshape(B * N, 1, T, H, W)
		x = F.interpolate(x, size=(T_f, H_f, W_f), mode='trilinear', align_corners=False)
		return (x > 0.5).float().reshape(B, N, T_f, H_f, W_f)

	# ------------------------------------------------------------------ #
	#  Lightning hooks                                                      #
	# ------------------------------------------------------------------ #

	def training_step(self, batch, batch_idx):
		videos, mask_index, onehot, obj_num, ori_obj_mask, _, _ =             self._prepare_inputs(batch)

		feat_low = self._get_feat_low(videos)
		_, _, T_f, H_f, W_f = feat_low.shape
		obj_mask_feat = self._downsample_obj_masks(ori_obj_mask, (T_f, H_f, W_f))

		obj_feat, cls_feat, _ = self.mil_model(
			feat_low, obj_mask_feat, onehot, obj_num
		)

		loss_dict = self.criterion({'obj_feat': obj_feat, 'cls_feat': cls_feat},
								   batch, mode='train')
		loss = sum(loss_dict.values())
		for k, v in loss_dict.items():
			self.log(k, v, prog_bar=True)
		self.log('total_loss', loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		videos, mask_index, onehot, obj_num, ori_obj_mask, _, _ =             self._prepare_inputs(batch)

		feat_low = self._get_feat_low(videos)
		_, _, T_f, H_f, W_f = feat_low.shape
		obj_mask_feat = self._downsample_obj_masks(ori_obj_mask, (T_f, H_f, W_f))

		# inference() returns binary {0,1} — convert to pseudo-logits so
		# calculate_metrics' sigmoid+threshold gives the correct result:
		#   sigmoid(5) ≈ 0.99 (kept)   sigmoid(-5) ≈ 0.01 (dropped)
		pred_mask_binary = self.mil_model.inference(
			feat_low, obj_mask_feat, ori_obj_mask, onehot, obj_num
		)
		pred_mask_logits = pred_mask_binary * 10.0 - 5.0

		iou_metrics = self.criterion(
			{'mask': pred_mask_logits}, batch, mode='val'
		)
		for k, v in iou_metrics.items():
			self.iou_metrics[k] += v
			self.iou_metrics_per_class[k][mask_index[0].item()].append(
				v.item() if isinstance(v, torch.Tensor) else float(v)
			)

	def on_validation_epoch_end(self):
		for k in self.iou_metrics:
			self.iou_metrics[k] = self.all_gather(self.iou_metrics[k]).mean()

		n           = num_val / num_gpus
		iou         = self.iou_metrics['iou']               / n
		tiou        = self.iou_metrics['temporal_iou']      / n
		mAP_tiou    = self.iou_metrics['mAP@tIoU']          / n
		precision   = self.iou_metrics['precision']         / n
		recall      = self.iou_metrics['recall']            / n
		f1          = self.iou_metrics['f1']                / n
		fp_ratio    = self.iou_metrics['fp_ratio']          / n
		overall_iou = (
			self.iou_metrics['overall_intersection']
			/ self.iou_metrics['overall_union']
		)

		self.log('val_IOU',         iou,         prog_bar=True)
		self.log('val_overall_IOU', overall_iou, prog_bar=False)
		self.log('val_tIOU',        tiou,        prog_bar=False)
		self.log('val_mAP_tIOU',    mAP_tiou,    prog_bar=False)
		self.log('val_precision',   precision,   prog_bar=False)
		self.log('val_recall',      recall,      prog_bar=False)
		self.log('val_f1',          f1,          prog_bar=False)
		self.log('val_fp_ratio',    fp_ratio,    prog_bar=False)

		if getattr(self.args, 'val', False):
			self._print_per_group_recall()
			if getattr(self.args, 'per_class_iou', False):
				self._print_per_class_iou()

		self._reset_val_state()

	# ------------------------------------------------------------------ #
	#  Metric printing                                                      #
	# ------------------------------------------------------------------ #

	def _print_per_group_recall(self):
		groups = [
			('c',  0,  12), ('c+', 12, 24),
			('k',  24, 36), ('k+', 36, 48),
			('p',  48, 56), ('p+', 56, 64),
		]
		for name, lo, hi in groups:
			vals = [
				v for cls in range(lo, hi)
				for v in self.iou_metrics_per_class['iou'][cls]
			]
			if not vals:
				continue
			print(f'(val) recall of {name}: {np.mean(vals):.4f}')

	def _print_per_class_iou(self):
		iou_per_class = []
		for cls in range(NUM_ACTOR_CLASSES):
			vals = self.iou_metrics_per_class['iou'][cls]
			iou_per_class.append(
				round(float(np.mean(vals)) * 100, 1) if vals else float('nan')
			)
		groups = [
			('c',  iou_per_class[0:12]),  ('c+', iou_per_class[12:24]),
			('k',  iou_per_class[24:36]), ('k+', iou_per_class[36:48]),
			('p',  iou_per_class[48:56]), ('p+', iou_per_class[56:64]),
		]
		for name, vals in groups:
			valid = [v for v in vals if not np.isnan(v)]
			mean  = np.mean(valid) if valid else float('nan')
			print(f'(val) per-class IOU [{name}]: {vals}  mean={mean:.1f}')

	# ------------------------------------------------------------------ #
	#  Optimiser                                                            #
	# ------------------------------------------------------------------ #

	def configure_optimizers(self):
		param_groups = [{'params': self.mil_model.parameters(), 'lr': 1e-4}]
		if not getattr(self.args, 'freeze_loc', False):
			param_groups.append({'params': self.recog_model.parameters(), 'lr': 1e-5})
		optimizer = optim.Adam(param_groups)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
		return [optimizer], [scheduler]