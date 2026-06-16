"""
eval_pseudo_mask.py
-------------------
Standalone evaluation script for initial and refined pseudo masks.
Bypasses pytorch_lightning and only loads the attention model.

The attention model can be loaded from:
  1. A raw pretrained checkpoint (args.attn_cp)        -- default
  2. A LocalizationModule training checkpoint           -- via --recog_ckpt

Usage:
    python eval_pseudo_mask.py --attn_model_name x3d --attn_cp weights/x3d.pth
    python eval_pseudo_mask.py --attn_model_name action_slot \\
        --recog_ckpt ../checkpoints/pred_obj/Action-slot_initial/best.ckpt
    python eval_pseudo_mask.py --attn_model_name x3d --attn_cp weights/x3d.pth \\
        --refine --vis --vis_dir ../vis/x3d --vis_n 20

Optional flags:
    --refine           Enable ProPainter-based attention refinement
    --vis              Save visualization GIFs
    --vis_dir <path>   Directory for GIFs (default: ../vis/<attn_model_name>)
    --vis_n <int>      Number of samples to visualize (default: 20)
    --per_class_iou    Print per-class IoU breakdown
"""

import sys
import os
import copy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')
sys.path.append('../ProPainter')

from inference_propainter import ProPainter
from core.utils import to_tensors
from generate_model import generate_model
from action_slot_utils import *
from segmentation import *
from baseline import *
from loss import *
import taco_loc
from parser import get_parser
from common import (
    NUM_ACTOR_CLASSES,
    build_attn_model,
    load_recog_model_from_training_ckpt,
    uses_slot_attention,
    GRADCAM_TARGET_LAYERS,
    _gather_by_index,
    binary_dilation_torch,
)
from vis_pseudo_mask import visualize_pseudo_mask_process


# ====================================================================== #
#  Metric tracking                                                         #
# ====================================================================== #

class PseudoMaskEvaluator:
    def __init__(self, args, device):
        self.args      = args
        self.device    = device
        self.criterion = SegmentationLoss(args, NUM_ACTOR_CLASSES)
        self.reset_metrics()

    def reset_metrics(self):
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
        self.num_samples = 0

    def update_metrics(self, loss_dict, mask_index):
        self.num_samples += 1
        idx = mask_index[0].item() if isinstance(mask_index, torch.Tensor) else mask_index[0]
        for k, v in loss_dict.items():
            if k in self.iou_metrics:
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                self.iou_metrics[k] += val
                self.iou_metrics_per_class[k][idx].append(val)

    def print_results(self, prefix="Pseudo mask"):
        n = self.num_samples
        if n == 0:
            print(f"[{prefix}] No samples evaluated.")
            return

        union       = self.iou_metrics['overall_union']
        overall_iou = (self.iou_metrics['overall_intersection'] / union) if union > 0 else 0.0

        print(f"\n{'='*52}")
        print(f"  {prefix}")
        print(f"{'='*52}")
        print(f"  mIoU        : {self.iou_metrics['iou']           / n:.4f}")
        print(f"  Overall IoU : {overall_iou:.4f}")
        print(f"  tIoU        : {self.iou_metrics['temporal_iou']  / n:.4f}")
        print(f"  mAP@tIoU    : {self.iou_metrics['mAP@tIoU']      / n:.4f}")
        print(f"  Precision   : {self.iou_metrics['precision']      / n:.4f}")
        print(f"  Recall      : {self.iou_metrics['recall']         / n:.4f}")
        print(f"  F1          : {self.iou_metrics['f1']             / n:.4f}")
        print(f"  FP Ratio    : {self.iou_metrics['fp_ratio']       / n:.4f}")
        print(f"{'='*52}\n")

        if getattr(self.args, 'per_class_iou', False):
            self._print_per_class_iou()

    def _print_per_class_iou(self):
        iou_per_class = []
        for cls in range(NUM_ACTOR_CLASSES):
            vals = self.iou_metrics_per_class['iou'][cls]
            iou_per_class.append(
                round(float(np.mean(vals)) * 100, 1) if vals else float('nan')
            )
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
            print(f"  per-class IOU [{name}]: {vals}  mean={mean:.1f}")


# ====================================================================== #
#  Attention helper                                                        #
# ====================================================================== #

def get_attention(attn_model, gradcam, videos, is_slot_attn, attn_model_name):
    """Run forward pass or GradCAM to extract spatial attention maps."""
    if is_slot_attn:
        with torch.no_grad():
            _, _, attn, _, _ = attn_model(videos)
    else:
        with torch.enable_grad():
            attn = gradcam.get_cam(videos)
        if attn_model_name in ('csn', 'i3d'):
            attn = F.interpolate(
                attn, (16, 8, 24), mode='trilinear', align_corners=False
            )
    return attn


# ====================================================================== #
#  Main                                                                    #
# ====================================================================== #

def main():
    args, _ = get_parser()

    torch.manual_seed(42)
    torch.set_float32_matmul_precision('medium')
    device = torch.device(
        f"cuda:{args.gpus[0]}" if torch.cuda.is_available() and args.gpus else "cpu"
    )

    # ── Visualization config ───────────────────────────────────────────
    do_vis  = getattr(args, 'vis', False)
    vis_n   = getattr(args, 'vis_n', 20)
    vis_dir = getattr(args, 'vis_dir', None)
    if do_vis and vis_dir is None:
        vis_dir = f"../vis/{getattr(args, 'attn_model_name', 'model')}"

    # ── 1. Data ────────────────────────────────────────────────────────
    print("Loading test dataset...")
    val_set    = taco_loc.TACO(args=args, split='val')
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False,
    )
    vis_n = len(val_set)

    # ── 2. Attention model ────────────────────────────────────────────
    recog_ckpt = getattr(args, 'recog_ckpt', None)
    if recog_ckpt:
        print(f"Loading attn model from training checkpoint: {recog_ckpt}")
        attn_model = load_recog_model_from_training_ckpt(recog_ckpt, args).to(device)
        effective_attn_model_name = args.model_name
    else:
        print(f"Loading attn model ({args.attn_model_name}) from {args.attn_cp}...")
        args.freeze_attn = True
        attn_model = build_attn_model(args).to(device)
        effective_attn_model_name = args.attn_model_name

    attn_model.eval()
    is_slot_attn = uses_slot_attention(effective_attn_model_name)

    # ── 3. GradCAM (non-slot models only) ─────────────────────────────
    gradcam = None
    if not is_slot_attn:
        get_target_layer = GRADCAM_TARGET_LAYERS.get(effective_attn_model_name)
        if get_target_layer is None:
            raise ValueError(
                f"No GradCAM target layer for '{effective_attn_model_name}'. "
                "Add it to GRADCAM_TARGET_LAYERS in common.py."
            )
        gradcam_copy = copy.deepcopy(attn_model).eval()
        for p in gradcam_copy.parameters():
            p.requires_grad_(True)
        gradcam = GradCAM(gradcam_copy, get_target_layer(gradcam_copy))

    # ── 4. ProPainter (refinement only) ───────────────────────────────
    propainter = None
    normalize  = None
    if args.refine:
        print("Loading ProPainter for refinement...")
        propainter = ProPainter(device)
        normalize  = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

    # ── 5. Evaluators ─────────────────────────────────────────────────
    evaluator_initial = PseudoMaskEvaluator(args, device)
    evaluator_refined = PseudoMaskEvaluator(args, device) if args.refine else None

    source_label = (
        f"training ckpt ({os.path.basename(recog_ckpt)})" if recog_ckpt
        else f"pretrained ({effective_attn_model_name})"
    )
    print(f"\nAttention source : {source_label}")
    print(f"Refinement       : {'on' if args.refine else 'off'}")
    print(f"Visualization    : {'on (first ' + str(vis_n) + ' samples)' if do_vis else 'off'}")
    print(f"Samples          : {len(val_set)}\n")

    # ── 6. Inference loop ─────────────────────────────────────────────
    for sample_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        B          = batch['videos'].shape[0]
        videos     = batch['videos'].to(device, dtype=torch.float32).permute(0, 2, 1, 3, 4)
        obj_masks  = batch['obj_masks'].to(device)
        mask_index = batch['mask_index'].to(device)
        T          = videos.shape[2]

        bin_obj_mask = (
            (obj_masks + 1).unsqueeze(1)
            == torch.arange(101, device=device).view(1, 101, 1, 1, 1)
        )[:, 1:, ...]

        # ── Initial pseudo mask ───────────────────────────────────────
        attn_before       = get_attention(attn_model, gradcam, videos,
                                          is_slot_attn, effective_attn_model_name)
        attn_before_query = _gather_by_index(attn_before, mask_index)
        attn_query_up     = F.interpolate(attn_before_query, (256, 768), mode='bilinear')
        initial_pseudo    = refine_pseudo_mask(
            attn_query_up, bin_obj_mask, mask_index=mask_index
        )

        dummy_actor = torch.zeros((B, NUM_ACTOR_CLASSES), device=device)
        dummy_ego   = torch.zeros((B, 1), device=device)

        pred_initial = {'pred_mask': initial_pseudo, 'actor': dummy_actor, 'ego': dummy_ego}
        with torch.no_grad():
            loss_dict_init = evaluator_initial.criterion(pred_initial, batch, mode='val')
        evaluator_initial.update_metrics(loss_dict_init, mask_index)

        # Refinement intermediates (populated below if --refine)
        comp_frames_vis  = None
        attn_after_vis   = None
        attn_diff_vis    = None
        refined_pseudo   = None

        # ── Refined pseudo mask (optional) ────────────────────────────
        if args.refine and propainter is not None:
            frames     = batch['frames'].to(device)
            frames_inp = batch['frames_inp'].to(device)

            inpaint_mask  = (F.interpolate(initial_pseudo, (256, 768),
                                           mode='bilinear') > 0.5).float()
            masks_dilated = binary_dilation_torch(
                inpaint_mask, iterations=args.mask_dilation
            )

            if masks_dilated.shape[1] != T:
                masks_dilated = F.interpolate(
                    masks_dilated.squeeze(2),
                    size=(T, 256, 768), mode='trilinear', align_corners=False,
                ).unsqueeze(2)
                masks_dilated = (masks_dilated > 0.5).float()

            comp_frames      = propainter.process_video(
                frames, frames_inp, masks_dilated, masks_dilated
            )
            comp_frames_vis  = comp_frames                              # keep for vis
            inpaint_input    = normalize(comp_frames).permute(0, 2, 1, 3, 4)

            attn_after       = get_attention(attn_model, gradcam, inpaint_input,
                                             is_slot_attn, effective_attn_model_name)
            attn_after_query = _gather_by_index(attn_after, mask_index)
            attn_after_up    = F.interpolate(attn_after_query, (256, 768), mode='bilinear')
            attn_after_vis   = attn_after_query                        # keep for vis

            attn_diff        = get_attn_dif(attn_query_up, attn_after_up)
            attn_diff_vis    = attn_diff                               # keep for vis

            refined_pseudo   = refine_pseudo_mask(
                attn_diff, bin_obj_mask, mask_index=mask_index
            )

            pred_refined = {'pred_mask': refined_pseudo, 'actor': dummy_actor, 'ego': dummy_ego}
            with torch.no_grad():
                loss_dict_ref = evaluator_refined.criterion(pred_refined, batch, mode='val')
            evaluator_refined.update_metrics(loss_dict_ref, mask_index)

        # ── Visualization ─────────────────────────────────────────────
        if do_vis and sample_idx < vis_n:
            precision_init = loss_dict_init['precision']
            precision_after = loss_dict_ref['precision']

            actor_name = actor_table[mask_index[0].item()]
            visualize_pseudo_mask_process(
                sample_idx     = sample_idx,
                actor_name     = actor_name,
                vis_dir        = vis_dir,
                frames_inp     = batch['frames_inp'],     # (B, T, H, W, 3)
                attn_before    = attn_before_query,
                initial_pseudo = initial_pseudo,
                comp_frames    = comp_frames_vis,
                attn_after     = attn_after_vis,
                attn_diff      = attn_diff_vis,
                refined_pseudo = refined_pseudo,
                precision_init = precision_init,
                precision_after = precision_after
            )

    # ── 7. Results ────────────────────────────────────────────────────
    evaluator_initial.print_results(
        prefix=f"Initial pseudo mask  [{source_label}]"
    )
    if evaluator_refined is not None:
        evaluator_refined.print_results(
            prefix=f"Refined pseudo mask  [{source_label}]"
        )

    if do_vis:
        print(f"\nGIFs saved to: {vis_dir}/")
        print("  rgb_attn_before/  -- RGB + attention before inpaint")
        print("  initial_mask/     -- initial pseudo mask on RGB")
        if args.refine:
            print("  rgb_after/        -- inpainted RGB + attention after")
            print("  refined_mask/     -- refined pseudo mask on RGB")
            print("  attn_diff/        -- attention difference heatmap")
            print("  side_by_side/     -- 4-panel composite")


if __name__ == '__main__':
    main()