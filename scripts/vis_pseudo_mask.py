"""
vis_pseudo_mask.py
------------------
Visualization utilities for the pseudo mask generation process.

Produces a sequence of JPEG images per sample per panel type on a black background:

    rgb/            -- raw RGB frames (no overlay)
    attn_before/    -- attention heatmap on black background
    initial_mask/   -- initial pseudo mask on black background
    attn_after/     -- attention heatmap on black background       [--refine]
    attn_diff/      -- attention diff heatmap on black background  [--refine]
    refined_mask/   -- refined pseudo mask on black background     [--refine]
    inpainted_rgb/  -- inpainted RGB frames (no overlay)           [--refine]

All functions operate on batch index 0 only (B=1 is assumed for visualization).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


# ====================================================================== #
#  Internal helpers                                                      #
# ====================================================================== #

def _to_rgb_frames(frames_inp: torch.Tensor) -> np.ndarray:
    """Convert frames_inp to (T, H, W, 3) uint8 numpy array.

    Accepts:
        (B, T, H, W, 3) uint8  -- dataloader format
        (B, T, 3, H, W) float  -- ProPainter output (0-1 or 0-255)
    """
    x = frames_inp[0]   # take batch index 0

    if x.dim() == 4 and x.shape[-1] == 3:
        # (T, H, W, 3) already
        return x.cpu().numpy().astype(np.uint8)
    elif x.dim() == 4 and x.shape[1] == 3:
        # (T, 3, H, W) -> (T, H, W, 3)
        x = x.permute(0, 2, 3, 1)
        arr = x.cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        return arr.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected frames shape: {x.shape}")


def _attn_heatmap_overlay(
    rgb_frames: np.ndarray,         # (T, H, W, 3) uint8
    attn:       torch.Tensor,       # (T, H, W) or (B, T, H, W)
    threshold:  float = 0.2,
    alpha:      float = 0.5,
) -> np.ndarray:
    """Overlay a jet heatmap on RGB frames where attn >= threshold.

    Returns (T, H, W, 3) uint8.
    """
    if attn.dim() == 4:
        attn = attn[0]          # (T, H, W)
    T, H, W = rgb_frames.shape[:3]
    cmap   = cm.get_cmap('jet')
    result = rgb_frames.astype(np.float32).copy()

    for t in range(T):
        a = attn[t].cpu().numpy()
        # Upsample attention to frame size if needed
        if a.shape != (H, W):
            a_t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
            a = F.interpolate(a_t, size=(H, W), mode='bilinear',
                              align_corners=False).squeeze().numpy()
        heat      = (cmap(a)[..., :3] * 255).astype(np.float32)
        mask      = (a >= threshold).astype(np.float32)[..., None]
        result[t] = result[t] * (1 - alpha * mask) + heat * (alpha * mask)

    return result.clip(0, 255).astype(np.uint8)


def _mask_overlay(
    rgb_frames:  np.ndarray,        # (T, H, W, 3) uint8
    pred_mask:   torch.Tensor,      # (B, T, H, W) logits or (T, H, W) binary
    color:       tuple = (255, 0, 0),
    alpha:       float = 0.6,
    threshold:   float = 0.5,
) -> np.ndarray:
    """Overlay a binary mask in a solid colour on RGB frames.

    Returns (T, H, W, 3) uint8.
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask[0]    # (T, H, W)

    T, H, W = rgb_frames.shape[:3]
    color_arr = np.array(color, dtype=np.float32)
    result    = rgb_frames.astype(np.float32).copy()

    for t in range(T):
        m = torch.sigmoid(pred_mask[t]).cpu().numpy()
        if m.shape != (H, W):
            m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            m   = F.interpolate(m_t, size=(H, W), mode='bilinear',
                                align_corners=False).squeeze().numpy()
        binary = (m > threshold).astype(np.float32)[..., None]
        result[t] = result[t] * (1 - alpha * binary) + color_arr * (alpha * binary)

    return result.clip(0, 255).astype(np.uint8)


def _save_frames(frames_np: np.ndarray, dir_path: str):
    """Save (T, H, W, 3) uint8 array as a sequence of JPEG files."""
    os.makedirs(dir_path, exist_ok=True)
    for t, frame in enumerate(frames_np):
        img = Image.fromarray(frame)
        # Saves as 000.jpg, 001.jpg, 002.jpg, etc.
        img.save(os.path.join(dir_path, f"{t:03d}.jpg"), quality=95)


# ====================================================================== #
#  Public API                                                            #
# ====================================================================== #
def visualize_pseudo_mask_process(
    sample_idx:     int,
    actor_name:     str,
    vis_dir:        str,
    frames_inp:     torch.Tensor,           # (B, T, H, W, 3) uint8
    attn_before:    torch.Tensor,           # (B, T, H, W)
    initial_pseudo: torch.Tensor,           # (B, T, H, W) soft logits
    comp_frames:    torch.Tensor = None,    # (B, T, 3, H, W) inpainted RGB
    attn_after:     torch.Tensor = None,    # (B, T, H, W)
    attn_diff:      torch.Tensor = None,    # (B, T, H, W) normalised diff
    refined_pseudo: torch.Tensor = None,    # (B, T, H, W) soft logits
    # ── NEW: Metric Tracking ──
    precision_init:     float = None,
    precision_after:     float = None,
):
    """Save all visualization frame sequences for one sample and log metrics."""
    name        = f"{sample_idx:05d}_{actor_name}"
    rgb_before  = _to_rgb_frames(frames_inp)      # (T, H, W, 3)
    
    # Create a pure black background of the exact same shape
    black_bg    = np.zeros_like(rgb_before)
    has_refine  = (comp_frames is not None)

    # ── 1-3. Base Visualizations ──────────────────────────────────────
    panel_rgb = rgb_before.copy()
    _save_frames(panel_rgb, os.path.join(vis_dir, "rgb", name))

    panel_attn_before = _attn_heatmap_overlay(black_bg, attn_before, alpha=1.0)
    _save_frames(panel_attn_before, os.path.join(vis_dir, "attn_before", name))

    panel_initial = _mask_overlay(black_bg, initial_pseudo, color=(255, 50, 50), alpha=1.0)
    _save_frames(panel_initial, os.path.join(vis_dir, "initial_mask", name))

    # ── 4-7. Refinement Visualizations ────────────────────────────────
    if has_refine:
        rgb_after   = _to_rgb_frames(comp_frames)     # (T, H, W, 3)
        black_bg_after = np.zeros_like(rgb_after)

        panel_rgb_after = rgb_after.copy()
        _save_frames(panel_rgb_after, os.path.join(vis_dir, "inpainted_rgb", name))

        panel_attn_after = _attn_heatmap_overlay(black_bg_after, attn_after, alpha=1.0)
        _save_frames(panel_attn_after, os.path.join(vis_dir, "attn_after", name))

        panel_refined = _mask_overlay(black_bg, refined_pseudo, color=(255, 50, 50), alpha=1.0)
        _save_frames(panel_refined, os.path.join(vis_dir, "refined_mask", name))

        panel_diff = _attn_heatmap_overlay(black_bg, attn_diff, threshold=0.1, alpha=1.0)
        _save_frames(panel_diff, os.path.join(vis_dir, "attn_diff", name))

    # ── 8. Record Metrics to CSV ──────────────────────────────────────
    if precision_init is not None:
        csv_path = os.path.join(vis_dir, "visualization_metrics.csv")
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a') as f:
            if write_header:
                f.write("sample_idx,actor_name,precision_init,precision_after,precision_diff\n")
            
            fp_ref_str = f"{precision_after:.4f}" if precision_after is not None else "N/A"
            fp_drop_str = f"{(precision_after - precision_init):.4f}" if precision_after is not None else "N/A"
            
            f.write(f"{sample_idx:05d},{actor_name},{precision_init:.4f},{fp_ref_str},{fp_drop_str}\n")