"""
extract_gt_mask.py
------------------
A lightweight script to extract and visualize the Ground Truth (GT) object mask 
for a specific scenario index directly from the dataloader.

Output:
    Saves White-on-Black masks (255 for object, 0 for background).
    
Usage:
    python extract_gt_mask.py
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.append('../datasets')
sys.path.append('../configs')

import taco_loc
from parser import get_parser
from common import actor_table

def main():
    args, _ = get_parser()
    
    # Target index specified by user
    TARGET_IDX = 2475
    
    # ── 1. Load Dataset ──────────────────────────────────────────────────
    print("Loading validation dataset...")
    # Make sure shuffle=False and batch_size=1 so sample_idx matches exactly
    val_set = taco_loc.TACO(args=args, split='val')
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=False, drop_last=False
    )
    
    print(f"Dataset loaded. Fast-forwarding to scenario {TARGET_IDX}...")

    # ── 2. Find and Extract ──────────────────────────────────────────────
    for sample_idx, batch in enumerate(tqdm(val_loader, desc="Searching")):
        if sample_idx == TARGET_IDX:
            
            # (1, T, H, W) -> contains instance IDs (-1 for bg, 0+ for objects)
            obj_masks = batch['obj_masks'] 
            # (1,) -> contains the target actor's class/ID index
            mask_index = batch['mask_index'] 
            
            target_id = mask_index[0].item()
            actor_name = actor_table[target_id]
            
            # Create Output Directory
            out_dir = f"../vis/gt_object_masks/{TARGET_IDX:05d}_{actor_name}"
            os.makedirs(out_dir, exist_ok=True)
            
            # Generate Binary Mask (1.0 for the target object, 0.0 for background)
            # NOTE: If you want ALL objects combined instead of just the target actor, 
            # change the line below to: target_mask = (obj_masks[0] >= 0).float().numpy()
            # target_mask = (obj_masks[0] == target_id).float().numpy()
            target_mask = (obj_masks[0] >= 0).float().numpy()
            
            T = target_mask.shape[0]
            print(f"\n✅ Found scenario {TARGET_IDX} (Actor: {actor_name})!")
            print(f"Extracting {T} frames of object mask...")
            
            # ── 3. Save as White-on-Black JPEGs ──────────────────────────
            for t in range(T):
                # Multiply by 255 to make the 1.0 mask pure White
                frame_img = (target_mask[t] * 255).astype(np.uint8)
                
                # Convert to Grayscale ('L' mode) Image
                img = Image.fromarray(frame_img, mode='L')
                
                # Save
                out_path = os.path.join(out_dir, f"{t:03d}.jpg")
                img.save(out_path, quality=95)
                
            print(f"Done! Masks successfully saved to: {out_dir}")
            break

if __name__ == '__main__':
    main()