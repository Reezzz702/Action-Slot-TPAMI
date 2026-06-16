"""
train_mil.py
------------
Training entry point for the MIL_baseline activity localization model.

Usage:
    python train_mil.py --model_name action_slot --backbone x3d \
        --cp weights/action_slot.pth --gpus 0 1 --epochs 30 --batch_size 4

Optional flags:
    --mil_hidden_dim   Hidden dimension for MIL MLP (default: 128)
    --freeze_loc       Freeze recog_model during training (Forced to True internally)
    --per_class_iou    Print per-class IoU breakdown at validation
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')
sys.path.append('../ProPainter')

import taco_loc
from parser import get_parser
from baseline import MIL_baseline
from common import (
    NUM_ACTOR_CLASSES,
    MILModule,
    build_recog_model,
    build_pl_trainer,
)


def main():
    args, _ = get_parser()
    logdir  = f'../checkpoints/mil_baseline/{args.backbone}'

    torch.manual_seed(42)
    torch.set_float32_matmul_precision('medium')

    # ── Data ──────────────────────────────────────────────────────────
    train_set = taco_loc.TACO(args=args, split='train_all')
    val_set   = taco_loc.TACO(args=args, split='test')

    import common
    common.num_val  = len(val_set)
    common.num_gpus = len(args.gpus)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False,
    )

    # ── Models ────────────────────────────────────────────────────────
    
    # FIX: Force the freeze flag to True so build_recog_model freezes weights
    args.freeze_loc = True
    recog_model = build_recog_model(args)

    mil_hidden = getattr(args, 'mil_hidden_dim', 128)
    if hasattr(recog_model, 'hidden_dim2'):
        feat_dim = recog_model.hidden_dim2
    elif hasattr(recog_model, 'in_c'):
        feat_dim = recog_model.in_c
    else:
        feat_dim = 192

    mil_model = MIL_baseline(
        feat_dim=192,
        hidden_dim=mil_hidden,
        num_classes=NUM_ACTOR_CLASSES,
    )

    # ── Callbacks / logger ────────────────────────────────────────────
    os.makedirs(logdir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir,
        filename='best',
        monitor='val_IOU',
        mode='max',
        save_top_k=1,
        verbose=True,
        every_n_epochs=1,
    )
    logger = TensorBoardLogger(save_dir=logdir, name='logs')

    best_ckpt   = os.path.join(logdir, 'best.ckpt')
    resume_path = best_ckpt if os.path.exists(best_ckpt) else None

    # ── Train ─────────────────────────────────────────────────────────
    trainer_module = MILModule(
        args=args,
        recog_model=recog_model,
        mil_model=mil_model,
        num_actor_class=NUM_ACTOR_CLASSES,
    )
    pl_trainer = build_pl_trainer(
        args, logdir,
        callbacks=[checkpoint_callback],
        logger=logger,
        resume_path=resume_path,
    )
    pl_trainer.fit(trainer_module, train_loader, val_loader)


if __name__ == '__main__':
    main()