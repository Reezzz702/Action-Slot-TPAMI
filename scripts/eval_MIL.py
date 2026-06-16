"""
eval_mil.py
-----------
Evaluation entry point for the MIL_baseline activity localization model.

Usage:
    python eval_mil.py --model_name action_slot --backbone x3d \\
        --cp weights/action_slot.pth --gpus 0 --batch_size 4

Optional flags:
    --mil_hidden_dim   Must match the value used during training (default: 128)
    --per_class_iou    Print per-class IoU breakdown (64 classes)
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
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
    logdir    = f'../checkpoints/mil_baseline/{args.backbone}'
    best_ckpt = os.path.join(logdir, 'best.ckpt')

    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(
            f"No checkpoint found at {best_ckpt}. "
            "Run train_mil.py first."
        )

    torch.manual_seed(42)
    torch.set_float32_matmul_precision('medium')

    # ── Data ──────────────────────────────────────────────────────────
    val_set = taco_loc.TACO(args=args, split='test')

    import common
    common.num_val  = len(val_set)
    common.num_gpus = len(args.gpus)

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False,
    )

    # ── Models ────────────────────────────────────────────────────────
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

    # ── Load checkpoint ───────────────────────────────────────────────
    trainer_module = MILModule.load_from_checkpoint(
        checkpoint_path=best_ckpt,
        args=args,
        recog_model=recog_model,
        mil_model=mil_model,
        num_actor_class=NUM_ACTOR_CLASSES,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    logger     = TensorBoardLogger(save_dir=logdir, name='logs')
    pl_trainer = build_pl_trainer(
        args, logdir,
        callbacks=[],
        logger=logger,
        resume_path=None,
    )
    pl_trainer.validate(trainer_module, val_loader)


if __name__ == '__main__':
    main()