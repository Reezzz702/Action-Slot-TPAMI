"""
train.py
--------
Training entry point for the activity localization model.

Usage:
	python train.py --backbone x3d --gpus 0 1 --epochs 30 --batch_size 4 ...

Resumes automatically from logdir/best.ckpt if it exists.

Optional flags:
	--attn_source      Attention source: "frozen" (pretrained recognizer, default)
					   or "live" (the model under training itself)
	--per_class_iou    Print per-class IoU breakdown (64 classes)
	--ad_train         Enable inpainting-based attention refinement
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
from parser_loc import get_parser
from common import (
	NUM_ACTOR_CLASSES,
	LocalizationModule,
	build_common_kwargs,
	build_pl_trainer,
)


def main():
	args, _ = get_parser()
	logdir  = '../checkpoints/pred_obj/action_slot_refine'

	torch.manual_seed(42)
	torch.set_float32_matmul_precision('medium')

	# ── Data ──────────────────────────────────────────────────────────
	train_set = taco_loc.TACO(args=args, split='train')
	val_set   = taco_loc.TACO(args=args, split='val')

	# num_val and num_gpus are referenced inside LocalizationModule's
	# on_validation_epoch_end — inject them into common so the shared
	# module can access them without circular imports.
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
	common_kwargs = build_common_kwargs(args)

	# ── Callbacks / logger ────────────────────────────────────────────
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

	# ── Resume from last best checkpoint if available ─────────────────
	best_ckpt   = os.path.join(logdir, 'best.ckpt')
	resume_path = best_ckpt if os.path.exists(best_ckpt) else None

	# ── Train ─────────────────────────────────────────────────────────
	trainer_module = LocalizationModule(**common_kwargs)
	pl_trainer     = build_pl_trainer(
		args, logdir,
		callbacks=[checkpoint_callback],
		logger=logger,
		resume_path=resume_path,
	)
	pl_trainer.fit(trainer_module, train_loader, val_loader)


if __name__ == '__main__':
	main()