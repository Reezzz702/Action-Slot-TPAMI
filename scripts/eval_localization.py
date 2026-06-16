"""
eval.py
-------
Evaluation entry point for the activity localization model.

Loads the best checkpoint from logdir and runs validation on the test split.
Supports both checkpoint styles:
  - Checkpoints that include the full model state ('pseudo_mask' runs)
  - Checkpoints saved by ModelCheckpoint (loaded via load_from_checkpoint)

Usage:
	python eval.py --backbone x3d --gpus 0 --batch_size 4 ...

Optional flags:
	--attn_source      Attention source: "frozen" (pretrained recognizer, default)
					   or "live" (the model under training/evaluation itself)
	--per_class_iou    Print per-class IoU breakdown across all 64 actor classes
	--vis              Visualise predicted masks (saved to logdir)
	--vis_attn_dif     Visualise attention difference maps
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
from parser_loc import get_parser
from common import (
	NUM_ACTOR_CLASSES,
	LocalizationModule,
	build_common_kwargs,
	build_pl_trainer,
)


def main():
	args, _ = get_parser()
	logdir  = '../checkpoints/pred_obj/action_slot_onehot_refine'
	best_ckpt = os.path.join(logdir, 'best.ckpt')

	if not os.path.exists(best_ckpt):
		raise FileNotFoundError(
			f"No checkpoint found at {best_ckpt}. "
			"Run train.py first, or point logdir to the correct directory."
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
	common_kwargs = build_common_kwargs(args)

	# ── Load checkpoint ───────────────────────────────────────────────
	# 'pseudo_mask' runs save the module directly (no structured checkpoint);
	# all other runs use ModelCheckpoint and need load_from_checkpoint.
	if 'pseudo_mask' in logdir:
		trainer_module = LocalizationModule(**common_kwargs)
	else:
		trainer_module = LocalizationModule.load_from_checkpoint(
			checkpoint_path=best_ckpt,
			**common_kwargs,
		)

	# ── Logger ────────────────────────────────────────────────────────
	logger = TensorBoardLogger(save_dir=logdir, name='logs')

	# ── Evaluate ──────────────────────────────────────────────────────
	pl_trainer = build_pl_trainer(
		args, logdir,
		callbacks=[],
		logger=logger,
		resume_path=None,   # eval never resumes — weights already loaded above
	)
	pl_trainer.validate(trainer_module, val_loader)


if __name__ == '__main__':
	main()