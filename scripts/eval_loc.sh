export TORCH_DISTRIBUTED_DEBUG=INFO
python eval_localization.py --attn_cp ../weights/taco_action_slot_best_model.pth --cp ../weights/taco_action_slot_best_model.pth \
        --root ../../data_collection/ --dataset taco --model_name action_slot --num_slots 64 --attn_model_name action_slot \
        --bg_slot --bg_mask --allocated_slot --seg_only --batch_size 4 --gpus 0 1 2 3 --val_every 2 --obj_mask \
        --ref --pseudo_mask --refine --attn_backbone x3d --backbone x3d --decoder cross_attn_onehot