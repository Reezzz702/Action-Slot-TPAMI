python extract_gt_mask.py --attn_model_name action_slot --attn_cp ../weights/taco_action_slot_best_model.pth \
    --refine --vis --vis_n 20 --vis_dir ../vis/action_slot_refined --model_name action_slot --root ../../data_collection/ \
     --dataset taco --num_slots 64 \
    --bg_slot --bg_mask --allocated_slot --seg_only --batch_size 1 --gpus 0 --val_every 2 --obj_mask \
    --ref --pseudo_mask --val --refine