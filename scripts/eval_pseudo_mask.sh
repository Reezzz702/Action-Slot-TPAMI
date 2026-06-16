python eval_pseudo_mask.py --attn_cp ../weights/best_slot_vps_model.pth --freeze_attn \
               --cp ../weights/best_slot_vps_model.pth --root ../../data_collection/ --dataset taco --model_name slot_vps --attn_model_name slot_vps --num_slots 64 \
                --bg_slot --bg_mask --allocated_slot --seg_only --batch_size 1 --gpus 1 --val_every 2 --obj_mask \
                --ref --pseudo_mask --val --refine