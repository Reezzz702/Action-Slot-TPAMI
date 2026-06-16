export TORCH_DISTRIBUTED_DEBUG=INFO
python train_MIL.py --attn_cp ../weights/taco_action_slot_best_model.pth --cp ../weights/taco_action_slot_best_model.pth \
        --root ../../data_collection/ --dataset taco --model_name action_slot --num_slots 64 \
        --bg_slot --bg_mask --allocated_slot --seg_only --batch_size 4 --gpus 0 1 2 3 --val_every 2 --obj_mask \
        --ref --pseudo_mask --cam_loss