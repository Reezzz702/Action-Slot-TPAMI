import cv2
import os
from PIL import Image
import numpy as np
import json
from tqdm import tqdm


actor_class = {
    4: "pedestrian",
    10:"car",
    12:"pedestrian",
    13:"rider",
    14:"car",
    15:'truck',
    16:'truck',
    18:"motorcycle",
    19:"bike",
}

actor_table = ['c:z1-z2', 'c:z1-z3', 'c:z1-z4',
                'c:z2-z1', 'c:z2-z3', 'c:z2-z4',
                'c:z3-z1', 'c:z3-z2', 'c:z3-z4',
                'c:z4-z1', 'c:z4-z2', 'c:z4-z3',
                'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
                'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
                'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
                'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
                'b:z1-z2', 'b:z1-z3', 'b:z1-z4',
                'b:z2-z1', 'b:z2-z3', 'b:z2-z4',
                'b:z3-z1', 'b:z3-z2', 'b:z3-z4',
                'b:z4-z1', 'b:z4-z2', 'b:z4-z3',
                'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
                'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
                'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
                'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
                'p:c1-c2', 'p:c1-c4', 
                'p:c2-c1', 'p:c2-c3', 
                'p:c3-c2', 'p:c3-c4', 
                'p:c4-c1', 'p:c4-c3', 
                'p+:c1-c2', 'p+:c1-c4', 
                'p+:c2-c1', 'p+:c2-c3', 
                'p+:c3-c2', 'p+:c3-c4', 
                'p+:c4-c1', 'p+:c4-c3',
                'bg']


def show_action_mask(action_masks, action):
    idx = 0
    while True:
        idx = idx%frame_len
        mask = action_masks[idx]
        image = np.zeros((mask.shape[0], mask.shape[1]))
        match_indices = np.where(mask==action)
        image[match_indices] = 1
        
        cv2.imshow("Action mask", image)
        key = cv2.waitKey(0)  # Wait for a key press
        if key == 27:  # ESC key to exit
                break
        elif key == ord('d') or key == 83:  # Right arrow key (next frame)
            idx += 1
        elif key == ord('a') or key == 81:  # Left arrow key (previous frame)
            idx -= 1
    cv2.destroyAllWindows()
    

def show_overlay_result(rgb_dir, save_gif=False):
    overlay_images = []
    for i, frame in enumerate(frame_names):
        rgb = np.array(Image.open(os.path.join(rgb_dir, frame_names[i].replace('png', 'jpg'))).convert('RGB'))
        mask = action_masks[i]
        # Define the color for the foreground (transparent background)
        foreground_color = [255*4, 0, 0]  # Red for foreground

        # Create the color mask with transparency
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # RGBA format
        color_mask[mask == 1] = foreground_color  # Set foreground color
        if i < 102:
            rgb[mask == 1] = (foreground_color + rgb[mask == 1])//5

        # Overlay the color mask on the image
        # overlay = cv2.addWeighted(rgb, 0.2, color_mask, 0.8, 0)

        # Convert back to RGB for visualization (ignore alpha for display purposes)
        overlay_images.append(rgb)
    
    # Convert NumPy arrays to Pillow Images
    frames = [Image.fromarray(np.uint8(img)) for img in overlay_images]
    if save_gif:
        # Save as a GIF
        output_gif = "sample_1.gif"
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],  # Add the remaining frames
            optimize=True,
            duration=200,  # Duration per frame in milliseconds
            loop=0         # Loop forever (set loop=1 for one loop only)
        )    


root = 'path/to/dataset'
f = open('../datasets/taco_test'+'_data.json')
f_label = open('../datasets/taco_test'+'_label.json')
scenario_list = json.load(f)
label_list = json.load(f_label)

print(f"Click the agents performing the target action to label.")
print(f"First click will store the current frame index as the start")
print(f"The second and after clicks will update the current frame index as the end")
count = 0
for scenario in tqdm(scenario_list):
    if not scenario in label_list:
        continue
    count += 1
    
    print(scenario)
    gt = label_list[scenario]
    agent_label = gt['agents']
    
    parent_folder, basic, variant = scenario.split('/')
    scenario_path = os.path.join(root,parent_folder,basic,'variant_scenario',variant)
    mask_dir = os.path.join(scenario_path, 'instance_segmentation/ins_front/')
    rgb_dir = os.path.join(scenario_path, 'rgb/front/')
    action_mask_dir = os.path.join(scenario_path, "action_mask")

    rgb_frame_names = [
        '/'.join(os.path.splitext(p)[:-1]) for p in os.listdir(rgb_dir)
        if os.path.splitext(p)[-1] in [".jpg"]
    ]
    
    # rgb_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    mask_frame_names = [
        '/'.join(os.path.splitext(p)[:-1]) for p in os.listdir(mask_dir)
        if os.path.splitext(p)[-1] in [".png"]
    ]
    
    # mask_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_names = list(set(rgb_frame_names) & set(mask_frame_names))
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    frame_len = len(frame_names)
    if os.path.exists(action_mask_dir):
        action_frame_names = [
            p for p in os.listdir(action_mask_dir)
            if os.path.splitext(p)[-1] in [".npz"]        
        ]
        
        if len(action_frame_names) == frame_len:
            continue
    
    action_masks = np.ones((frame_len, 512, 1536)) * 64
    for action, flag in enumerate(agent_label):
        idx = 0
        agent_dict = {}
        if not flag:
            continue
        
        
        print(actor_table[action])
        def mouse_click(event,x,y,flags,param):    
            if event == cv2.EVENT_LBUTTONDOWN: 
                target_bgr = tuple(mask[y, x])
                if target_bgr[-1] not in actor_class:
                    print('Click on non-actor pixel')
                    print(target_bgr)
                else:
                    print(actor_class[target_bgr[-1]])
                    if target_bgr not in agent_dict:
                        print(f"Start frame {idx}")
                        agent_dict[target_bgr] = [idx]
                    else:
                        if len(agent_dict[target_bgr]) == 2:
                            print(f"Replace the original end frame {agent_dict[target_bgr][1]} with {idx}.")
                            agent_dict[target_bgr][1] = idx
                        else:
                            print(f"End frame: {idx}")
                            agent_dict[target_bgr].append(idx)
                    
        cv2.namedWindow("Front view RGB")
        cv2.setMouseCallback("Front view RGB", mouse_click)        
        while True:
            idx = idx%frame_len
            rgb = np.array(Image.open(os.path.join(rgb_dir, f'{frame_names[idx]}.jpg')).convert('RGB'))
            mask = np.array(Image.open(os.path.join(mask_dir, f'{frame_names[idx]}.png')).convert('RGB'))

            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Front view RGB", mask)
            key = cv2.waitKey(0)  # Wait for a key press
            if key == 27:  # ESC key to exit
                    break
            elif key == ord('d') or key == 83:  # Right arrow key (next frame)
                idx += 1
            elif key == ord('a') or key == 81:  # Left arrow key (previous frame)
                idx -= 1
        cv2.destroyAllWindows()
        
        resize_masks = []
        for i, frame in enumerate(frame_names):
            mask = np.array(Image.open(os.path.join(mask_dir, f'{frame}.png')).convert('RGB'))
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            
            for agent, idx_list in agent_dict.items():
                if i >= idx_list[0] and i <= idx_list[1]:
                    match_indices = np.where(np.all(mask==agent, axis=-1))
                    action_masks[i][match_indices]=action
            
            resize_action_mask = cv2.resize(action_masks[i], (768,256), interpolation=cv2.INTER_NEAREST)
            resize_masks.append(resize_action_mask)
            
    for action, flag in enumerate(agent_label): 
        if flag:
            print(actor_table[action])
            show_action_mask(resize_masks, action)     
    
    os.makedirs(action_mask_dir, exist_ok=True)
    for i, frame in enumerate(frame_names):
        mask = resize_masks[i]
        np.savez_compressed(os.path.join(action_mask_dir, f'{frame}.npz'), a=mask)