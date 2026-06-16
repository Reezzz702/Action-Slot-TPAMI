import copy
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append('../ProPainter')
from core.utils import to_tensors


# ── Backbone → video subfolder mapping ────────────────────────────────
_VIDEO_SUBFOLDER = {
    'mvit':     'downsampled_224/',
    'videoMAE': 'downsampled_224/',
}
_VIDEO_SUBFOLDER_DEFAULT = 'downsampled/'


class TACO(Dataset):

    def __init__(
        self,
        args,
        split='val',
        root='/data/carla_dataset/data_collection',
        Max_N=20,
        action=None,
    ):
        root = args.root

        self.action     = action
        self.split      = split
        self.args       = args
        self.seq_len    = args.seq_len
        self.Max_N      = Max_N
        self.num_class  = 64
        self.is_test    = (split == 'test' or split == 'val')

        # ── Per-scenario lists ─────────────────────────────────────────
        self.maps          = []
        self.id            = []
        self.variants      = []
        self.scenario_name = []
        self.videos_list   = []
        self.seg_list      = []
        self.obj_seg_list  = []
        self.action_seg_list = []   # only populated for test split
        self.idx           = []
        self.gt_ego        = []
        self.gt_actor      = []
        self.slot_eval_gt  = []
        self.mask_index    = []
        self.max_num_obj   = []

        # ── Load scenario / label JSON files ──────────────────────────
        with open(f'../datasets/taco_{split}_data.json') as f:
            scenario_list = json.load(f)
        with open(f'../datasets/taco_{split}_label.json') as f:
            label_list = json.load(f)

        if self.is_test and split=='test':
            scenario_list = scenario_list[:600]

        # ── Video subfolder depends on backbone ───────────────────────
        video_subfolder = _VIDEO_SUBFOLDER.get(args.model_name, _VIDEO_SUBFOLDER_DEFAULT)

        # ── Obj mask subfolder comes from args (default: pred by GD-SAM2) ─
        obj_mask_folder = getattr(args, 'obj_mask_folder', 'obj_mask_pred_gd_SAM2')
        print(obj_mask_folder)
        for scenario in tqdm(scenario_list):
            if scenario not in label_list:
                continue

            gt = label_list[scenario]

            # ── Labels ────────────────────────────────────────────────
            if args.box:
                proposal_train_label, gt_ego, gt_actor = get_labels(
                    args, gt, num_slots=self.Max_N
                )
            elif 'slot' in args.model_name and not args.allocated_slot:
                proposal_train_label, gt_ego, gt_actor = get_labels(
                    args, gt, num_slots=args.num_slots
                )
            else:
                gt_ego, gt_actor = get_labels(args, gt, num_slots=args.num_slots)

            # ── Paths ──────────────────────────────────────────────────
            parent_folder, basic, variant = scenario.split('/')
            scenario_path    = Path(root) / parent_folder / basic / 'variant_scenario' / variant
            video_folder_path = scenario_path / 'rgb' / video_subfolder
            obj_mask_path     = scenario_path / obj_mask_folder
            action_mask_path  = scenario_path / 'action_mask'
            bg_mask_path      = scenario_path / 'mask' / 'background'

            # ── Derive temporal range from the rgb folder ──────────────
            # Using rgb as the source of truth (always present, unlike
            # action_mask which only exists for the test split).
            rgb_files = sorted(
                p for p in video_folder_path.iterdir()
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ) if video_folder_path.is_dir() else []

            if len(rgb_files) < 50:
                continue

            start_frame = int(rgb_files[0].stem)
            end_frame   = int(rgb_files[-1].stem)
            num_frame   = end_frame - start_frame + 1
            step        = num_frame // self.seq_len

            # ── Validate action_mask only for test split ───────────────
            if self.is_test and not action_mask_path.is_dir():
                print(f'no action seg gt for scenario: {scenario}')
                continue

            # ── Build per-action-id sample lists ──────────────────────
            for action_id, a in enumerate(gt_actor):
                if a < 1:
                    continue

                videos       = []
                segs         = []
                obj_f        = []
                idx_list_all = []
                action_segs  = []

                for start_offset in range(50):
                    start = start_frame + start_offset
                    if start + (self.seq_len - 1) * step > end_frame:
                        break

                    videos_temp     = []
                    seg_temp        = []
                    idx_temp        = []
                    obj_temp        = []
                    action_seg_temp = []

                    for i in range(start, end_frame + 1, step):
                        stem    = str(i).zfill(8)
                        img_p   = video_folder_path / f'{stem}.jpg'
                        seg_p   = bg_mask_path      / f'{stem}.png'
                        obj_p   = obj_mask_path     / f'{stem}.npz'
                        act_p   = action_mask_path  / f'{stem}.npz'

                        if img_p.is_file():
                            videos_temp.append(str(img_p))
                            idx_temp.append(i - start_frame)

                        if seg_p.is_file():
                            seg_temp.append(str(seg_p))

                        if obj_p.is_file():
                            obj_temp.append(str(obj_p))

                        # action seg only needed at test time
                        if self.is_test and act_p.is_file():
                            action_seg_temp.append(str(act_p))

                        if len(videos_temp) == self.seq_len:
                            break

                    if (
                        len(videos_temp) == self.seq_len
                        and len(obj_temp) == self.seq_len
                    ):
                        videos.append(videos_temp)
                        idx_list_all.append(idx_temp)
                        segs.append(seg_temp)
                        obj_f.append(obj_temp)
                        if self.is_test:
                            action_segs.append(action_seg_temp)

                # Skip this action_id if no valid samples were collected
                if not videos:
                    continue

                self.maps.append(parent_folder)
                self.id.append(basic)
                self.variants.append(variant)
                self.scenario_name.append(os.path.join(parent_folder, basic, variant))
                self.videos_list.append(videos)
                self.idx.append(idx_list_all)
                self.seg_list.append(segs)
                self.obj_seg_list.append(obj_f)
                self.gt_ego.append(gt_ego)
                self.mask_index.append(action_id)

                if self.is_test:
                    self.action_seg_list.append(action_segs)

                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    self.gt_actor.append(proposal_train_label)
                    self.slot_eval_gt.append(gt_actor)
                else:
                    self.gt_actor.append(gt_actor)

        if args.box:
            if args.gt:
                self.parse_tracklets()
            else:
                self.parse_tracklets_detection()

        print(f'num_videos: {len(self.variants)}')

    # ------------------------------------------------------------------ #
    #  Tracklet helpers                                                    #
    # ------------------------------------------------------------------ #

    def parse_tracklets_detection(self):
        """Read predicted tracklets from tracks/pred/downsampled.txt."""

        def _parse_tracklet_file(lines):
            out = {}
            for line in lines:
                parts = line.split()[:6]
                frame, obj_id = int(parts[0]), int(parts[1])
                box = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
                out.setdefault(frame, {})[obj_id] = box
            return out

        for data, idx in tqdm(zip(self.videos_list, self.idx)):
            root = Path(data[0][0]).parents[2]
            track_path = root / 'tracks' / 'pred' / 'downsampled.txt'

            with open(track_path) as f:
                tracklet = _parse_tracklet_file(f.readlines())

            for i, idx_list in enumerate(idx):
                out = np.zeros((self.seq_len, self.Max_N, 4))
                obj_id_dict = {}
                count = 0

                for j, index in enumerate(idx_list):
                    frame_tracklet = tracklet.get(int(index) + 1, {})
                    for obj_id, box in frame_tracklet.items():
                        if obj_id not in obj_id_dict:
                            obj_id_dict[obj_id] = count
                            count += 1
                        slot = obj_id_dict[obj_id]
                        if slot < self.Max_N:
                            out[j][slot] = box

                np.save(str(root / 'tracks' / 'pred' / str(i)), out)

    def tracklet_counter(self):
        """Filter scenarios by number of tracked objects."""
        remove_set = set()

        for idx, data in enumerate(self.videos_list):
            num_samples = len(data)
            root = Path(data[num_samples // 2][0]).parents[2]

            for subdir in ['tracks', 'tracks/gt', 'tracks/pred']:
                (root / subdir).mkdir(exist_ok=True)

            with open(root / 'bbox.json') as f:
                bboxs = json.load(f)

            obj_id_dict = {}
            count = 0
            sample = data[num_samples // 2]

            for frame_path in sample:
                frame_stem = Path(frame_path).stem
                for obj_id in bboxs.get(frame_stem, {}):
                    if obj_id not in obj_id_dict:
                        obj_id_dict[obj_id] = count
                        count += 1

            n = self.args.num_objects
            if (n == 10 and count > 10) or \
               (n == 20 and not (10 <= count <= 20)) or \
               (n == 21 and count < 20):
                remove_set.add(idx)

        self.videos_list = [
            v for i, v in enumerate(self.videos_list) if i not in remove_set
        ]

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.videos_list)

    def __getitem__(self, index):
        data = {
            'ego':        self.gt_ego[index],
            'actor':      self.gt_actor[index],
            'ad_actor':   copy.deepcopy(self.gt_actor[index]),
            'id':         self.id[index],
            'variants':   self.variants[index],
            'map':        self.maps[index],
            'mask_index': self.mask_index[index],
            # Initialise list fields — populated below
            'videos':     [],
            'bg_seg':     [],
            'obj_masks':  [],
            'obj_num':    [],
            'action_seg': [],
            'frames':     [],
        }

        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            data['slot_eval_gt'] = self.slot_eval_gt[index]

        # ── Sample index ──────────────────────────────────────────────
        n_samples = len(self.videos_list[index])
        if 'train' in self.split:
            sample_idx = random.randint(0, n_samples - 1)
        else:
            sample_idx = n_samples // 2

        seq_videos     = self.videos_list[index][sample_idx]
        seq_action_seg = self.action_seg_list[index][sample_idx] if self.is_test else []
        seq_seg        = self.seg_list[index][sample_idx] if self.args.bg_mask else []
        obj_masks_list = self.obj_seg_list[index][sample_idx] if self.args.obj_mask else []

        # ── Tracklets ─────────────────────────────────────────────────
        if self.args.box:
            root = Path(seq_videos[0]).parents[2]
            suffix = 'gt' if self.args.gt else 'pred'
            track_path = root / 'tracks' / suffix / f'{sample_idx}.npy'
            data['box'] = np.load(str(track_path))

        # ── Per-frame loading ─────────────────────────────────────────
        for i in range(self.seq_len):
            x    = Image.open(seq_videos[i]).convert('RGB')
            x_np = np.array(x)
            data['frames'].append(x)
            data['videos'].append(x_np)

            if self.args.bg_mask and seq_seg and i % self.args.mask_every_frame == 0:
                bg_seg = np.array(Image.open(seq_seg[i]).convert('L'))
                data['bg_seg'].append(bg_seg)

            if self.args.obj_mask and obj_masks_list:
                obj_mask = np.load(obj_masks_list[i])['a']
                data['obj_masks'].append(obj_mask)
                data['obj_num'].append(int(np.max(obj_mask)) + 1)

            # action_seg only loaded at test time
            if self.is_test and seq_action_seg:
                action_seg = np.load(seq_action_seg[i])['a']
                action_seg = cv2.resize(
                    action_seg, (768, 256), interpolation=cv2.INTER_NEAREST
                )
                data['action_seg'].append(action_seg)

        # ── Collate ───────────────────────────────────────────────────
        data['videos']     = torch.stack(to_np(data['videos'], self.args.backbone))
        data['frames_inp'] = np.stack([np.array(f).astype(np.uint8) for f in data['frames']])
        data['frames']     = to_tensors()(data['frames']) * 2 - 1

        if data['bg_seg']:
            data['bg_seg'] = to_np_no_norm(data['bg_seg'])
        else:
            data['bg_seg'] = []

        if data['obj_masks']:
            data['obj_masks'] = np.stack(data['obj_masks'], axis=0)
            data['obj_num']   = np.stack(data['obj_num'],   axis=0)

        if data['action_seg']:
            data['action_seg'] = np.stack(data['action_seg'], axis=0)

        return data

def get_binary_obj_mask(fg_mask, max_instance_id=100):
    """
    Convert [T, H, W] foreground mask to binary [N, T, H, W] object masks.
    
    Args:
        fg_mask: np.ndarray of shape [T, H, W], values 0..N-1 or -1 (background)
        max_instance_id: optional int, number of instances N (if known)
        
    Returns:
        binary_mask: np.ndarray of shape [N, T, H, W], dtype=bool
    """
    T, H, W = fg_mask.shape
    fg_mask = fg_mask.copy()
    
    unique_ids = np.unique(fg_mask)
    unique_ids = unique_ids[unique_ids != -1]  # remove background
    N = max_instance_id if max_instance_id is not None else (unique_ids.max() + 1)

    binary_mask = np.zeros((N, T, H, W), dtype=np.float32)
    
    for n in unique_ids:
        binary_mask[n] = (fg_mask == n)

    return binary_mask

def get_obj_mask(obj_path):
    obj_masks = np.load(obj_path)
    # obj_masks = list(seg_dict.values())
    obj_num = obj_masks.shape[0]
    if obj_masks.shape[0] == 0:
        obj_masks = torch.zeros([64, 32, 96], dtype=torch.int32)
    else:
        obj_masks = torch.from_numpy(np.stack(obj_masks, 0))
    # img = torch.flip(torch.from_numpy(img).type(torch.int).permute(2,0,1),[0])
    obj_masks = obj_masks.type(torch.int)
    pad_num = 64 - obj_masks.shape[0]
    obj_masks = torch.cat((obj_masks, torch.zeros([pad_num, 32, 96], dtype=torch.int32)), dim=0)
    obj_masks = obj_masks.type(torch.float32)

    return obj_masks, obj_num


def scale(image, scale=2.0, model_name=None):

    if scale == -1.0:
        (width, height) = (224, 224)
    else:
        (width, height) = (int(image.width // scale), int(image.heighft // scale))
    # (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    return im_resized



def to_np(v, backbone):
    if backbone != 'inception':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
    else:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])        
    for i, _ in enumerate(v):
        v[i] = transform(v[i])
    return v

def to_np_no_norm(v):
    transform = transforms.Compose([
                transforms.ToTensor(),
                ])
    for i, _ in enumerate(v):
        v[i] = transform(v[i])
    return v

def get_labels(args, gt, num_slots=64):   
    num_class = 64
    model_name = args.model_name
    allocated_slot = args.allocated_slot
    agent_label = gt['agents']
    ego_label = gt['ego']

    ego_table = {'e:z1-z1': 0, 'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3}

    actor_table = { 'c:z1-z2': 0, 'c:z1-z3':1, 'c:z1-z4':2,
                    'c:z2-z1': 3, 'c:z2-z3': 4, 'c:z2-z4': 5,
                    'c:z3-z1': 6, 'c:z3-z2': 7, 'c:z3-z4': 8,
                    'c:z4-z1': 9, 'c:z4-z2': 10, 'c:z4-z3': 11,

                    'c+:z1-z2': 12, 'c+:z1-z3':13, 'c+:z1-z4':14,
                    'c+:z2-z1': 15, 'c+:z2-z3': 16, 'c+:z2-z4': 17,
                    'c+:z3-z1': 18, 'c+:z3-z2': 19, 'c+:z3-z4': 20,
                    'c+:z4-z1': 21, 'c+:z4-z2': 22, 'c+:z4-z3': 23,

                    'b:z1-z2': 24, 'b:z1-z3':25, 'b:z1-z4':26,
                    'b:z2-z1': 27, 'b:z2-z3': 28, 'b:z2-z4': 29,
                    'b:z3-z1': 30, 'b:z3-z2': 31, 'b:z3-z4': 32,
                    'b:z4-z1': 33, 'b:z4-z2': 34, 'b:z4-z3': 35,

                    'b+:z1-z2': 36, 'b+:z1-z3':37, 'b+:z1-z4':38,
                    'b+:z2-z1': 39, 'b+:z2-z3': 40, 'b+:z2-z4': 41,
                    'b+:z3-z1': 42, 'b+:z3-z2': 43, 'b+:z3-z4': 44,
                    'b+:z4-z1': 45, 'b+:z4-z2': 46, 'b+:z4-z3': 47,


                    'p:c1-c2': 48, 'p:c1-c4': 49, 
                    'p:c2-c1': 50, 'p:c2-c3': 51, 
                    'p:c3-c2': 52, 'p:c3-c4': 53, 
                    'p:c4-c1': 54, 'p:c4-c3': 55,

                    'p+:c1-c2': 56, 'p+:c1-c4': 57, 
                    'p+:c2-c1': 58, 'p+:c2-c3': 59, 
                    'p+:c3-c2': 60, 'p+:c3-c4': 61, 
                    'p+:c4-c1': 62, 'p+:c4-c3': 63 
                    }

    ego_label = torch.tensor(ego_label)
    agent_label = torch.FloatTensor(agent_label)
    proposal_train_label = []
    if ('slot' in model_name and not allocated_slot) or 'ARG'in model_name or 'ORN'in model_name:
        proposal_train_label = matches = [x for x in agent_label if x > 0]
        while (len(proposal_train_label)!= num_slots):
            proposal_train_label.append(num_class)
        proposal_train_label = torch.LongTensor(proposal_train_label)
        return proposal_train_label, ego_label, agent_label
    else:
        return ego_label, agent_label
    
def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask