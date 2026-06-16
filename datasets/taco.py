import os
import json
from PIL import Image
import cv2
import torch.nn.functional as F

import scipy.ndimage
import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
import torchvision.transforms as transforms

sys.path.append('../ProPainter')
from core.utils import to_tensors

class TACO(Dataset):

    def __init__(self, 
                args,
                split='val',
                root='/data/carla_dataset/data_collection',
                Max_N=20, action=None):
        root = args.root

        self.action = action
        self.split = split
        self.model_name = args.model_name
        self.seq_len = args.seq_len
        self.maps = []
        self.id = []
        self.variants = []
        self.scenario_name = []
        self.args =args

        self.videos_list = []
        self.seg_list = []
        self.obj_seg_list = []
        self.action_seg_list = []

        self.idx = []
        self.gt_ego = []
        self.gt_actor = []
        self.slot_eval_gt = []


        self.step = []
        self.start_idx = []
        self.num_class = 64
        self.max_num_obj = []
        
        self.Max_N = Max_N


        max_num_label_a_video = 0
        total_label = 0
        max_frame_a_video = 0
        min_frame_a_video = 100
        total_frame = 0
        total_videos = 0


        n=0

        f = open('../datasets/taco_'+split+'_data.json')
        scenario_list = json.load(f)
        f_label = open('../datasets/taco_'+split+'_label.json')
        label_list = json.load(f_label)
        for scenario in tqdm(scenario_list):
            if not scenario in label_list:
                continue
            gt = label_list[scenario]
                    

            # ------------get labels-------------
            # get multi-instance multi-class labels for object-aware methods
            if self.args.box:
                proposal_train_label, gt_ego, gt_actor = get_labels(args, gt, num_slots=self.Max_N)
            # get multi-instance multi-class labelsfor non-allocated slot-based methods
            elif 'slot' in args.model_name and not args.allocated_slot:
                proposal_train_label, gt_ego, gt_actor = get_labels(args, gt, num_slots=args.num_slots)
            # get multi-label for allocated slot-based and video-level methods
            else:
                gt_ego, gt_actor = get_labels(args, gt, num_slots=args.num_slots)


            # ------------statistics-------------
            if torch.count_nonzero(gt_actor) > max_num_label_a_video:
                max_num_label_a_video = torch.count_nonzero(gt_actor)
            total_label += torch.count_nonzero(gt_actor)
                             
            video_folder = ['downsampled/', 'downsampled_224/']
            if args.model_name == 'mvit' or args.model_name == 'videoMAE':
                video_folder = video_folder[1]
            else:
                video_folder = video_folder[0]

            parent_folder, basic, variant = scenario.split('/')
            
            scenario_path = os.path.join(root,parent_folder,basic,'variant_scenario',variant)
            video_folder_path = os.path.join(scenario_path,'rgb',video_folder)
            if os.path.isdir(video_folder_path):
                check_data = [os.path.join(video_folder_path,img) for img in os.listdir(video_folder_path) if os.path.isfile(os.path.join(video_folder_path,img))]
                check_data.sort()
            else:
                continue

            if len(check_data) < 50:
                continue

            videos = []
            segs = []
            obj_f = []
            idx = []
            action_segs = []

            start_frame = int(check_data[0].split('/')[-1].split('.')[0])
            end_frame = int(check_data[-1].split('/')[-1].split('.')[0])
            num_frame = end_frame - start_frame + 1
            step = num_frame // self.seq_len

            max_num = 50
            for m in range(max_num):
                start = start_frame + m
                if start_frame + (self.seq_len-1)*step > end_frame:
                    break
                videos_temp = []
                seg_temp = []
                idx_temp = []
                obj_temp = []
                action_seg_temp = []
                for i in range(start, end_frame+1, step):
                    imgname = f"{str(i).zfill(8)}.jpg"
                    segname = f"{str(i).zfill(8)}.png"
                    boxname = f"{str(i).zfill(8)}.json"
                    objname = f"{str(i).zfill(8)}.npy"
                    action_segname = f"{str(i).zfill(8)}.npz"
                    if os.path.isfile(os.path.join(video_folder_path,imgname)):
                        videos_temp.append(os.path.join(video_folder_path,imgname))
                        idx_temp.append(i-start_frame)
                    if os.path.isfile(os.path.join(scenario_path,'mask','background',segname)):
                        seg_temp.append(os.path.join(scenario_path,'mask','background',segname))
                    if os.path.isfile(os.path.join(scenario_path,'mask','object',objname)):
                        obj_temp.append(os.path.join(scenario_path,'mask','object',objname))
                    if os.path.isfile(os.path.join(scenario_path, 'action_mask', action_segname)):
                        action_seg_temp.append(os.path.join(scenario_path,'action_mask', action_segname))
                        
                    if len(videos_temp) == self.seq_len:
                        break
                if len(videos_temp) == self.seq_len:
                    videos.append(videos_temp)
                    idx.append(idx_temp)
                    segs.append(seg_temp)
                    obj_f.append(obj_temp)
                    action_segs.append(action_seg_temp)

            self.maps.append(parent_folder)
            self.id.append(basic)
            self.variants.append(variant)
            self.scenario_name.append(os.path.join(parent_folder, basic, variant))
            self.videos_list.append(videos)
            self.idx.append(idx)
            self.seg_list.append(segs)
            self.obj_seg_list.append(obj_f)
            self.gt_ego.append(gt_ego)
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

        print('num_videos: ' + str(len(self.variants)))


    def parse_tracklets_detection(self):
        """
            read {scenario}/tracking_pred_2/tracks/front.txt
            format: frame, id, x1, y1, x2, y2 
        """
        
        def parse_tracklet():
            # frame_id: {id: [x1,y1,x2,y2]}
            out = {}
            for line in tracklet:
                line = line.split(' ')[:6]
                frame = int(line[0])
                obj_id = int(line[1])
                box = [int(line[2]),int(line[3]),int(line[4]),int(line[5])]
                if frame not in out:
                    out[frame] = {}
                out[frame][obj_id] = box
            return out
            
        
        for data,idx in tqdm(zip(self.videos_list,self.idx)):
            root = data[0][0].split('/')
            root = root[:-3]
            root = '/'+os.path.join(*root)
            f = open(os.path.join(root,'tracks','pred','downsampled.txt'))
            tracklet = f.readlines()
            # parse_tracklet
            tracklet = parse_tracklet()
            f.close()
            # for every sample]
            assert len(data) == len(idx)
            for i,idx_list in enumerate(idx):
                out = np.zeros((self.seq_len,self.Max_N,4))
                obj_id_dict = {}
                # tracklet id
                count = 0
                # img frame id
                for j,index in enumerate(idx_list):
                    try:
                        for obj_id in tracklet[int(index)+1]:
                            if obj_id not in obj_id_dict:
                                obj_id_dict[obj_id] = count
                                count += 1
                            try:
                                out[j][obj_id_dict[obj_id]] = tracklet[int(index)+1][obj_id]
                            except:
                                continue
                    except:
                        continue
                np.save(os.path.join(root,'tracks','pred','%s' % (i)),out)
                        
    def tracklet_counter(self):
        """
            tracklet (List[List[Dict]]):
                T , boxes per_frame , key: obj_id
            return:
                T x N x 4
        """

        for idx, data in enumerate(self.videos_list):
            num_samples = len(data)
            root = data[num_samples//2][0].split('/')
            root = root[:-3]
            root = '/'+os.path.join(*root)
            if not os.path.isdir(os.path.join(root,'tracks')):
                os.mkdir(os.path.join(root,'tracks'))
            if not os.path.isdir(os.path.join(root,'tracks','gt')):
                os.mkdir(os.path.join(root,'tracks','gt'))
            if not os.path.isdir(os.path.join(root,'tracks','pred')):
                os.mkdir(os.path.join(root,'tracks','pred'))
            # read bbox.json
            f = open(os.path.join(root,'bbox.json'))
            bboxs = json.load(f)
            f.close()
            obj_id_dict = {}
            count = 0
            remove_data_list = []
            sample = data[num_samples//2]
            for j,frame_idx in enumerate(sample):
                frame_idx = frame_idx.split('/')[-1][:-4]
                for obj_id, box in bboxs[frame_idx].items():
                    if obj_id not in obj_id_dict:
                        obj_id_dict[obj_id] = count
                        count += 1

                if self.args.num_objects == 10 and count > 10:
                    remove_data_list.append(data)
                    break
                if self.args.num_objects == 20 and count < 10 and count > 20:
                    # self.videos_list.remove(data)
                    # del self.videos_list[idx]
                    remove_data_list.append(data)
                    break
                if self.args.num_objects == 21 and count < 20:
                    # self.videos_list.remove(data)
                    # del self.videos_list[idx]
                    remove_data_list.append(data)
                    break
        for video in self.videos_list:
            remove = False
            for remove_data in remove_data_list:
                print(video)
                print(remove_data)
                if remove_data == video:
                    remove = True
                    break
            if remove:
                self.videos_list.remove(video)


    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.videos_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['videos'] = []
        data['bg_seg'] = []
        data['obj_masks'] = []
        data['raw'] = []
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]
        data['obj_num'] = []
        data['action_seg'] = []
        data['masks_dilated'] = []
        data['flow_masks'] = []
        data['frames'] = []
        
        mask_index_candidate = torch.where(self.gt_actor[index]==1)[0]
        select_candidate_idx = random.randrange(0, len(mask_index_candidate))
        data['mask_index'] = torch.stack([mask_index_candidate[select_candidate_idx]])
        
        data['map'] = self.maps[index]
        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            data['slot_eval_gt'] = self.slot_eval_gt[index]

        if 'train' in self.split:
            sample_idx = random.randint(0, len(self.videos_list[index])-1)
        else:
            sample_idx = len(self.videos_list[index])//2

        seq_videos = self.videos_list[index][sample_idx]
        seq_action_seg = self.action_seg_list[index][sample_idx]
        if self.args.bg_mask:
            seq_seg = self.seg_list[index][sample_idx]
        if self.args.obj_mask or (self.args.plot and self.args.plot_mode==''):
            obj_masks_list = self.obj_seg_list[index][sample_idx]

        # add tracklets
        if self.args.box:
            track_path = seq_videos[0].split('/')
            track_path = track_path[:-3]
            if self.args.gt:
                track_path = '/' + os.path.join(*track_path,'tracks','gt',str(sample_idx)) + '.npy'
            else:
                track_path = '/' + os.path.join(*track_path,'tracks','pred',str(sample_idx)) + '.npy'
            tracklets = np.load(track_path)
            data['box'] = tracklets
            
        mask_color = np.random.randint(256, size=3)
        for i in range(self.seq_len):

            x = Image.open(seq_videos[i]).convert('RGB')
            x_np = np.array(x)
                
            # if self.split =='train' or self.split == 'val':
            if self.args.bg_mask:
                if self.args.bg_mask and i %self.args.mask_every_frame == 0:
                    bg_seg = np.array(Image.open(seq_seg[i]).convert('L'))
                    data['bg_seg'].append(bg_seg)
                    input_mask = cv2.resize(bg_seg, (x_np.shape[1], x_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.args.obj_mask:
                obj_mask, obj_num = get_obj_mask(obj_masks_list[i])
                data['obj_masks'].append(obj_mask)
                data['obj_num'].append(obj_num)
                    
            # if self.args.plot or self.args.plot_pseudo_mask or self.args.action_seg:
            data['raw'].append(x)
            
            # x = scale(x, 2, self.args.model_name)
            data['videos'].append(x_np)

        # frames = [Image.fromarray(np.uint8(img)) for img in data['videos']]
        # output_gif = f"../gif/mask_bg_test.gif"
        # frames[0].save(
        #     output_gif,
        #     save_all=True,
        #     append_images=frames[1:],  # Add the remaining frames
        #     optimize=True,
        #     duration=100,  # Duration per frame in milliseconds
        #     loop=0         # Loop forever (set loop=1 for one loop only)
        # )
        
        # if self.args.plot or self.args.plot_pseudo_mask or self.args.action_seg:
        #     data['raw'] = to_np_no_norm(data['raw'])
    
        data['videos'] = torch.stack(to_np(data['videos'], self.args.backbone))
        data['bg_seg'] = to_np_no_norm(data['bg_seg'])
        
        # if self.args.plot_pseudo_mask:
        data['raw'] = np.stack([np.array(f).astype(np.uint8) for f in data['raw']])
        return data


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