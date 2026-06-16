import argparse
import json
import os
import sys
from tqdm import tqdm
import scipy.ndimage

import time
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import torch.nn.functional as F
import os
from hsluv import hsluv_to_rgb
from torchvision import models
import matplotlib.image
from scipy.optimize import linear_sum_assignment
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, hamming_loss
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass
import torchvision.transforms as transforms

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')
sys.path.append('../ProPainter')

from taco_ad import to_np, binary_mask
import taco_ad
from inference_propainter import ProPainter
from core.utils import to_tensors
from generate_model import generate_model
from action_slot_utils import *
from parser_eval import get_eval_parser
import warnings

torch.backends.cudnn.benchmark = True


# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

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

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def get_recall(pred_mask, gt_mask, num_classes, mode="class", input='seg'):
    pred_all_mask = torch.zeros_like(gt_mask)
    for i in range(num_classes):
        pred_all_mask = torch.logical_or(pred_all_mask, pred_mask[:,:,i,:,:])

    if mode != "class":
        gt = torch.sum((gt_mask < 64), dtype=torch.float32)
        pred = torch.zeros_like(gt_mask)
        for i in range(num_classes):
            pred = torch.logical_or(pred, pred_mask[:,:,i,:,:])
        tp = torch.sum((pred == 1) & (gt_mask < 64), dtype=torch.float32)
        return tp/gt
    
    recall_class = []
    for i in range(num_classes):
        # pred_mask_i = pred_mask[:,:,i,:,:]
        pred_mask_i = pred_all_mask
        tp = torch.sum((pred_mask_i == 1) & (gt_mask == i), dtype=torch.float32)
        gt = torch.sum((gt_mask == i), dtype=torch.float32)
        if gt == 0:
            recall_class.append(np.nan)
        else:
            recall_class.append(tp / (gt))
    return np.array(recall_class)

def calculate_metrics(pred_mask, gt_mask, num_classes, actor):
    """
    Calculate metrics including mIOU, tIOU, and sIOU.

    :param num_classes: Number of classes in segmentation (excluding background if needed)
    :return: A dict containing each metrics. 
    """
    mean_iou = []
    temporal_iou = []
    spatial_iou = []
    cls_agn_iou = []
    cls_agn_tiou = []
    cls_agn_siou = []
    gt_mask = gt_mask.numpy()
    actor = actor[0]
    pred_mask = pred_mask[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(16):
            pred_mask_i = pred_mask[i]     
            gt_mask_i = gt_mask[i]
            per_class_iou = []
            per_class_temporal_iou = []
            per_class_spatial_iou = []
            gt_all_mask = np.zeros((256, 768))
            pred_all_mask = np.zeros((256, 768))

            for class_id in range(num_classes):
                pred_mask_class = pred_mask_i[class_id]
                if actor[class_id].data != True:
                    iou = np.nan
                    t_iou = np.nan
                    s_iou = np.nan
                else:
                    pred_class = (pred_mask_class == 1)
                    gt_class = (gt_mask_i == class_id)

                    if gt_class.sum() > 0 or pred_class.sum() > 0:
                        if pred_class.sum() > 0 and gt_class.sum() > 0:
                            t_iou = 1
                        else:
                            t_iou = 0
                    else:
                        t_iou = np.nan

                    intersection = np.logical_and(pred_class, gt_class).sum()
                    union = np.logical_or(pred_class, gt_class).sum()

                    if union == 0:  # Avoid division by zero (ignore class if not present in ground truth & prediction)
                        iou = np.nan
                    else:
                        iou = intersection / union

                    if pred_class.sum() > 0:
                        s_iou = iou
                    else:
                        s_iou = np.nan
                per_class_iou.append(iou)
                per_class_temporal_iou.append(t_iou)
                per_class_spatial_iou.append(s_iou)
                
                
                match_indices = np.where(gt_mask_i == class_id)
                gt_all_mask[match_indices] = 1    

                match_indices = np.where(pred_mask_class == 1)
                pred_all_mask[match_indices] = 1
                
            intersection = np.logical_and(pred_class, gt_class).sum()
            union = np.logical_or(pred_class, gt_class).sum()
            if union == 0:  # Avoid division by zero (ignore class if not present in ground truth & prediction)
                iou = np.nan
            else:
                iou = intersection / union
                
            cls_agn_iou.append(iou)                                    
            if pred_all_mask.sum() > 0:
                s_iou = iou
            else:
                s_iou = np.nan
            
            if pred_all_mask.sum() > 0 or gt_all_mask.sum() > 0:
                if pred_all_mask.sum() > 0 and gt_all_mask.sum() > 0:
                    t_iou = 1
                else:
                    t_iou = 0
            else:
                t_iou = np.nan
                
            cls_agn_tiou.append(t_iou)
            cls_agn_siou.append(s_iou)
            
            temporal_iou.append(per_class_temporal_iou)
            spatial_iou.append(per_class_spatial_iou)
            mean_iou.append(per_class_iou)

        per_class_iou = np.nanmean(mean_iou, axis=0)
        per_class_temporal_iou = np.nanmean(temporal_iou, axis=0)
        per_class_spatial_iou = np.nanmean(spatial_iou, axis=0)
        
        mean_iou = np.nanmean(per_class_iou, axis=0)
        temporal_iou = np.nanmean(per_class_temporal_iou, axis=0)
        spatial_iou = np.nanmean(per_class_spatial_iou, axis=0)
        cls_agn_iou = np.nanmean(cls_agn_iou, axis=0)
        
    metrics = {
        "per_class_iou": per_class_iou,
        'mean_iou': mean_iou,
        'per_class_temporal_iou': per_class_temporal_iou,
        'temporal_iou': temporal_iou,
        'per_class_spatial_iou': per_class_spatial_iou,
        'spatial_iou': spatial_iou,
        'cls_agn_iou': cls_agn_iou,
        'cls_agn_tiou': cls_agn_tiou,
        'cls_agn_siou': cls_agn_siou
    }
    return metrics


def get_centroids(masks, actor=None):
    """
    Compute centroids of binary instance segmentation masks.
    
    Parameters:
        masks (numpy array): (H, W, N) binary mask for N instances.
    
    Returns:
        List of (x, y) centroids for each instance.
    """
    centroids = []
    for i in range(masks.shape[-1]):
        if actor is not None and i not in actor:
            continue
        if np.max(masks[:,:,i])==0:
            y, x, = np.nan, np.nan
        else:
            y, x = center_of_mass(masks[:, :, i])  # Compute centroid
        centroids.append((x, y))  # Store (x, y) as tuple
    return np.array(centroids)

def compute_cost_matrix(att_centroids, mask_centroids):
    """
    Compute Euclidean distance cost matrix between attention maps and instance masks.
    
    Parameters:
        att_centroids (numpy array): (N, 2) centroid coordinates of attention maps.
        mask_centroids (numpy array): (M, 2) centroid coordinates of instance masks.
    
    Returns:
        cost_matrix (numpy array): (N, M) distance matrix.
    """
    N, M = len(att_centroids), len(mask_centroids)
    cost_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if np.isnan(mask_centroids[j]).any():
                cost_matrix[i,j] = 384*2
            else:
                cost_matrix[i, j] = np.linalg.norm(att_centroids[i] - mask_centroids[j])  # Euclidean distance
    return cost_matrix

def match_attention_to_masks(att_maps, instance_masks, action_mapping):
    """
    Match attention maps to instance masks using bipartite matching.
    
    Parameters:
        att_maps (numpy array): (H, W, N) attention maps.
        instance_masks (numpy array): (H, W, M) binary instance masks.
    
    Returns:
        matches (dict): Mapping from attention index to instance index.
    """
    # Get centroids
    att_centroids = get_centroids(att_maps, action_mapping)
    mask_centroids = get_centroids(instance_masks)

    # if np.isnan(mask_centroids).any():
    #     return None
        
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(att_centroids, mask_centroids)

    # Solve assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create match dictionary
    matches = {action_mapping[att_idx]: mask_idx for att_idx, mask_idx in zip(row_ind, col_ind)}
    return matches


def calculate_obj_mask_IOU(attn, actor, pred_actor, action_mask, obj_mask, obj_num, map, id, v, raw):
    pred_actor = torch.sigmoid(pred_actor)
    pred_actor = pred_actor > 0.2
    pred_actor = pred_actor[0]

    attn = attn.detach()
    m_l, m_n, m_h, m_w = attn.shape[1], attn.shape[2], attn.shape[3], attn.shape[4]
    attn = torch.reshape(attn, (-1, 1, m_h, m_w))
    attn = F.interpolate(attn, (128,384), mode='bilinear')
    attn = torch.reshape(attn, (1, m_l, m_n, 128, 384))

    obj_mask = torch.stack(obj_mask, dim=0)
    obj_mask = torch.permute(obj_mask, (1,0,2,3,4))
    obj_mask = F.interpolate(obj_mask, (64, 128, 384))

    obj_num = obj_num[0]
    obj_mask = obj_mask[0]
    actor = actor[0]
    attn = attn[0]

    mean_iou = []
    t_iou = []
    s_iou = []
    
    raw = torch.stack(raw, dim=0) # [T, B, C, H, W]
    raw = torch.permute(raw, (1,2,0,3,4)) # [B, C, T, H, W]

    raw = raw.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
    cur_raw = F.interpolate(raw, (3, 128,384))
    cur_raw = cur_raw[0]
    image_list = []
    color_1 = np.array([1.0, 0.0, 0.0])    # Red
    color_2 = np.array([0.0, 1.0, 0.0])    # Green


    path = os.path.join(logdir, 'obj_mask_all')
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(path, map+'_'+id + '_' + v)
    if not os.path.exists(path):
        os.makedirs(path)

    for j in range(16):
        gt_all_mask = np.zeros((128, 384))
        obj_all_mask = np.zeros((128, 384))

        attn_filter = True
        for i, a in enumerate(actor):   
            if a == 1.0:
                match_indices = np.where(action_mask[j]==i)
                gt_all_mask[match_indices] = 1
            if pred_actor[i].data == True:
                attn_filter = attn_filter or (attn[j][i].max() > 0.1)

        if attn_filter:
            for i in range(obj_num[j]):
                match_indices = np.where(obj_mask[j][i] == 1)
                obj_all_mask[match_indices] = 1
            
        intersection = np.logical_and(obj_all_mask, gt_all_mask).sum()
        union = np.logical_or(obj_all_mask, gt_all_mask).sum()

        if union == 0:  # Avoid division by zero (ignore class if not present in ground truth & prediction)
            iou = np.nan
        else:
            iou = intersection / union
            
        if gt_all_mask.sum()>0:
            s_iou.append(iou)
        else:
            s_iou.append(np.nan)
            
        if gt_all_mask.sum()>0 or obj_all_mask.sum()>0:
            if obj_all_mask.sum()>0 and gt_all_mask.sum()>0:
                t_iou.append(1)
            else:
                t_iou.append(0)
        else:
            t_iou.append(np.nan)
        
        mean_iou.append(iou)

    #     raw_j = cur_raw[j].permute(1,2,0).cpu().numpy()
    #     image = np.empty_like(raw_j)
    #     image[:] = raw_j
        
    #     match_indices = np.where(gt_all_mask == 1)
    #     image[match_indices] = color_2

    #     match_indices = np.where(obj_all_mask == 1)
    #     image[match_indices] = color_1
        
    #     image = image * 255
    #     image_list.append(image)
            
    # frames = [Image.fromarray(np.uint8(img)) for img in image_list]

    # output_gif = os.path.join(path, f"all_mask_attn_filtered.gif")
    # frames[0].save(
    #     output_gif,
    #     save_all=True,
    #     append_images=frames[1:],  # Add the remaining frames
    #     optimize=True,
    #     duration=200,  # Duration per frame in milliseconds
    #     loop=0         # Loop forever (set loop=1 for one loop only)
    # )


    mean_iou = np.nanmean(mean_iou)
    s_iou = np.nanmean(s_iou)
    t_iou = np.nanmean(t_iou)
    
    metrics = {
        'mean_iou': mean_iou,
        's_iou': s_iou,
        't_iou': t_iou
    }
    return metrics

def generate_pseudo_mask(attn, actor, pred_actor, obj_mask, obj_num, mode):
    # actor = actor[0]
    # pred_actor = pred_actor[0]
    pred_actor = torch.sigmoid(pred_actor)
    pred_actor = pred_actor > 0.5
    

    seq_len = 16
    attn = attn.detach()
    m_l, m_n, m_h, m_w = attn.shape[1], attn.shape[2], attn.shape[3], attn.shape[4]
    attn = torch.reshape(attn, (-1, 1, m_h, m_w))
    attn = F.interpolate(attn, (256, 768), mode='bilinear')
    attn = torch.reshape(attn, (1, m_l, m_n, 256, 768))
    
    
    if mode == "obj":
        obj_mask = torch.stack(obj_mask, dim=0)
        obj_mask = torch.permute(obj_mask, (1,0,2,3,4))
        obj_mask = F.interpolate(obj_mask, (64, 256, 768))
    
        pseudo_mask_batch = []
        for b in range(len(actor)):
            attn_b = attn[b]
            obj_mask_b = obj_mask[b]
            obj_num_b = obj_num
            pred_actor_b = pred_actor[b]

            action_mapping = []
            for i, a in enumerate(pred_actor_b):
                if a.data == True:
                    action_mapping.append(i)

            pseudo_mask = np.ones((seq_len, 256, 768)) * 64
            for j in range(seq_len):
                masks_j = attn_b[j]
                obj_mask_j = obj_mask_b[j]

                matches = {}
                if obj_num_b[j] > 0:
                    masks_j = masks_j[:-1, ...]
                    masks_j = masks_j.permute(1,2,0)    # (H,W,N)
                    masks_j = masks_j.cpu().numpy()
                            
                    obj_mask_j = obj_mask_j[:obj_num_b[j], ...]
                    obj_mask_j = obj_mask_j.permute(1,2,0)
                    obj_mask_j = obj_mask_j.cpu().numpy().astype('uint8')

                    matches = match_attention_to_masks(masks_j, obj_mask_j, action_mapping)

                    if matches is None:
                        return None
                    
                obj_mask_j = np.transpose(obj_mask_j, (2,0,1))  #(N,H,W)
                for action in action_mapping:
                    if action in matches:
                        match_indices = np.where(obj_mask_j[matches[action]] == 1)                    
                    else:
                        match_indices = [] 
                    pseudo_mask[j][match_indices]=action

    elif mode == "attn":
        # for b in range(len(actor)):
        batch_size = len(actor)
        # attn_b = attn[b]
        # obj_mask_b = obj_mask[b]
        # obj_num_b = obj_num[b]
        # pred_actor_b = pred_actor[b]

        pseudo_mask = torch.zeros((batch_size, seq_len, 65, 256, 768))
        
        # match_indices = torch.where(attn > 0.1)
        # pseudo_mask[match_indices] = 1
        for b in range(batch_size):
            for i,a in enumerate(actor[b]):
                if a == 1.0:
                    for j in range(seq_len):
                        masks_j = attn[b][j]
                        match_indices = torch.where(masks_j[i] > threshold)
                        pseudo_mask[b][j][i][match_indices]=1
        return pseudo_mask
    
    pseudo_mask_batch.append(pseudo_mask)        
    return np.stack(pseudo_mask, axis=0)
                
                
def plot_pseudo_mask(raw, actor, pred_actor, pseudo_mask, action_seg=None, batch_num=None):
    actor = actor[0]
    pred_actor = pred_actor[0]
    # action_seg = action_seg[0]

    pred_actor = torch.sigmoid(pred_actor)
    pred_actor = pred_actor > 0.5
    
    # cur_raw = F.interpolate(raw, (3, 256,768))
    cur_raw = raw[0]

    alpha_1 = 0.1

    color_1 = np.array([255, 0, 0])    # Red
    color_2 = np.array([0.0, 1.0, 0.0])    # Green
    
    pseudo_mask = pseudo_mask[0]
    os.makedirs(f'../gif/attn_{threshold}', exist_ok=True)
    
    if len(pseudo_mask.shape) == 3:
        image_list = []
        for idx in range(16):
            # idx = idx%16
            raw_j = cur_raw[idx].cpu().numpy()
            image = np.empty_like(raw_j)
            image[:] = raw_j
            pseudo_mask_j = pseudo_mask[idx]

            match_indices = torch.where(pseudo_mask_j>0)
            image[match_indices] = color_1
            image_list.append(image)
            
        frames = [Image.fromarray(np.uint8(img)) for img in image_list]

        output_gif = os.path.join(f'../gif/attn_{threshold}', f"mask_{batch_num}.gif")
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],  # Add the remaining frames
            optimize=True,
            duration=200,  # Duration per frame in milliseconds
            loop=0         # Loop forever (set loop=1 for one loop only)
        )
    
    if len(pseudo_mask.shape) == 4:
        for i, a in enumerate(actor):
            if a == 1.0:     
                image_list = []
                for idx in range(16):
                    raw_j = cur_raw[idx].cpu().numpy()
                    image = np.empty_like(raw_j)
                    image[:] = raw_j
                    pseudo_mask_j = pseudo_mask[idx]

                    if action_seg is not None:
                        action_seg_j = action_seg[idx]
                        match_indices = np.where(action_seg_j==i)
                        image[match_indices] = color_2

                    match_indices = np.where(pseudo_mask_j[i]==1)
                    image[match_indices] = color_1
                    
                    image_list.append(image)
                
                frames = [Image.fromarray(np.uint8(img)) for img in image_list]

                output_gif = os.path.join(f'../gif/attn_{threshold}', f"{actor_table[i]}.gif")
                frames[0].save(
                    output_gif,
                    save_all=True,
                    append_images=frames[1:],  # Add the remaining frames
                    optimize=True,
                    duration=200,  # Duration per frame in milliseconds
                    loop=0         # Loop forever (set loop=1 for one loop only)
                )
    
torch.cuda.empty_cache()
args, logdir = get_eval_parser()
print(args)

class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
        
    """

    def __init__(self, args, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.args = args
        self.propainter = ProPainter(self.args.device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def validate(self, model, dataloader, epoch):
        model.eval()

        t_confuse_sample, t_confuse_both_sample, t_confuse_pred, t_confuse_both_pred, t_confuse_both_miss, t_confuse_far_both_sample, t_confuse_far_both_miss = 0, 0, 0, 0, 0, 0, 0

        with torch.no_grad():   
            num_batches = 0
            total_ego = 0
            total_actor = 0

            correct_ego = 0
            correct_actor = 0
            label_actor_list = []
            map_pred_actor_list = []
            # num_selected_sample = 0
            per_class_iou_list = []
            mean_iou_list = []
            temporal_iou_per_class_list = []
            temporal_iou_list = []
            spatial_iou_per_class_list = []
            spatial_iou_list = []
            
            obj_mean_iou_list = []
            obj_s_iou_list = []
            obj_t_iou_list = []

            cls_agn_iou_list = []
            cls_agn_siou_list = []
            cls_agn_tiou_list = []
            
            recall = []
            cls_agn_recall = []

            for batch_num, data in enumerate(tqdm(dataloader)):
  
                if self.args.bg_mask:
                    mask = []
                    mask_in = data['bg_seg']
                    for i in range(self.args.seq_len):
                        if i % self.args.mask_every_frame==0:
                            mask.append(mask_in[i//self.args.mask_every_frame].to(self.args.device, dtype=torch.float32))
                        else:
                            temp_mask = torch.zeros_like(mask_in[0]).to(self.args.device, dtype=torch.float32)
                            mask.append(temp_mask)
                    h, w = mask[0].shape[-2], mask[0].shape[-1]
                    mask = torch.stack(mask, 0)
                    l, b, _, h, w = mask.shape
                    mask = torch.reshape(mask, (l, b, h, w))
                    mask = torch.permute(mask, (1, 0, 2, 3)) #[batch, len, h, w]
                
                map = data['map'][0]
                id = data['id'][0]
                v = data['variants'][0]
                raw = data['raw']
                obj_masks = data['obj_masks']
                obj_num = data['obj_num']
                action_seg = data['action_seg']
                
                
                if args.val_confusion:
                    confusion_label = data['confusion_label']
                scenario = map + '_'+id + '_' + v

                if args.box:
                    box_in = data['box']

                inputs = data['videos'].to(self.args.device, dtype=torch.float32) #[b, T, C, h, w]
                inputs = torch.permute(inputs, (0, 2, 1, 3, 4)) #[b, C, T, h, w]

                if args.box:
                    if isinstance(box_in,np.ndarray):
                        boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
                    else:
                        boxes = box_in.to(args.device, dtype=torch.float32)
                
                batch_size = inputs.shape[0]
                ego = data['ego'].to(args.device)
                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    actor = data['actor'].to(args.device)
                else:
                    actor = torch.FloatTensor(data['actor']).to(args.device)


                if ('slot' in args.model_name) or args.box or 'mvit' in args.model_name:
                    if args.box:
                        pred_ego, pred_actor = model(inputs, boxes)
                    else:
                        if self.args.inpaint != 'GT':
                            pred_ego, pred_actor, attn = model(inputs)
                            obj_masks = F.interpolate(obj_masks, (64, 256, 768))
                            pseudo_mask_batch = generate_pseudo_mask(attn, actor, pred_actor, obj_masks, obj_num, 'obj')
                            recall_class = get_recall(obj_masks, action_seg, 64)
                            recall.append(recall_class)
                            cls_agn_recall.append(get_recall(obj_masks, action_seg, 64, 'agn'))

                        if self.args.inpaint == 'GT':
                            frames = data['frames']
                            frames_inp = data['frames_inp']
                            flow_masks = data['flow_masks']
                            masks_dilated = data['masks_dilated']
                            save_name = 'gt_inpaint'
                            
                        elif self.args.inpaint == 'attn':
                            frames = data['frames']
                            frames_inp = data['frames_inp']
                            save_name = 'attn'
                            
                            flow_masks_batch = []
                            masks_diliated_batch = []
                            pseudo_mask = torch.zeros((pseudo_mask_batch.size(0), pseudo_mask_batch.size(1), pseudo_mask_batch.size(3), pseudo_mask_batch.size(4)))
                            for i in range(64):
                                pseudo_mask = torch.logical_or(pseudo_mask, pseudo_mask_batch[:,:,i,:,:])                          
                            # pseudo_mask = pseudo_mask_batch[:,:,6,:,:]
                            # plot_pseudo_mask(raw, actor, pred_actor, pseudo_mask, batch_num=batch_num)
                            
                            pseudo_mask_batch = pseudo_mask
                            for b in range(pseudo_mask_batch.shape[0]):
                                flow_masks = []
                                masks_dilated = []
                                for i in range(pseudo_mask_batch.shape[1]):
                                    mask_img = pseudo_mask_batch[b][i]
                                    if self.args.mask_dilation > 0:
                                        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=self.args.mask_dilation).astype(np.uint8)
                                    else:
                                        flow_mask_img = binary_mask(mask_img).astype(np.uint8)
                                    flow_masks.append(Image.fromarray(flow_mask_img * 255))
                                    
                                    if self.args.mask_dilation > 0:
                                        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=self.args.mask_dilation).astype(np.uint8)
                                    else:
                                        mask_img = binary_mask(mask_img).astype(np.uint8)
                                    masks_dilated.append(Image.fromarray(mask_img * 255))
                        
                                masks_diliated_batch.append(to_tensors()(masks_dilated))
                                flow_masks_batch.append(to_tensors()(flow_masks))
                            masks_dilated = torch.stack(masks_diliated_batch)
                            flow_masks = torch.stack(flow_masks_batch)

                        if self.args.inpaint != '':
                            
                            frames, frames_inp, flow_masks, masks_dilated = frames.to(self.args.device), frames_inp.to(self.args.device), flow_masks.to(self.args.device), masks_dilated.to(self.args.device)
                            comp_frames = self.propainter.process_video(frames, frames_inp, masks_dilated, flow_masks)
                            inpaint_input = []
                            vis_inpaint(comp_frames, '../gif/attn_test', batch_num)
                            # comp_frames_np = comp_frames.mul(255).permute(0,1,3,4,2)
                            # comp_frames_np = comp_frames.cpu().numpy()
                            # for b in range(batch_size):
                            #     img_list = [Image.fromarray(np.uint8(img)) for img in comp_frames_np[b]]
                            #     # output_gif = os.path.join(f'../gif/attn_{threshold}', f'{save_name}.gif')
                            #     output_gif = os.path.join(f'../gif/attn_test', f'{save_name}_{batch_num}.gif')
                            #     img_list[0].save(
                            #         output_gif,
                            #         save_all=True,
                            #         append_images=img_list[1:],  # Add the remaining frames
                            #         optimize=True,
                            #         duration=200,  # Duration per frame in milliseconds
                            #         loop=0         # Loop forever (set loop=1 for one loop only)
                            #     )

                            comp_frames = self.normalize(comp_frames)

                                # inpaint_input.append(torch.stack(to_np(comp_frames[b], self.args.backbone)))
                                # for idx in range(len(comp_frames_b)):
                                #     f = comp_frames_b[idx]
                                #     f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                                #     img_save_root = os.path.join('../gif', save_name, str(idx).zfill(4)+'.png')
                                #     imwrite(f, img_save_root)
                                    
                            
                            # inpaint_input = torch.stack(inpaint_input).to(self.args.device, dtype=torch.float32)
                            inputs = torch.permute(comp_frames, (0, 2, 1, 3, 4)).to(dtype=torch.float32) #[b, C, T, h, w]
                            
                            pred_ego, pred_actor, attn = model(inputs)
                        
                        if self.args.mask_input == 'pseudo':
                            pseudo_mask_batch = generate_pseudo_mask(attn, actor, pred_actor, obj_masks, obj_num, 'attn')
                            inputs = []
                            pseudo_mask_batch = pseudo_mask_batch.permute(2,0,1,3,4)
                            attn_mask = torch.ones((1, 16, 128, 384)) * 64
                            for a in range(65):
                                match_indices = torch.where(pseudo_mask_batch[a] == 1)
                                attn_mask[match_indices] = a
                                
                            for i in range(self.args.seq_len):
                                mask_color = np.random.randint(256, size=3)
                                input_mask = cv2.resize(attn_mask[0][i].numpy(), (768, 256), interpolation=cv2.INTER_NEAREST)
                                match_indices = np.where(input_mask<64)
                                img = raw[i][0].numpy()
                                img[match_indices] = mask_color
                                inputs.append(img)
                            inputs = torch.stack(to_np(inputs, self.args.model_name, self.args.backbone)).to(args.device, dtype=torch.float32).unsqueeze(0)
                            inputs = torch.permute(inputs, (0, 2, 1, 3, 4)) #[b, C, T, h, w]
                            pred_ego, pred_actor, attn = model(inputs)

                        # if args.action_seg:
                        #     obj_metrics = None
                        #     pseudo_mask_batch = generate_pseudo_mask(attn, actor, pred_actor, obj_masks, obj_num, 'attn')
                        #     if pseudo_mask_batch is None:
                        #         print(map+'_'+id + '_' + v)
                        #         continue
                        #     metrics = calculate_metrics(pseudo_mask_batch, action_seg[0], 64, actor)
                        #     # obj_metrics = calculate_obj_mask_IOU(attn, actor, pred_actor, action_seg[0], obj_masks, obj_num, map, id, v, raw)
                        #     if self.args.plot_pseudo_mask:
                        #         plot_pseudo_mask(map, id, v, raw, actor, pred_actor, logdir, pseudo_mask_batch, action_seg)
                            
                        #     per_class_iou_list.append(metrics['per_class_iou'])
                        #     mean_iou_list.append(metrics['mean_iou'])
                        #     temporal_iou_per_class_list.append(metrics['per_class_temporal_iou'])
                        #     temporal_iou_list.append(metrics['temporal_iou'])
                        #     spatial_iou_per_class_list.append(metrics['per_class_spatial_iou'])
                        #     spatial_iou_list.append(metrics['spatial_iou'])
                            
                        #     if obj_metrics:
                        #         obj_mean_iou_list.append(obj_metrics['mean_iou'])
                        #         obj_s_iou_list.append(obj_metrics['s_iou'])
                        #         obj_t_iou_list.append(obj_metrics['t_iou'])
                            
                        #     cls_agn_iou_list.append(metrics['cls_agn_iou'])
                        #     cls_agn_tiou_list.append(metrics['cls_agn_tiou'])
                        #     cls_agn_siou_list.append(metrics['cls_agn_siou'])
                            
                else:
                    pred_ego, pred_actor = model(inputs)


                num_batches += 1
                pred_ego = torch.nn.functional.softmax(pred_ego, dim=1)
                _, pred_ego = torch.max(pred_ego.data, 1)

                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
                    _, pred_actor_idx = torch.max(pred_actor.data, -1)
                    pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
                    map_batch_new_pred_actor = []
                    for i, b in enumerate(pred_actor_idx):
                        map_new_pred = np.zeros(num_actor_class, dtype=np.float32)+1e-5

                        for j, pred in enumerate(b):
                            if pred != num_actor_class:
                                if pred_actor[i, j, pred] > map_new_pred[pred]:
                                    map_new_pred[pred] = pred_actor[i, j, pred]
                        map_batch_new_pred_actor.append(map_new_pred)
                    map_batch_new_pred_actor = np.array(map_batch_new_pred_actor)
                    map_pred_actor_list.append(map_batch_new_pred_actor)
                    label_actor_list.append(data['slot_eval_gt'])
                else:
                    pred_actor = torch.sigmoid(pred_actor)
                    map_pred_actor_list.append(pred_actor.detach().cpu().numpy())
                    label_actor_list.append(actor.detach().cpu().numpy())

                total_ego += ego.size(0)
                correct_ego += (pred_ego == ego).sum().item()

            map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
            label_actor_list = np.stack(label_actor_list, axis=0)
            
            map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
            label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
            map_pred_actor_list = np.array(map_pred_actor_list)
            label_actor_list = np.array(label_actor_list)
   
            recall = np.array(recall)
            recall_class = np.nanmean(recall, axis=0)
            recall = np.nanmean(recall_class, axis=0)

            # for i, a in enumerate(cls_agn_recall):
            #     if np.isnan(a):
            #         print(i)
            
            print('**********************')   
            print(f'(val) recall of the actor: {recall}')
            print(f'(val) recall of the c: {recall_class[:12]}')
            print(f'(val) recall of the c+: {recall_class[12:24]}')
            print(f'(val) recall of the k: {recall_class[24:36]}')
            print(f'(val) recall of the k+: {recall_class[36:48]}')
            print(f'(val) recall of the p: {recall_class[48:56]}')
            print(f'(val) recall of the p+: {recall_class[56:64]}')
            print(f'(val) class agnostic recall: {np.nanmean(cls_agn_recall)}')            
            print('**********************')   

            # mask = ~np.isnan(label_actor_list) & ~np.isnan(map_pred_actor_list)
            # ground_truth_filtered = label_actor_list[mask]
            # predictions_filtered = map_pred_actor_list[mask]
            # predictions_filtered = np.reshape(predictions_filtered, (num_batches, 64))
            # ground_truth_filtered = np.reshape(ground_truth_filtered, (num_batches, 64))
            
            # mAP = average_precision_score(
            #         ground_truth_filtered,
            #         predictions_filtered.astype(np.float32),
            #         )
            # c_mAP = average_precision_score(
            #         label_actor_list[:, :1],
            #         map_pred_actor_list[:, :1].astype(np.float32)
            #         )
            # b_mAP = average_precision_score(
            #         label_actor_list[:, 24:36],
            #         map_pred_actor_list[:, 24:36].astype(np.float32)
            #         )
            # p_mAP = average_precision_score(
            #         label_actor_list[:, 48:56],
            #         map_pred_actor_list[:, 48:56].astype(np.float32),
            #         )
            # group_c_mAP = average_precision_score(
            #         label_actor_list[:, 12:24],
            #         map_pred_actor_list[:, 12:24].astype(np.float32)
            #         )
            # group_b_mAP = average_precision_score(
            #         label_actor_list[:, 36:48],
            #         map_pred_actor_list[:, 36:48].astype(np.float32)
            #         )
            # group_p_mAP = average_precision_score(
            #         label_actor_list[:, 56:64],
            #         map_pred_actor_list[:, 56:64].astype(np.float32),
            #         )
            mAP_per_class = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32), 
                    average=None)
            
            mAP = np.nanmean(mAP_per_class)
            # mAP = np.nanmean(mAP_per_class)
            for i, ap in enumerate(mAP_per_class):
                mAP_per_class[i] = np.round(ap, 3)*100
            print(f'(val) mAP of the actor: {mAP}')
            # print(f'(val) mAP of the c: {c_mAP}')
            # print(f'(val) mAP of the b: {b_mAP}')
            # print(f'(val) mAP of the p: {p_mAP}')
            # print(f'(val) mAP of the c+: {group_c_mAP}')
            # print(f'(val) mAP of the b+: {group_b_mAP}')
            # print(f'(val) mAP of the p+: {group_p_mAP}')

            print(f'(val) AP of the c: {mAP_per_class[:12]}')
            print(f'(val) AP of the c+: {mAP_per_class[12:24]}')
            print(f'(val) AP of the k: {mAP_per_class[24:36]}')
            print(f'(val) AP of the k+: {mAP_per_class[36:48]}')
            print(f'(val) AP of the p: {mAP_per_class[48:56]}')
            print(f'(val) AP of the p+: {mAP_per_class[56:64]}')

            print('**********************')
            print(f'acc of the ego: {correct_ego/total_ego}')
            print('**********************')

            print(f'mean IOU:   {np.nanmean(mean_iou_list)}')
            print(f'temporal mean IOU:   {np.nanmean(temporal_iou_list)}')
            print(f'spatial mean IOU:   {np.nanmean(spatial_iou_list)}')
            
            per_class_iou = np.nanmean(per_class_iou_list, axis=0)
            print(f'(val) IOU of the c: {per_class_iou[:12]}')
            print(f'(val) IOU of the c+: {per_class_iou[12:24]}')
            print(f'(val) IOU of the k: {per_class_iou[24:36]}')
            print(f'(val) IOU of the k+: {per_class_iou[36:48]}')
            print(f'(val) IOU of the p: {per_class_iou[48:56]}')
            print(f'(val) IOU of the p+: {per_class_iou[56:64]}')

            print('**********************')
            print(f'obj mean IOU:   {np.nanmean(obj_mean_iou_list)}')
            print(f'obj temporal mean IOU:   {np.nanmean(obj_t_iou_list)}')
            print(f'obj spatial mean IOU:   {np.nanmean(obj_s_iou_list)}')
            
            print('**********************')
            print(f'cls agn mean IOU:   {np.nanmean(cls_agn_iou_list)}')
            print(f'cls agn temporal mean IOU:   {np.nanmean(cls_agn_tiou_list)}')
            print(f'cls agn spatial mean IOU:   {np.nanmean(cls_agn_siou_list)}')

            
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 4
num_actor_class = 64
threshold = 0.5

# Data
val_set = taco_ad.TACO(args=args, split='val')
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

model = generate_model(args, num_ego_class, num_actor_class).cuda()
trainer = Engine(args)

model_path = os.path.join(args.cp)
model.load_state_dict(torch.load(model_path))

trainer.validate(model, dataloader_val, None)

