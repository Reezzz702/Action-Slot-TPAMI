# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import json
import time

import torch
import torchvision
import sys

# sys.path.append('..')
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

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

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            # masks_img.append(Image.open(os.path.join(mpath, mp)))
            masks_img.append(np.load(os.path.join(mpath, mp))['a'])

    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        # mask_img = np.array(mask_img.convert('L'))
        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class ProPainter(object):
    def __init__(self, device, use_half=True, subvideo_length=80, raft_iter=20, neighbor_length=10, ref_stride=10):
        self.device = device
        self.use_half = use_half
        self.subvideo_length = subvideo_length
        self.raft_iter = raft_iter
        self.neighbor_length = neighbor_length
        self.ref_stride = ref_stride

        pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

        ##############################################
        # set up RAFT and flow competition model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        self.fix_raft = RAFT_bi(ckpt_path, device)
        
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        self.fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(device)
        self.fix_flow_complete.eval()


        ##############################################
        # set up ProPainter model
        ##############################################
        ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                        model_dir='weights', progress=True, file_name=None)
        self.model = InpaintGenerator(model_path=ckpt_path).to(device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

        if use_half:
            self.fix_flow_complete = self.fix_flow_complete.half()
            self.model = self.model.half()


    def compute_optical_flow(self, frames):
        """
        Compute bidirectional optical flow using RAFT.
        """
        video_length = frames.size(1)
        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) <= 720:
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        with torch.no_grad():
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:, f:end_f], iters=self.raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:, f - 1:end_f], iters=self.raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=self.raft_iter)
        return gt_flows_bi

    def complete_flow(self, gt_flows_bi, flow_masks):
        """
        Complete the bidirectional flow.
        """
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.subvideo_length)
                pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks[:, s_f:e_f + 1]
                )
                pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks[:, s_f:e_f + 1]
                )

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()

            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()

        return pred_flows_bi


    def generate_inpainting(self, frames, gt_flows_bi, masks_dilated):
        """
        Generate inpainting results using model with computed flows.
        """
        video_length = frames.size(1)
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, self.subvideo_length)
        updated_frames, updated_masks = [], []
        pad_len = 10

        for f in range(0, video_length, subvideo_length_img_prop):
            s_f = max(0, f - pad_len)
            e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

            pred_flows_bi_sub = (gt_flows_bi[0][:, s_f:e_f - 1], gt_flows_bi[1][:, s_f:e_f - 1])
            prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                masked_frames[:, s_f:e_f], pred_flows_bi_sub, masks_dilated[:, s_f:e_f], 'nearest'
            )
            updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                prop_imgs_sub.view(*frames[:, s_f:e_f].shape) * masks_dilated[:, s_f:e_f]
            updated_masks_sub = updated_local_masks_sub.view(*masks_dilated[:, s_f:e_f].shape)

            updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
            updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
            torch.cuda.empty_cache()

        updated_frames = torch.cat(updated_frames, dim=1)
        updated_masks = torch.cat(updated_masks, dim=1)
        return updated_frames, updated_masks


    def process_video(self, frames, frames_inp, masks_dilated, flow_masks):
        """
        Process the video through flow computation, inpainting, and feature propagation.
        """
        device = frames.device
        self.model.to(device)
        self.fix_flow_complete.to(device)
        
        B,T,C,H,W = frames.shape
        gt_flows_bi = self.compute_optical_flow(frames)

        # use fp16
        if self.use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

        pred_flows_bi = self.complete_flow(gt_flows_bi, flow_masks)
        
        updated_frames, updated_masks = self.generate_inpainting(frames, pred_flows_bi, masks_dilated)
        
        ori_frames = frames_inp.permute(1,0,2,3,4) # T,B,H,W,C

        # comp_frames = [[None] * frames.size(1)] * batch_size
        comp_frames = torch.full((T, B, H, W, C), fill_value=-1, dtype=torch.uint8).to(device)
        neighbor_stride = self.neighbor_length // 2
        ref_num = self.subvideo_length // self.ref_stride if T > self.subvideo_length else -1

        # Feature propagation + transformer
        for f in range(0, T, neighbor_stride):
            neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(T, f + neighbor_stride + 1))]
            ref_ids = get_ref_index(f, neighbor_ids, T, self.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (gt_flows_bi[0][:, neighbor_ids[:-1], :, :, :], gt_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            with torch.no_grad():
                
                l_t = len(neighbor_ids)
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = (pred_img + 1) / 2
                pred_img = (pred_img * 255).to(torch.uint8).permute(1,0,3,4,2) # T,B,H,W,C

                # Prepare binary masks
                binary_masks = masks_dilated[:, neighbor_ids, :, :, :].permute(1,0,3,4,2) # T,B,H,W,C

                # Compute composited frames in batch
                masked_pred_imgs = pred_img * binary_masks
                masked_ori_frames = ori_frames[neighbor_ids] * (1 - binary_masks)
                batched_imgs = masked_pred_imgs + masked_ori_frames
                

                # Blend with existing frames if necessary
                for i, idx in enumerate(neighbor_ids):
                    if (comp_frames[idx] == -1).all():
                        comp_frames[idx] = batched_imgs[i]
                    else:
                        comp_frames[idx] = (comp_frames[idx].float() * 0.5 + batched_imgs[i].float() * 0.5).to(torch.uint8)
                        
        return comp_frames.div(255).permute(1,0,4,2,3) # B,T,C,H,W

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args()

    propainter = ProPainter(device)

    root = '/media/hcis-s15/ssd2/data_collection'
    f = open('/media/hcis-s15/ssd2/Action-slot-PAMI/datasets/taco_val'+'_data.json')
    f_label = open('/media/hcis-s15/ssd2/Action-slot-PAMI/datasets/taco_val'+'_label.json')
    scenario_list = json.load(f)
    label_list = json.load(f_label)

    size = (768, 256)
    for scenario in tqdm(scenario_list[:1]):
        parent_folder, basic, variant = scenario.split('/')
        scenario_path = os.path.join(root,parent_folder,basic,'variant_scenario',variant)
        rgb_dir = os.path.join(scenario_path, 'rgb/downsampled/')
        action_mask_dir = os.path.join(scenario_path, "action_mask")        
        
        gt = label_list[scenario]
        agent_label = gt['agents']

        rgb_frame_names = [
            '/'.join(os.path.splitext(p)[:-1]) for p in os.listdir(rgb_dir)
            if os.path.splitext(p)[-1] in [".jpg"]
        ]
        
        frame_names = None
        # rgb_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if os.path.exists(action_mask_dir):
            action_mask_frame_names = [
                '/'.join(os.path.splitext(p)[:-1]) for p in os.listdir(action_mask_dir)
                if os.path.splitext(p)[-1] in [".npz"]
            ]
            
            action_mask_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            frame_names = action_mask_frame_names
            frame_len = len(frame_names)
        else:
            continue

        ori_frames_batch = []
        frames_batch = []
        masks_batch = []
        flow_batch = []
        action_batch = []
        for action, flag in enumerate(agent_label):
            if not flag:
                continue
            if action > 0:
                break
            action_batch.append(action)
            print(actor_table[action])
            overlay_images = []
            mask_list = []
            frames = []
            masks_dilated = []
            flow_masks = []
            
            for frame in range(0, frame_len, frame_len//14):
                rgb = Image.open(os.path.join(rgb_dir, f'{frame_names[frame]}.jpg')).convert('RGB')
                # resized_rgb = rgb.resize((768,256))
                action_mask = np.load(os.path.join(action_mask_dir, f'{frame_names[frame]}.npz'))['a']
                
                frames.append(rgb)
                
                mask_img = np.zeros((256,768))
                match_indices = np.where(action_mask==action)
                mask_img[match_indices] = 255

                # if size is not None:
                #     mask_img = mask_img.resize(size, Image.NEAREST)
                # mask_img = np.array(mask_img.convert('L'))
                # Dilate 8 pixel so that all known pixel is trustworthy
                if args.mask_dilation > 0:
                    flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.mask_dilation).astype(np.uint8)
                else:
                    flow_mask_img = binary_mask(mask_img).astype(np.uint8)
                # Close the small holes inside the foreground objects
                # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
                # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
                flow_masks.append(Image.fromarray(flow_mask_img * 255))
                
                if args.mask_dilation > 0:
                    mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.mask_dilation).astype(np.uint8)
                else:
                    mask_img = binary_mask(mask_img).astype(np.uint8)
                masks_dilated.append(Image.fromarray(mask_img * 255))
        
            # frames, fps, size, video_name = read_frame_from_videos(rgb_dir)
            if not args.width == -1 and not args.height == -1:
                size = (args.width, args.height)
            if not args.resize_ratio == 1.0:
                size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

            # frames, size, out_size = resize_frames(frames, size)
            out_size = size
            fps = None
            fps = args.save_fps if fps is None else fps
            save_root = os.path.join(scenario_path, 'masked_rgb')
            if not os.path.exists(save_root):
                os.makedirs(save_root, exist_ok=True)

            frames_inp = [np.array(f).astype(np.uint8) for f in frames]
            
            ori_frames_batch.append(frames_inp)
            frames_batch.append(to_tensors()(frames))
            masks_batch.append(to_tensors()(masks_dilated))
            flow_batch.append(to_tensors()(flow_masks))
        
        
        frames = torch.stack(frames_batch) * 2 - 1    
        flow_masks = torch.stack(flow_batch)
        masks_dilated = torch.stack(masks_batch)
        ori_frames_batch = np.stack(ori_frames_batch)
        frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

        comp_frames = propainter.process_video(frames, ori_frames_batch, masks_dilated, flow_masks)
        
        # save each frame
        for b, comp_frames_b in enumerate(comp_frames):
            
            for idx in range(len(comp_frames_b)):
                f = comp_frames_b[idx]
                f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                img_save_root = os.path.join(save_root, f'{actor_table[action_batch[b]]}', str(idx).zfill(4)+'.png')
                imwrite(f, img_save_root)
                
            comp_frames_b = [cv2.resize(f, out_size) for f in comp_frames_b]
            img_list = [Image.fromarray(img) for img in comp_frames_b]
            output_gif = os.path.join(save_root, f'{actor_table[action_batch[b]]}.gif')
            img_list[0].save(
                output_gif,
                save_all=True,
                append_images=img_list[1:],  # Add the remaining frames
                optimize=True,
                duration=100,  # Duration per frame in milliseconds
                loop=0         # Loop forever (set loop=1 for one loop only)
            )