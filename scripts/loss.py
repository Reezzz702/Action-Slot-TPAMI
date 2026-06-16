import numpy as np
import warnings
from sklearn.metrics import average_precision_score
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from action_slot_utils import *

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


# ====================================================================== #
#  ActionSlot recognition loss                                             #
# ====================================================================== #

class ActionSlotLoss(nn.Module):
    def __init__(self, args, num_actor_class, attention_res=None):
        super(ActionSlotLoss, self).__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        self.attention_res = attention_res
        self.ego_ce = nn.CrossEntropyLoss(reduction='mean')
        self.actor_loss_type = self._parse_actor_loss(args)
        self.attn_loss_type  = self._parse_attn_loss(args)

    def _parse_actor_loss(self, args):
        # pos_weight = torch.ones([self.num_actor_class]) * args.bce_pos_weight
        # self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        # return 0
        if ('slot' in args.model_name and not args.allocated_slot) or args.box:
            ce_weights = torch.ones(self.num_actor_class + 1) * args.ce_pos_weight
            ce_weights[-1] = args.ce_neg_weight
            self.bce = nn.CrossEntropyLoss(reduction='mean', weight=ce_weights)
            return 1
        else:
            pos_weight = torch.ones([self.num_actor_class]) * args.bce_pos_weight
            self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
            return 0

    def _parse_attn_loss(self, args):
        flag = 0
        if (('slot' in args.model_name and not args.allocated_slot) or args.box) and args.obj_mask:
            flag = 1
        elif 'slot' in args.model_name and args.allocated_slot:
            if not args.bg_mask and args.action_attn_weight > 0:
                flag = 2
            elif args.obj_mask:
                flag = 1
            elif args.bg_slot and args.bg_mask and args.action_attn_weight > 0. and args.bg_attn_weight > 0.:
                flag = 3
            elif args.bg_slot and args.bg_mask and args.bg_attn_weight > 0. and not args.action_attn_weight > 0.:
                flag = 4
        if flag > 0:
            self.obj_bce = nn.BCELoss()
        if args.ref:
            flag = 3
        # flag = 0
        return flag

    def ego_loss(self, pred, label):
        if pred is None:
            return None
        return self.ego_ce(pred, label)

    def actor_loss(self, pred, label):
        if self.actor_loss_type == 1:
            bs, num_queries = pred.shape[:2]
            out_prob = pred.clone().detach().flatten(0, 1).softmax(-1)
            actor_gt_np = label.clone().detach()
            tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
            C = -out_prob[:, tgt_ids].clone().detach()
            C = C.view(bs, num_queries, -1).cpu()
            sizes = [len(v) for v in actor_gt_np]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            indx = [(torch.as_tensor(i, dtype=torch.int64, device=pred.device).detach(),
                     torch.as_tensor(j, dtype=torch.int64, device=pred.device).detach())
                    for i, j in indices]
            batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
            src_idx = torch.cat([src for (src, _) in indx]).detach()
            idx = (batch_idx, src_idx)
            target_classes_o = torch.cat([t[J] for t, (_, J) in zip(label, indx)]).to(pred.device)
            target_classes = torch.full(pred.shape[:2], self.num_actor_class,
                                        dtype=torch.int64, device=out_prob.device)
            target_classes[idx] = target_classes_o
            actor_loss = self.bce(pred.transpose(1, 2), target_classes)
        else:
            actor_loss = self.bce(pred, label)
        return actor_loss

    def attn_loss(self, attn, label, actor, validate):
        attn_loss = None
        bg_attn_loss = None
        action_attn = None
        bg_attn = None
        device = attn.device

        if self.args.bg_attn_weight > 0:
            bg_seg = []
            bg_seg_in = label['bg_seg']
            for i in range(self.args.seq_len // self.args.mask_every_frame):
                bg_seg.append(bg_seg_in[i].to(device, dtype=torch.float32))
            h, w = bg_seg[0].shape[-2], bg_seg[0].shape[-1]
            bg_seg = torch.stack(bg_seg, 0)
            l, b, _, h, w = bg_seg.shape
            bg_seg = torch.reshape(bg_seg, (l, b, h, w))
            bg_seg = torch.permute(bg_seg, (1, 0, 2, 3))

        if self.attn_loss_type == 1:
            obj_mask_list = []
            obj_mask = label['obj_masks']
            for i in range(self.args.seq_len // self.args.mask_every_frame):
                obj_mask_list.append(obj_mask[i].to(device, dtype=torch.float32))
            obj_mask_list = torch.stack(obj_mask_list, 0)
            obj_mask_list = torch.permute(obj_mask_list, (1, 0, 2, 3, 4))
            b, l, n, h, w = obj_mask_list.shape

            attn_loss = 0.0
            b, l, n, h, w = attn.shape
            attn = torch.reshape(attn, (-1, 1, 8, 24))
            attn = F.interpolate(attn, size=(32, 96))
            attn = torch.reshape(attn, (b, l, n, 32, 96))
            attn = attn[:, ::self.args.mask_every_frame, :, :, :].reshape(b, -1, n, 32, 96)

            b, seq, n_obj, h, w = obj_mask_list.shape
            mask_detach = attn.detach().flatten(3, 4).cpu().numpy()
            mask_gt_np  = obj_mask_list.flatten(3, 4).detach().cpu().numpy()
            scores = np.zeros((b, 4, n, n_obj))
            for i in range(b):
                for j in range(4):
                    ce = (np.matmul(np.log(mask_detach[i, j]), mask_gt_np[i, j].T)
                          + np.matmul(np.log(1 - mask_detach[i, j]), (1 - mask_gt_np[i, j]).T))
                    scores[i, j] += ce
            for i in range(b):
                for j in range(4):
                    matches = linear_sum_assignment(-scores[i, j])
                    id_slot, id_gt = matches
                    attn_loss += self.obj_bce(attn[i, j, id_slot, :, :], obj_mask_list[i, j, id_gt, :, :])

        elif self.attn_loss_type == 2:
            b, l, n, h, w = attn.shape
            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)
            action_attn = attn[:, :, :self.num_actor_class, :, :]
            class_idx = (label['actor'] == 0.0).view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))
            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32).to(attn.device)
            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

        elif self.attn_loss_type == 3:
            attn = torch.permute(attn, (0, 2, 1, 3, 4))
            b, l, n, h, w = attn.shape
            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)
            action_attn = attn[:, :, :self.num_actor_class, :, :]
            bg_attn = attn[:, ::self.args.mask_every_frame, -1, :, :].reshape(b, -1, h, w)
            class_idx = (label['actor'] == 0.0).view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))
            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32, device=attn.device)
            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])
            bg_attn_loss = self.args.bg_attn_weight * self.obj_bce(bg_attn, bg_seg)

        elif self.attn_loss_type == 4:
            b, l, n, h, w = attn.shape
            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)
            bg_attn = attn[:, ::self.args.mask_every_frame, -1, :, :].reshape(b, l // self.args.mask_every_frame, h, w)
            bg_attn_loss = self.obj_bce(bg_attn, bg_seg)

        elif self.attn_loss_type == 5:
            b, n, l, o = attn.shape
            cls_label = label['actor'].to(attn.device)
            absent_mask = (cls_label == 0).float().unsqueeze(-1).unsqueeze(-1)
            attn_cls = attn[:, :-1, :, :]
            absent_attn = attn_cls * absent_mask
            absent_loss = absent_attn.pow(2).mean()
            bg_attn = attn[:, n - 1, :, :]
            target = torch.zeros_like(bg_attn)
            target[..., -1] = 1.0
            bg_loss = F.mse_loss(bg_attn, target)
            attn_loss = absent_loss
            bg_attn_loss = bg_loss

        loss = {'attn_loss': attn_loss, 'bg_attn_loss': bg_attn_loss}
        if validate:
            loss['action_inter'] = None
            loss['action_union'] = None
            loss['bg_inter']     = None
            loss['bg_union']     = None
            if action_attn is not None:
                action_attn_pred = action_attn[class_idx] > 0.5
                inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                loss['action_inter'] = inter
                loss['action_union'] = union
            if bg_attn is not None:
                bg_attn_pred = bg_attn > 0.5
                inter, union = inter_and_union(bg_attn_pred, bg_seg, 1, 1)
                loss['bg_inter'] = inter
                loss['bg_union'] = union
        return loss

    def forward(self, pred, label, validate=False):
        ego_loss       = self.ego_loss(pred['ego'], label['ego'])
        actor_loss     = self.actor_loss(pred['actor'], label['actor'])
        attention_loss = self.attn_loss(pred['attn'], label, pred['actor'], validate)
        return {'ego': ego_loss, 'actor': actor_loss, 'attn': attention_loss}


# ====================================================================== #
#  Ranking loss                                                            #
# ====================================================================== #

class RankingLoss(nn.Module):
    def __init__(self, args, num_actor_class):
        super().__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        self.t = 1

    def rank_loss(self, obj_feat, cls_feat, cls_id, margin=0.3):
        B, N, T, D = obj_feat.shape
        device = obj_feat.device
        obj_feat = obj_feat.permute(0, 2, 1, 3)
        cls_feat_exp = cls_feat.unsqueeze(0)
        obj_feat_exp = obj_feat.unsqueeze(1)
        sim = (obj_feat_exp * cls_feat_exp.unsqueeze(3)).sum(dim=-1)
        A = sim.max(dim=-1).values
        A_pos = A[torch.arange(B), torch.arange(B)]
        A_pos_expand = A_pos.unsqueeze(1).expand(B, B, T)
        cls_id_col = cls_id.view(1, -1).expand(B, B)
        cls_id_row = cls_id.view(-1, 1).expand(B, B)
        cls_diff_mask = (cls_id_col != cls_id_row).to(device)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        valid_mask = (cls_diff_mask & eye_mask).unsqueeze(-1).expand(B, B, T)
        loss = (margin + A[valid_mask] - A_pos_expand[valid_mask]).clamp(min=0)
        num_neg = valid_mask.sum().clamp(min=1)
        return {'rank_loss': loss.sum() / num_neg}

    def forward(self, pred, batch, mode='train'):
        if mode == 'train':
            obj_feat = pred['obj_feat']
            cls_feat = pred['cls_feat']
            device   = obj_feat.device
            cls_id   = batch['mask_index']
            return self.rank_loss(obj_feat, cls_feat, cls_id)
        else:
            pred_mask  = pred['mask']
            device     = pred_mask.device
            gt_mask    = batch['action_seg'].to(device)
            mask_index = batch['mask_index'].to(device)
            B, T, H, W = gt_mask.shape
            gt_mask_one_hot = F.one_hot(gt_mask.long(), num_classes=65).permute(0, 4, 1, 2, 3)
            gt_mask_one_hot = torch.gather(
                gt_mask_one_hot, 1,
                mask_index.view(B, 1, 1, 1, 1).expand(-1, -1, 16, H, W)
            ).squeeze(1).float()
            return calculate_metrics(pred_mask, gt_mask_one_hot)


# ====================================================================== #
#  Segmentation loss (with optional cam_loss)                              #
# ====================================================================== #

class SegmentationLoss(nn.Module):
    def __init__(self, args, num_actor_class):
        super().__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        pos_weight = torch.ones([self.num_actor_class]) * args.bce_pos_weight
        self.actor_bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    def segmentation_loss(self, pred_mask, gt_mask, dice_weight=1.0, focal_weight=0.0):
        """
        Args:
            pred_mask: (B, T, H, W) — raw logits
            gt_mask:   (B, T, H, W) — binary ground truth
        """
        pos_weight = self.args.bce_pos_weight * torch.ones_like(pred_mask).to(pred_mask.device)
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask.float(), pos_weight=pos_weight)
        total_loss = bce

        if dice_weight > 0:
            prob = torch.sigmoid(pred_mask)
            intersection = (prob * gt_mask).sum(dim=(1, 2, 3))
            union = prob.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
            dice = 1 - (2 * intersection + 1) / (union + 1)
            valid = gt_mask.sum(dim=(1, 2, 3)) > 0
            if valid.any():
                total_loss += dice_weight * dice[valid].mean()

        if focal_weight > 0:
            logits_clipped = pred_mask.clamp(min=-10, max=10)
            prob = torch.sigmoid(logits_clipped)
            pt = torch.where(gt_mask == 1, prob, 1 - prob)
            focal = -((1 - pt) ** 2) * torch.log(pt + 1e-6)
            total_loss += focal_weight * focal.mean()

        return {'seg_loss': total_loss}

    def ad_loss(self, pred_actor, gt_actor, query_cls):
        B, C = pred_actor.shape
        device = pred_actor.device
        query_mask = F.one_hot(query_cls, num_classes=C).float().to(device)
        loss_query = F.binary_cross_entropy_with_logits(
            pred_actor * query_mask,
            torch.zeros_like(pred_actor) * query_mask,
            reduction='none',
        )
        loss_query = (loss_query * query_mask).sum() / query_mask.sum().clamp(min=1)
        keep_mask  = 1.0 - query_mask
        loss_keep  = self.actor_bce(pred_actor * keep_mask, gt_actor * keep_mask)
        return {'ad_loss': loss_query + loss_keep}

    def cam_loss(self, features, pseudo_masks, class_idx, k_c=6.0, eps=1e-6):
        """CAM contrastive loss.

        Args:
            features:     (B, D, T, H, W) — feature map from recog_model
            pseudo_masks: (B, 1, T, H, W) — soft mask for the queried class.
                          cam_loss treats it as C=1 since only one class is
                          queried per sample; class_idx identifies which class.
            class_idx:    (B,) — queried activity class index per sample
            k_c:          scaling factor for cosine similarities
        """
        B, D, T, H, W = features.shape
        losses = []

        for b in range(B):
            cls      = class_idx[b].item()
            # pseudo_masks is (B, 1, T, H, W) — treat channel 0 as the queried class
            mask_b   = pseudo_masks[b]   # (1, T, H, W)

            for t in range(T):
                feat_bt  = features[b, :, t]  # (D, H, W)
                mask_ct  = mask_b[0, t]        # (H, W)

                if mask_ct.sum() == 0:
                    continue

                total_mass = mask_ct.sum()
                ys, xs = torch.meshgrid(
                    torch.arange(H, device=features.device),
                    torch.arange(W, device=features.device),
                    indexing='ij',
                )
                center_y = ((mask_ct * ys).sum() / total_mass).round().long().clamp(0, H - 1)
                center_x = ((mask_ct * xs).sum() / total_mass).round().long().clamp(0, W - 1)
                mu_pos   = F.normalize(feat_bt[:, center_y, center_x], dim=0)

                feats_flat = feat_bt.reshape(D, -1).T          # (H*W, D)
                mask_flat  = mask_ct.reshape(-1) > 0.5
                if mask_flat.sum() == 0:
                    continue
                pos_feats = F.normalize(feats_flat[mask_flat], dim=1)

                # Negative: average feature of non-masked region
                bg_mask = ~mask_flat
                if bg_mask.sum() == 0:
                    continue
                mu_neg  = F.normalize(feats_flat[bg_mask].mean(dim=0), dim=0)

                logit_pos = k_c * (pos_feats @ mu_pos)
                logit_neg = k_c * (pos_feats @ mu_neg)

                loss_pos = -torch.log(torch.sigmoid(logit_pos) + eps).mean()
                loss_neg = -torch.log(torch.sigmoid(-logit_neg) + eps).mean()
                losses.append(loss_pos + loss_neg)

        cam_loss_val = (
            torch.stack(losses).mean() if losses
            else torch.tensor(0.0, device=features.device)
        )
        return {'cam_loss': cam_loss_val}

    def forward(self, pred, batch, mode='train'):
        if mode == 'train':
            pred_mask   = pred['pred_mask']
            pseudo_mask = pred['pseudo_mask']
            device      = pred_mask.device
            query_cls   = batch['mask_index'].to(device)

            loss_dict = {}
            loss_dict.update(self.segmentation_loss(pred_mask, pseudo_mask))

            # CAM contrastive loss — enabled via args.cam_loss
            if getattr(self.args, 'cam_loss', False):
                feat = pred.get('feat')
                if feat is None:
                    raise KeyError(
                        "pred['feat'] is required for cam_loss but was not found. "
                        "Make sure shared_step adds feat_high to the pred dict."
                    )
                # pseudo_mask: (B, H, W) or (B, T, H, W) — normalise to (B, 1, T, H, W)
                pm = pseudo_mask
                if pm.dim() == 3:
                    # (B, H, W) -> (B, 1, 1, H, W) -> broadcast over T
                    T = feat.shape[2]
                    pm = pm.unsqueeze(1).unsqueeze(1).expand(-1, 1, T, -1, -1)
                elif pm.dim() == 4:
                    # (B, T, H, W) -> (B, 1, T, H, W)
                    pm = pm.unsqueeze(1)

                # Upsample pseudo_mask spatial dims to match feat if needed
                B, _, T_f, H_f, W_f = feat.shape
                if pm.shape[-2:] != (H_f, W_f):
                    pm = F.interpolate(
                        pm.reshape(B, T_f, H_f, W_f),   # wrong — do it properly below
                        size=(H_f, W_f), mode='bilinear', align_corners=False,
                    )
                    # Correct reshape: (B, 1, T, H, W) -> (B*T, 1, Hp, Wp) -> resize -> back
                    pm_orig = pseudo_mask.unsqueeze(1) if pseudo_mask.dim() == 4 else pseudo_mask.unsqueeze(1).unsqueeze(1).expand(-1, 1, T_f, -1, -1)
                    pm_rs = pm_orig.reshape(B * T_f, 1, pm_orig.shape[-2], pm_orig.shape[-1])
                    pm_rs = F.interpolate(pm_rs, size=(H_f, W_f), mode='bilinear', align_corners=False)
                    pm    = pm_rs.reshape(B, 1, T_f, H_f, W_f)

                loss_dict.update(self.cam_loss(feat, pm, query_cls))

            return loss_dict

        else:
            pred_mask  = pred['pred_mask']
            device     = pred_mask.device
            gt_mask    = batch['action_seg'].to(device)
            mask_index = batch['mask_index'].to(device)
            B, T, H, W = gt_mask.shape
            gt_mask_one_hot = F.one_hot(gt_mask.long(), num_classes=65).permute(0, 4, 1, 2, 3)
            gt_mask_one_hot = torch.gather(
                gt_mask_one_hot, 1,
                mask_index.view(B, 1, 1, 1, 1).expand(-1, -1, 16, H, W)
            ).squeeze(1).float()
            return calculate_metrics(pred_mask, gt_mask_one_hot)


# ====================================================================== #
#  Metrics                                                                 #
# ====================================================================== #

def calculate_metrics(pred, gt):
    """
    Args:
        pred: (B, T, H, W) logits
        gt:   (B, T, H, W) or (B, 1, T, H, W) binary ground truth
    """
    if gt.dim() == 5:
        gt = gt.squeeze(1)

    B, T, H, W = pred.shape
    pred = (torch.sigmoid(pred) > 0.5).float()
    gt   = (F.interpolate(gt.unsqueeze(1).float(), size=(T, H, W),
                          mode='trilinear', align_corners=False).squeeze(1) > 0.5).float()

    pred_mask = pred.bool()
    gt_mask   = gt.bool()

    fp_ratio     = (pred_mask.float().sum(dim=(1, 2, 3)) > 700).float().sum()
    intersection = (pred_mask & gt_mask).view(B, -1).sum(dim=1).float()
    union        = (pred_mask | gt_mask).view(B, -1).sum(dim=1).float()
    iou          = intersection / union.clamp(min=1e-6)

    tp = (pred_mask & gt_mask).view(B, -1).sum(dim=1).float()
    fp = (pred_mask & ~gt_mask).view(B, -1).sum(dim=1).float()
    fn = (~pred_mask & gt_mask).view(B, -1).sum(dim=1).float()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    tIoU = compute_temporal_iou_framewise(pred, gt, threshold=0.5)

    thresholds  = torch.arange(0.5, 1.0, 0.05).to(pred.device)
    APs_batch = []
    for b in range(B):
        APs = []
        for thresh in thresholds:
            TP = (tIoU[b] >= thresh).float()
            FP = 1 - TP
            APs.append(TP.sum() / (TP.sum() + FP.sum() + 1e-6))
        APs_batch.append(torch.stack(APs).mean())
    APs_batch = torch.tensor(APs_batch, device=pred.device)

    return {
        'iou':                  iou.sum(),
        'overall_intersection': intersection.sum(),
        'overall_union':        union.sum(),
        'precision':            precision.sum(),
        'recall':               recall.sum(),
        'f1':                   f1.sum(),
        'fp_ratio':             fp_ratio,
        'temporal_iou':         tIoU.sum(),
        'mAP@tIoU':             APs_batch.sum(),
    }


def compute_temporal_iou_framewise(pred, gt, threshold=0.5):
    """
    Args:
        pred, gt: (B, T, H, W) binary tensors
    Returns:
        tIoU_per_sample: (B,)
    """
    B, T, H, W = pred.shape
    pred_mask = pred.bool()
    gt_mask   = gt.bool()

    iou_per_frame = torch.zeros(B, T, device=pred.device)
    for t in range(T):
        inter = (pred_mask[:, t] & gt_mask[:, t]).flatten(1).sum(1).float()
        union = (pred_mask[:, t] | gt_mask[:, t]).flatten(1).sum(1).float()
        iou_per_frame[:, t] = inter / union.clamp(min=1e-6)

    TP_mask    = (iou_per_frame >= threshold).float()
    pred_active = pred_mask.view(B, T, -1).any(dim=2).float()
    gt_active   = gt_mask.view(B, T, -1).any(dim=2).float()

    TP = TP_mask * pred_active * gt_active
    FP = pred_active * (1 - TP_mask)
    FN = gt_active * (1 - pred_active)

    return TP.sum(dim=1) / (TP.sum(dim=1) + FP.sum(dim=1) + FN.sum(dim=1) + 1e-6)