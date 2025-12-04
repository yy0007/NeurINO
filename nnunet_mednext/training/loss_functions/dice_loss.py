#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Callable
from unittest import result
import torch
from nnunet_mednext.training.loss_functions.TopK_loss import TopKLoss
from nnunet_mednext.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet_mednext.utilities.nd_softmax import softmax_helper
from nnunet_mednext.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
from skimage.morphology import skeletonize, dilation, ball  
from scipy.spatial import cKDTree 
import networkx as nx


class GDL(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, square_volumes=False):
        """
        square_volumes will square the weight term. The paper recommends square_volumes=True; I don't (just an intuition)
        """
        super(GDL, self).__init__()

        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(x.shape, y.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x)
            if x.device.type == "cuda":
                y_onehot = y_onehot.cuda(x.device.index)
            y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, self.square)

        # GDL weight computation, we use 1/V
        volumes = sum_tensor(y_onehot, axes) + 1e-6 # add some eps to prevent div by zero

        if self.square_volumes:
            volumes = volumes ** 2

        # apply weights
        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes

        # sum over classes
        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        # compute dice
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        dc = dc.mean()

        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc  # -dc


class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc
    
class SoftSkeletonLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_tube: bool = True, do_bg: bool = False, smooth: float = 1.0):  # ddp: bool = True
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super().__init__()

        if do_bg:
            raise RuntimeError("skeleton recall does not work with background")
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.do_tube = do_tube
        # self.ddp = ddp
        
    def skeletonize_batch(self, y):
        # y shape: [B, 1, D, H, W]
        device = y.device
        seg_all = y.detach().cpu().numpy().astype(np.uint8)
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
        
        # Skeletonize
        for b in range(seg_all.shape[0]):
            for c in range(seg_all.shape[1]):                                
                if not np.sum(bin_seg[b, c]) == 0:  
                    skel = (skel > 0).astype(np.int16)
                    if self.do_tube:
                        skel = dilation(skel, ball(1))
                        # skel = dilation(dilation(skel))
                    skel *= seg_all[b, c].astype(np.int16)  
                    seg_all_skel[b, c] = skel
    
        return torch.from_numpy(seg_all_skel).to(device).float() 

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape  # [B, 2, D, H, W], [B, 1, D, H, W]

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        x = x[:, 1:] 

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            y = self.skeletonize_batch(y)  # [B, 1, D, H, W], y_skel 
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)  
                y_onehot = y_onehot[:, 1:]
    
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        inter_rec = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        ske_loss = (inter_rec + self.smooth) / (torch.clip(sum_gt+self.smooth, 1e-8))

        ske_loss = ske_loss.mean()
        return -ske_loss 
    
class SkeletonGraphLoss(nn.Module): 
    def __init__(self, 
                 apply_nonlin=None,
                 do_tube=True,
                 lambda_node=1.0, 
                 lambda_edge=0.5, 
                 lambda_path=0.2,
                 thresh=0.5):
        super().__init__()
        self.do_tube = do_tube
        self.apply_nonlin = apply_nonlin
        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge
        self.lambda_path = lambda_path
        self.skeleton_thresh = thresh

    # ======= Skeletonize mask =======
    def skeletonize_batch(self, y):
        # y 的shape会是: [B, 1, D, H, W]
        device = y.device
        seg_all = y.detach().cpu().numpy().astype(np.uint8)
        # Add tubed skeleton GT
        bin_seg = (seg_all > 0)
        seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
        
        # Skeletonize
        for b in range(seg_all.shape[0]):
            for c in range(seg_all.shape[1]):                                
                if not np.sum(bin_seg[b, c]) == 0:  
                    skel = skeletonize(bin_seg[b, c])
                    skel = (skel > 0).astype(np.int16)
                    if self.do_tube:
                        skel = dilation(skel, ball(1))
                        # skel = dilation(dilation(skel))
                    skel *= seg_all[b, c].astype(np.int16)  
                    seg_all_skel[b, c] = skel
    
        return torch.from_numpy(seg_all_skel).to(device).float() 

    # ======= Convert skeleton to graph =======
    def _skeleton_to_graph(self, skel):
        """
        Convert binary skeleton mask to graph.
        Supports [1, 1, D, H, W] or [D, H, W]. 
        Nodes: branch points + endpoints
        Edges: continuous skeleton paths between nodes
        """
        if skel.ndim == 5:
            skel_np = skel.detach().cpu().numpy()[0, 0]
        elif skel.ndim == 3:
            skel_np = skel.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected input shape: {skel.shape}")
        
        G = nx.Graph()
        coords = np.argwhere(skel_np > 0)

        # Add nodes (all skeleton voxels)
        for i, c in enumerate(coords):
            G.add_node(i, pos=tuple(c))

        # Add edges based on voxel adjacency (6/26-connectivity)
        kd = cKDTree(coords)
        pairs = kd.query_pairs(r=2.0)  # within distance 1–2 voxel = connected
        for i, j in pairs:
            G.add_edge(i, j)

        return G, coords
    
    def _node_distance(self, gt_nodes, pred_nodes):
        if len(gt_nodes) == 0 or len(pred_nodes) == 0:
            return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")
        kd_gt = cKDTree(gt_nodes)
        kd_pred = cKDTree(pred_nodes)
        d1, _ = kd_gt.query(pred_nodes, k=1)
        d2, _ = kd_pred.query(gt_nodes, k=1)
        return torch.tensor((d1.mean() + d2.mean()) / 2.0)

    # ======= Compute edge continuity difference =======
    def _edge_distance(self, G_gt, G_pred):
        if G_gt.number_of_edges() == 0 or G_pred.number_of_edges() == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        edge_diff = abs(G_gt.number_of_edges() - G_pred.number_of_edges())
        norm_diff = edge_diff / (G_gt.number_of_edges() + 1e-6)
        return torch.tensor(norm_diff, dtype=torch.float32)

    # ======= Compute path-level distance =======
    def _path_distance(self, G_gt, G_pred, gt_nodes, pred_nodes):  # 局部断裂 / 连通性
        if len(G_gt.nodes) < 2 or len(G_pred.nodes) < 2:
            return torch.tensor(1.0)
        gt_lengths = [len(p) for p in nx.connected_components(G_gt)]
        pred_lengths = [len(p) for p in nx.connected_components(G_pred)]
        mean_diff = abs(np.mean(gt_lengths) - np.mean(pred_lengths)) / (np.mean(gt_lengths) + 1e-6)
        return torch.tensor(mean_diff, dtype=torch.float32)

    # ======= Forward =======
    def forward(self, net_output, target, loss_mask=None):    # 针对 binary segmentation 的 
        # net_output: [B, 2, D, H, W]
        # target: [B, 1, D, H, W]
        B, C = net_output.shape[:2] 
        device = net_output.device 
        if self.apply_nonlin is not None:
            prob_all = self.apply_nonlin(net_output)  # [B, 2, D, H, W] 
        else:
            prob_all = net_output

        # Foreground channel
        if prob_all.shape[1] > 1:
            prob = prob_all[:, 1:2]  # [B, 1, D, H, W] 
        else:
            prob = prob_all

        loss_batch = 0.0
        for b in range(B):
            with torch.no_grad():
                pred_binary = (prob > self.skeleton_thresh).float()
                gt_skel = self.skeletonize_batch(target[b:b+1])
                pred_skel = self.skeletonize_batch(pred_binary[b:b+1])

                G_gt, gt_nodes = self._skeleton_to_graph(gt_skel)
                G_pred, pred_nodes = self._skeleton_to_graph(pred_skel)

            D_node = self._node_distance(gt_nodes, pred_nodes).to(device)
            D_edge = self._edge_distance(G_gt, G_pred).to(device)
            D_path = self._path_distance(G_gt, G_pred, gt_nodes, pred_nodes).to(device)

            loss_batch += (self.lambda_node * D_node +
                    self.lambda_edge * D_edge +
                    self.lambda_path * D_path)
            
        # loss = loss_batch / B
        
        return loss_batch / B


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
    
class DC_CE_SkeletonLoss(nn.Module):  
    def __init__(self,
                 soft_dice_kwargs,
                 ce_kwargs,
                 skel_rec_kwargs, 
                 weight_ce=1.0,
                 weight_dice=1.0,
                 weight_skel=0.5, 
                 ignore_label=None):
        """
        Combines Dice + CE + Skeleton Recall Loss.
        """
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_skel = weight_skel
        self.ignore_label = ignore_label

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.skel_rec = SoftSkeletonLoss(apply_nonlin=softmax_helper, **skel_rec_kwargs)  # **soft_skelrec_kwargs

    def forward(self, net_output, target):
        # print('########## net_output shape:', net_output.shape)  # [2, 2, 32, 32, 32]
        if self.ignore_label is not None:
            # Mask out ignored voxels
            mask = (target != self.ignore_label).float()
            target = torch.where(mask.bool(), target, 0)
        else:
            mask = None
            
        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0 
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        skel_rec_loss = self.skel_rec(net_output, target, loss_mask=mask) if self.weight_skel != 0 else 0 
        total = self.weight_dice * dc_loss + self.weight_ce * ce_loss + self.weight_skel * skel_rec_loss
        return total
    
class DC_CE_SkeletonGraph_Loss(nn.Module):    
    def __init__(self,
                 soft_dice_kwargs,
                 ce_kwargs,
                 skel_graph_kwargs, 
                 weight_ce=1.0,
                 weight_dice=1.0,
                 weight_skel_graph=0.1,       # <—— 新增参数
                 ignore_label=None,
                 f_type='identity'):  # <—— f() 类型，可选 sigmoid/tanh
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.beta = weight_skel_graph
        self.ignore_label = ignore_label
        self.f_type = f_type

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.skel_graph = SkeletonGraphLoss(apply_nonlin=softmax_helper, **skel_graph_kwargs)  

    def f(self, x):
        if self.f_type == 'sigmoid':
            return torch.sigmoid(x)
        elif self.f_type == 'tanh':
            return torch.tanh(x)
        else:
            return x  # identity

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            mask = (target != self.ignore_label).float()
            target = torch.where(mask.bool(), target, 0)
        else:
            mask = None
            
        dc_loss = torch.abs(self.dc(net_output, target, loss_mask=mask)) if self.weight_dice != 0 else 0 
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        skel_graph_loss = self.skel_graph(net_output, target, loss_mask=mask)

        scale_factor = self.beta * self.f(skel_graph_loss) 
        total = (self.weight_dice * dc_loss + self.weight_ce * ce_loss) + scale_factor * (dc_loss + ce_loss)
        # print('dc_loss:', dc_loss)
        # print('ce_loss:', ce_loss)
        # print('total:', total)
        return total


class DC_and_CE_loss_with_class_wts(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss_with_class_wts, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'

        ce_kwargs['reduction'] = 'none'     # We do this anyway here

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, cl_weights=1):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        
        # print(ce_loss.shape, cl_weights.shape)
        ce_loss = ce_loss * cl_weights
        ce_loss = ce_loss.mean()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, aggregate="sum"):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.aggregate = aggregate
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(net_output, target)

        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result


class GDL_and_CE_loss(nn.Module):
    def __init__(self, gdl_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(GDL_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = GDL(softmax_helper, **gdl_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result


class DC_and_CE_and_ACS_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 weight_acs=1, log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_and_ACS_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_acs = weight_acs
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        
        self.acs = absolute_cosine_similarity

    def forward(self, net_output, target, feature_maps):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        acs_loss = self.acs(feature_maps)

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_acs * acs_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


def absolute_cosine_similarity(feature_maps, sub_patcher=True):

    result = 0.0
    for net_output in feature_maps:
        
        if len(net_output.shape) == 5:
            b, d, h, w, c = net_output.shape

        else:
            b, h, w, c = net_output.shape

        # print(net_output.shape)
        net_output = net_output / torch.linalg.vector_norm(net_output, ord=2, dim=-1, keepdim=True)
        net_output = net_output.reshape(b, -1, c)

        result = (net_output @ net_output.transpose(-1,-2)) - 1.0
        result +=  -1.0 * (torch.abs(result).mean())     # -1.0 lossifies it w.r.t. nnUNet
    return result