# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        #nn.BCEWithLogitsLoss: This loss combines a Sigmoid layer and the BCELoss in one single class
        #BCE: Binary Cross Entropy between the target and the output
        #h['cls_pw']=h['obj_pw']=(1, 0.5, 2.0)
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        #tcls: classes
        #tbox: bounding boxes
        #indices: get the image index, anchor indices, grid coordinate (x,y) = 4 elements
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss()
        # p is the output of the model (3 detectors), len=3: [16(batch size),3(3anchors),80,80,85(1+4+80)], [16,3,40,40,85], [16,3,20,20,85]
        # input targets: [image(image id in the current batch),class(class id),x,y,w,h(normalized bounding box)],

        na, nt = self.na, targets.shape[0]  # number of anchors (na=3), number of targets in this batch
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain [1,1,1,1,1,1,1]

        #ai is the anchor index
        #torch.arange(na).float().view(na,1)
        # tensor([[0.],
        # [1.],
        # [2.]])
        #torch.arange(na).float().view(na,1).repeat(1,nt) repeat nt times in the second dimension
        # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        # [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]])
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        #targets.repeat(na,1,1).shape=[3, nt, 6], change from [nt,6] to [3, nt, 6]
        #ai[:,:,None].shape change from [3,nt] to [3,nt,1]
        #torch.cat in the second dimension become [3, nt, 7], added anchor index to each nt vector [image,class,x,y,w,h,ai(1,2,3 for each na)]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        #off is one offset matrix (5,2)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # off= tensor([[ 0.00000,  0.00000],
                    # [ 0.50000,  0.00000],
                    # [ 0.00000,  0.50000],
                    # [-0.50000,  0.00000],
                    # [ 0.00000, -0.50000]])

        for i in range(self.nl): # for every detector header, nl=3, sample rate from 8,16,32
            anchors = self.anchors[i] # [3,3,2] get the anchor for the current feature map, i-th one: [3,2]

            #p[i].shape=[16(batch size),3(3anchors),80,80,85(1+4+80)]
            #torch.tensor(p[i].shape)[[3, 2, 3, 2]] create [80,80,80,80]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain in the current layer, e.g., [1,1,80,80,80,80,1], 80,80 is the current layer feature resolution

            # Match targets to anchors
            t = targets * gain # convert normalized xywh to the current feature map (e.g., size=80x80),shape[3,nt,7]

            if nt:# if box exists, get every anchor for each box
                # Using shape to matche box with anchors, not IOU
                #anchors[:, None] shape change from [3,2] to [3,2,1]
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio to three different anchors [3,nt,2], w-box/w-anchor,h-box/h-anchor

                #torch.max(r,1./r).shape=[3,nt,2], 
                #torch.max Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim
                #torch.max(r, 1. / r).max(2)[0] get the values
                #model.hyp['anchor_t']=4
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare with threshold, return [3,nt] True/False
                # get all boxes that meets this requirements: 1/4 < box_wh/anchor_wh < 4
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                
                #if the box wh ratio over three anchors is large than threshold, filter out these boxes
                #only boxes matched with anchor remains, t shape change from [3, nt, 7] to [nt-filtered, 7]
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy stores box's xy coordinate relative to the upper-left corner of the feature map, shape [nt-filtered,2]
                gxi = gain[[2, 3]] - gxy  # inverse, use feature map size-gxy=(40-x,40-y) means the xy coordinate relative to the bottom-right corner

                # divide each grid box into 4 parts via g=0.5, j,k,l,m (True/False value) means the box in each 4 parts
                # First check box x,y coordinate > 1, and distance to the cell upper-left corner is <0.5
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) #j shape=[5,nf], stack 5 parts together

                #torch.repeat The number of times to repeat this tensor along each dimension
                t = t.repeat((5, 1, 1))[j]# t[nf,7] to [nf*3,7], each box must have three box cell (add two nearby cell for the anchor box matching)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]#[nf*3,2], generate offset for each box
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            #b means image id in this batch (size=nf*3), c is the class (size=nf*3)

            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            #get the grid cell xy indices
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices, size=nf*3

            # Append
            a = t[:, 6].long()  # get anchor indices for each box value=0,1,2, size=nf*3
            #save the image index, anchor indices, grid coordinate (x,y) = 4 elements
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            
            #get the delta xy relative to the current cell, and width, height
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch #3 elements each
