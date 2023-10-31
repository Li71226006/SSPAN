import torch.nn as nn
import torch
from torch.nn import functional as F
from ssim import SSIM

def get_bin_label(self, label_onehot, bin_size, th=0.01):
    cls_percentage = F.adaptive_avg_pool2d(label_onehot, bin_size)
    cls_label = torch.where(cls_percentage>0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage))
    cls_label[(cls_percentage<th)&(cls_percentage>0)] = self.ignore_index
    return cls_label

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.crit = nn.BCELoss(reduction='none')

    def binary_focal_loss(self, input, target, valid_mask):
        input = input[valid_mask]
        target = target[valid_mask]
        pt = torch.where(target == 1, input, 1 - input)
        ce_loss = self.crit(input, target)
        loss = torch.pow(1 - pt, self.gamma) * ce_loss
        loss = loss.mean()
        return loss
        
    def	forward(self, input, target):
        valid_mask = (target != self.ignore_index)
        K = target.shape[1]
        total_loss = 0
        for i in range(K):
            total_loss += self.binary_focal_loss(input[:,i], target[:,i], valid_mask[:,i])
        return total_loss / K

class TotalLoss(nn.Module):

    def __init__(self, ignore_index=255, reduction='mean', target_onehot, patch_size):
        super(DSNLoss, self).__init__()
        self.get_bin_label(target_onehot, bin_size)
        self.ignore_index = ignore_index
        self.region_loss = FocalLoss(ignore_index=ignore_index)
        self.pixel_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        self.ssim = SSIM()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        region_gt = self.get_bin_label(target_onehot, patch_size)
        l_r = self.region_loss(preds[0], region_gt)

        scale_pred_p = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=False)
        l_p = self.pixel_loss(scale_pred_p, target)

        scale_pred = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=False)
        l_ce = self.ce(scale_pred, target)
        l_ssim = self.ssim(scale_pred,target)


        total_loss = (l_p + l_r*0.4)+(l_ce+l_ssim)

            return total_loss


