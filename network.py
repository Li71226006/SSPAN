import torch.nn as nn
from torch.nn import functional as F
import torch

from backbone import ResNet101, Bottleneck



def get_backbone():
    model = ResNet101(Bottleneck,[3, 4, 23, 3])
    return model

def patch_split(input, bin_size):
    """
    B C H W -> B S^2(bin_size*bin_size) rH rW C
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size
    bin_num_w = bin_size
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0,2,4,3,5,1).contiguous()
    out = out.view(B,-1,rH,rW,C)
    return out

def patch_recover(input, bin_size):
    """
    B S^2(bin_size*bin_size) rH rW C -> B C H W
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size
    bin_num_w = bin_size
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous()
    out = out.view(B, C, H, W)
    return out


class Conv_m(nn.Module):
    def __init__(self, feature_in, feature_out, K_size, Bias, Padding=0):
        super(Conv_m,self).__init__()
        self.conv = nn.Conv2d(feature_in, feature_out, kernel_size=K_size,padding=Padding,bias=Bias)
        self.norm_layer = nn.BatchNorm2d(feature_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.relu(x)
        return x 


class LG(nn.Module):
    def __init__(self, num_1, num_2):
        super(LG, self).__init__()
        self.conv1 = nn.Conv2d(num_1, num_1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_2, num_2, bias=False)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out

class DROPOUT(nn.Module):
    def __init__(self, feature_in,feature_out):
        super(DROPOUT, self).__init__()
        self.drop = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(feature_in,feature_out, kernel_size=1, bias=False)
    def forward(self, x):
        out = self.drop(x)
        out = self.conv(out)
        return out

class FUSE(nn.Module):
    def __init__(self, num_chanel):
        super(FUSE, self).__init__()
        self.conv = nn.Conv2d(num_chanel, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        return out

class POOL(nn.Module):
    def __init__(self, bin_size):
        super(POOL, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.pool(x)
        out = self.sigmoid(out)
        return out


class Segmentation_Head(nn.Module):
        def __init__(self,patch_size,n_classes):
            super(Segmentation_Head, self).__init__()
            feature_in = 512
            feature_inner = feature_in // 2
            self.patch_size = patch_size
            patch_num = patch_size * patch_size
            self.LG = LG(patch_num,feature_in)
            self.Fuse = FUSE(patch_num)
            self.Query = nn.Linear(feature_in, feature_inner)
            self.Key = nn.Linear(feature_in, feature_inner)
            self.Value = nn.Linear(feature_in, feature_inner)
            self.conv1 = Conv_m(feature_inner, feature_in, K_size=1, Bias=False)
            self.conv2 = Conv_m(feature_in,feature_in, K_size=3, Padding=1, Bias=False)
            self.conv3 = Conv_m(2048+feature_in,feature_in, K_size=3, Padding=1, Bias=False)
            self.drop = DROPOUT(feature_in,n_classes)


        def forward(self, M_1,F,I,R):
            F_p = patch_split(F,self.patch_size) # [B, S * S, rH, rW, C4]
            I_p = patch_split(I,self.patch_size) # [B, S * S, rH, rW, K]
            residual = F # [B, C4, H, W]
            B, _, rH, rW, C4 = F_p.shape
            K = I_p.shape[-1]
            F_p = F_p.view(B, -1, rH*rW, C4) # [B, S * S, rH * rW, C4]
            I_p = I_p.view(B, -1, rH*rW, K) # [B, S * S, rH * rW, K]

            bin_confidence = R.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, S * S, K, 1]
            pixel_confidence = torch.softmax(I_p, dim=2)  # [B, S * S, rH * rW, K]
            F_l = torch.matmul(pixel_confidence.transpose(2, 3), F_p) * bin_confidence  # [B, S * S, K, C4]
            F_l = self.LG(F_l) # [B, S * S, K, C4]
            F_g = self.Fuse(F_l).repeat(1, I_p.shape[1], 1, 1) # [B, S * S, K, C4]

            query = self.Query(F_p) # [B, S * S rH * rW, C4//2]
            key = self.Key(F_l) # [B, S * S, K, C4//2]
            value = self.Value(F_g)# [B, S * S, K, C4//2]

            aff_map = torch.matmul(query, key.transpose(2, 3)) # [B, S * S, rH * rW, K]
            aff_map = torch.softmax(aff_map, dim=-1)
            F_s = torch.matmul(aff_map, value) # [B, S * S, rH * rW, C4//2]
            F_s = F_s.view(B, -1, rH, rW, value.shape[-1])
            F_s = patch_recover(F_s,self.patch_size) # [B, C4//2, H, W]
            Fo1 = residual + self.conv1(F_s) # [B, C4, H, W]
            out = self.conv3(torch.cat([self.conv2(Fo1), M_1], dim=1)) # [B, K, H, W]
            out = self.drop(out) # [B, K, H, W]
            return out



class Main_branch(nn.Module):
        def __init__(self,patch_size,n_classes):
            super(Main_branch, self).__init__()
            self.backbone = get_backbone()
            self.conv1 = Conv_m(self.backbone.M1_channels, self.backbone.M1_channels//4, K_size=3, Padding=1, Bias=False)
            self.conv2 = Conv_m(self.backbone.M2_channels, self.backbone.M2_channels//4, K_size=3, Padding=1, Bias=False)
            self.drop1 = DROPOUT(self.backbone.M1_channels//4, n_classes)
            self.drop2 = DROPOUT(self.backbone.M2_channels//4, n_classes)
            self.pool = POOL(patch_size)

        def forward(self,x):
            M2,M1 = self.backbone(x) # M1:[B, C1, H, W],M2:[B, C2, H, W]
            F =  self.conv1(M1) # [B, C4, H, W]
            I = self.drop1(F) # [B, K, H, W]
            R = self.pool(I) # [B, K, S, S]
            P = self.drop2(self.conv2(M2)) # [B, K, H, W]
            return M1,F,I,R,P




class SSPAN(nn.Module):
    def __init__(self, num_classes=3, patch_size=4):
        super(SSPAN, self).__init__()
        self.main = Main_branch(patch_size,num_classes)
        self.head = Segmentation_Head(patch_size,num_classes)

    def forward(self, x):
        M1,F,I,R,P = self.main(x)
        out = self.head(M1,F,I,R)
        return (R,P,out)

        


if __name__ == '__main__':

    net = SSPAN()
    input = torch.normal(mean=0.,std=1.,size=(1,3,480,640))
    R,P,out = net(input)
    print("R_shape:{},P_shape:{},out_shape:{}".format(R.shape,P.shape,out.shape))


