"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""
import numbers
import torch
import torch.nn as nn
from subpixel import shuffle_down, shuffle_up###################
import torch.nn.functional as F
from pdb import set_trace as stx
from Blanced_attention import BlancedAttention, BlancedAttention_CAM_SAM_ADD
from coordatt import CoordAtt
from torchvision import transforms as trans
from einops import rearrange
#import numpy as np
#from wavelet import wt,iwt
#from pywt import dwt2, wavedec2
#from lifting import WaveletHaar2D, LiftingScheme2D, Wavelet2D
##########################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return [x_LL, x_HL, x_LH, x_HH]#torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False 

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
##########################################################################
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def st_conv(in_channels, out_channels, kernel_size, bias=False, stride = 2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)		
##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
##########################################################################
## S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat*2, n_feat)
        #self.CON_FEA = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        #self.CA_fea = BlancedAttention_CAM_SAM_ADD(n_feat, reduction)
        #self.CA_fea = CCALayer(n_feat, reduction, bias=bias)
    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1,x2), 1))
        #FEA_2 = self.CON_FEA(torch.cat((x1,x2), 1))
        #resin = FEA_1 + FEA_2
        res= self.CA_fea(FEA_1) + x1
        #res += resin
        return res#x1 + resin
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
		
# contrast-aware channel attention module
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
	
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
        
        
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

###########################################################################
### Gated-Dconv Feed-Forward Network (GDFN)
#class FeedForward(nn.Module):
#    def __init__(self, dim, ffn_expansion_factor, bias):
#        super(FeedForward, self).__init__()
#        self.act1 = nn.PReLU()
#        hidden_features = int(dim*ffn_expansion_factor)
#
#        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#
#        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#
#        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#    def forward(self, x):
#        x = self.project_in(x)
#        x1, x2 = self.dwconv(x).chunk(2, dim=1)
#        x = self.act1(x1) * x2
#        x = self.project_out(x)
#        return x
#
#
#
###########################################################################
### Multi-DConv Head Transposed Self-Attention (MDTA)
#class Attention(nn.Module):
#    def __init__(self, dim, num_heads, bias):
#        super(Attention, self).__init__()
#        self.num_heads = num_heads
#        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        
#        #self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
#        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#        
#
#
#    def forward(self, k_fea, v_fea, q_fea):
#        b,c,h,w = q_fea.shape
#        q = self.q(q_fea)
#        k = self.k(k_fea)
#        v = self.v(v_fea)
#        #qkv = self.qkv_dwconv(self.qkv(x))
#        #q,k,v = qkv.chunk(3, dim=1)   
#        
#        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#        q = torch.nn.functional.normalize(q, dim=-1)
#        k = torch.nn.functional.normalize(k, dim=-1)
#
#        attn = (q @ k.transpose(-2, -1)) * self.temperature
#        attn = attn.softmax(dim=-1)
#
#        out = (attn @ v)
#        
#        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#        out = self.project_out(out)
#        return out
##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)#nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        #self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        #self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    # def forward(self, x):
        # b, c, h, w = x.shape

        # qkv = self.qkv_dwconv(self.qkv(x))
        # q, k, v = qkv.chunk(3, dim=1)
    def forward(self, k_fea, v_fea, q_fea):
        b,c,h,w = q_fea.shape
        q = self.q(q_fea)
        k = self.k(k_fea)
        v = self.v(v_fea) 
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=k_fea.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##  Sparse Transformer Block (STB) 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_key = LayerNorm(dim, LayerNorm_type)
        self.norm_query = LayerNorm(dim, LayerNorm_type)
        self.norm_value = LayerNorm(dim, LayerNorm_type)
        
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, in1, in2):
        # x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))
        x = in2 + self.attn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))
        x = x + self.ffn(self.norm2(x))
        return x
##########################################################################
class h_sigmoid(nn.Module):
    #def __init__(self, inplace=True):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
############  V (H) attention Layer
class CoordAtt_V(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_V, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))#nn.AdaptiveAvgPool2d((None, 1)),for training

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h_pool = self.pool_h(x)##n,c,h,1
        y = self.conv1(x_h_pool)
        y_bn = self.bn1(y)
        y_act = self.act(y_bn) 
		
        a_h_att = self.conv_h(y_act).sigmoid()


        out = identity * a_h_att

        return out
############  H (W) attention Layer
class CoordAtt_H(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_H, self).__init__()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        #x_h = self.pool_h(x)
        x_w_pool = self.pool_w(x).permute(0, 1, 3, 2)##n,c,W,1
        y = self.conv1(x_w_pool)
        y_bn = self.bn1(y)
        y_act = self.act(y_bn) 
        
        y_act_per = y_act.permute(0, 1, 3, 2)
        a_w_att = self.conv_w(y_act_per).sigmoid()

        out = identity * a_w_att

        return out
        
        
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = res + x
        return res
        
##########################################################################
## Channel Attention Block (CAB)
class CAB1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB1, self).__init__()
        # modules_body1 = []
        # modules_body1.append(conv(n_feat, n_feat//2, kernel_size, bias=bias))
        # modules_body1.append(act)
        # modules_body1.append(conv(n_feat//2, n_feat, kernel_size, bias=bias))
        # modules_body2 = []
        # modules_body2.append(conv(n_feat, n_feat//2, kernel_size, bias=bias))
        # modules_body2.append(act)
        # modules_body2.append(conv(n_feat//2, n_feat, kernel_size, bias=bias))
        self.conv_ca1 = conv(n_feat, n_feat//2, kernel_size, bias=bias)
        self.conv_ca2 = conv(n_feat//2, n_feat//2, kernel_size, bias=bias)
        self.conv_ca3 = conv(n_feat//2, n_feat//2, kernel_size, bias=bias)
        self.conv_ca4 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        # self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        # res1 = self.body1(x)
        # res2 = self.body2(x)
        fea1 = self.conv_ca1(x)
        fea2 = self.conv_ca2(fea1)
        fea3 = self.conv_ca3(fea1)
        res = self.conv_ca4(torch.cat((fea2, fea3), 1))
        res = self.CA(res)
        res = res + x
        return res
		
## Enhanced Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(ECAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
		
## Channel Attention Block (CAB)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = []
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))
        modules_body.append(act)
        modules_body.append(depthwise_separable_conv(n_feat, n_feat))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        #res += x
        return res
        
class CAB1_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB1_dsc, self).__init__()
        # modules_body1 = []
        # modules_body1.append(conv(n_feat, n_feat//2, kernel_size, bias=bias))
        # modules_body1.append(act)
        # modules_body1.append(conv(n_feat//2, n_feat, kernel_size, bias=bias))
        # modules_body2 = []
        # modules_body2.append(conv(n_feat, n_feat//2, kernel_size, bias=bias))
        # modules_body2.append(act)
        # modules_body2.append(conv(n_feat//2, n_feat, kernel_size, bias=bias))
        self.conv_ca1 = depthwise_separable_conv(n_feat, n_feat//2)
        self.conv_ca2 = depthwise_separable_conv(n_feat//2, n_feat//2)
        self.conv_ca3 = depthwise_separable_conv(n_feat//2, n_feat//2)
        self.conv_ca4 = depthwise_separable_conv(n_feat, n_feat)
        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.CA = CoordAtt(n_feat, n_feat)
        # self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        # res1 = self.body1(x)
        # res2 = self.body2(x)
        fea1 = self.conv_ca1(x)
        fea2 = self.conv_ca2(fea1)
        fea3 = self.conv_ca3(fea1)
        res = self.conv_ca4(torch.cat((fea2, fea3), 1))
        res = self.CA(res)
        res = res + x
        return res
##########################################################################
## Supervised Attention Module
# class SAM(nn.Module):
    # def __init__(self, n_feat, kernel_size, bias):
        # super(SAM, self).__init__()
        # self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        # self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        # self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    # def forward(self, x, x_img):
        # x1 = self.conv1(x)
        # img = self.conv2(x) + x_img
        # x2 = torch.sigmoid(self.conv3(img))
        # x1 = x1*x2
        # x1 = x1+x
        # return x1, img
##########################################################################
##########################################################################
#class TransformerBlock(nn.Module):
#    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#        super(TransformerBlock, self).__init__()
#
#        self.norm_key = LayerNorm(dim, LayerNorm_type)
#        self.norm_query = LayerNorm(dim, LayerNorm_type)
#        self.norm_value = LayerNorm(dim, LayerNorm_type)
#        
#        self.attn = Attention(dim, num_heads, bias)
#        self.norm2 = LayerNorm(dim, LayerNorm_type)
#        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#
#    def forward(self, in1, in2):
#        # print('in1', in1.shape)
#        # print('in2', in2.shape)
#        # a = self.norm_key(in1)
#        # b = self.norm_query(in2)
#        # print('norm_key(in1)', a.shape)
#        # print('norm_query(in2)', b.shape)
#        x = in2 + self.attn(self.norm_key(in1),self.norm_value(in1),self.norm_query(in2))
#        x = x + self.ffn(self.norm2(x))
#
#        return x
        
class STEM_att(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=False):
        super(STEM_att, self).__init__()
        # global average pooling: feature --> point
        
        act=nn.PReLU()
        #num_blocks = 1
        heads = 4
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  ## Other option 'BiasFree'

        #self.down_img = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)#DownSample1()
        self.conv_stem0 = conv(n_feat, n_feat//2, kernel_size=1, bias=bias)
        #self.conv_stem1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv_stem2 = conv(n_feat, n_feat//2, kernel_size=1, bias=bias)
        self.former = TransformerBlock(dim=n_feat//2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        # self.conv_stem3 = conv(n_feat, n_feat//2, kernel_size, bias=bias)
        # self.S2FB = S2FB_2(n_feat, reduction, bias=bias, act=act)
        #self.CA_fea = CALayer(n_feat, reduction, bias=bias)
        self.conv_stem3 = conv(n_feat//2, n_feat, kernel_size=1, bias=bias)
        
    def forward(self, input1, input2):
        res_1 = self.conv_stem0(input1)
        res_2 = self.conv_stem2(input2)
        att_fea = self.conv_stem3(self.former(res_2, res_1))
        return att_fea
        #return torch.cat([img_down_fea * rain_mask, img_fea],1) 
##########################################################################
## Direction Attention Block (DAB)
class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(DAB, self).__init__()

        self.Main_fea = CAB(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.V_fea = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.H_fea = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        self.HH_fea = CAB_dsc(n_feat,kernel_size, reduction, bias=bias, act=act)
        
        self.V_ATT = CoordAtt_V(n_feat, n_feat)
        self.H_ATT = CoordAtt_H(n_feat, n_feat)
        self.HH_ATT = CoordAtt(n_feat, n_feat)
        
        self.conv_fuse1 = conv(n_feat*3, n_feat, kernel_size=1, bias=bias)
        self.conv_fuse2 = conv(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_fuse3 = conv(n_feat, n_feat, kernel_size=1, bias=bias)
        # self.conv_fuse4 = conv(n_feat, n_feat, kernel_size=1, bias=bias)
        # self.conv_fuse5 = conv(n_feat, n_feat, kernel_size=1, bias=bias)
        
        self.STEM_att12 = STEM_att(n_feat, kernel_size, bias=bias)
        self.STEM_att13 = STEM_att(n_feat, kernel_size, bias=bias)
        self.STEM_att14 = STEM_att(n_feat, kernel_size, bias=bias)
        # self.FB_1 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        # self.FB_2 = S2FB_2(n_feat, reduction, bias=bias, act=act)
        
    def forward(self, x):#[x_main_rain, x_V_fea, x_H_fea]
        main_r_fea = self.Main_fea(x[0])
        main_b_fea = x[0] - main_r_fea
        v_ATT_fea = self.V_fea(self.V_ATT(main_r_fea))
        h_ATT_fea = self.H_fea(self.H_ATT(main_r_fea))
        hh_ATT_fea = self.HH_fea(self.HH_ATT(main_r_fea))
        
        # fea_v = self.conv_fuse1(self.STEM_att12(v_ATT_fea, x[1]))
        # fea_h = self.conv_fuse2(self.STEM_att13(h_ATT_fea, x[2]))
        # fea_hh = self.conv_fuse3(self.STEM_att14(hh_ATT_fea, x[3]))
        
        v_feas = x[1] - v_ATT_fea
        h_feas = x[2] - h_ATT_fea
        hh_feas = x[3] - hh_ATT_fea #- self.conv_fuse3(fea_v + fea_h)
        fea_fuse = self.conv_fuse1(torch.cat([v_feas, h_feas, hh_feas],1))
        # v_feas = self.V_fea(x[1] - self.FB_1(v_ATT_fea, x[1]))
        # h_feas = self.H_fea(x[2] - self.FB_2(h_ATT_fea, x[2]))
        fea_feedback = self.conv_fuse2(self.STEM_att12(fea_fuse, main_b_fea))
        # v_feas = self.V_fea(x[1]) - v_ATT_fea 
        # h_feas = self.H_fea(x[2]) - h_ATT_fea
        return [fea_feedback, v_feas, h_feas, hh_feas]
		
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [DAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        #modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):#[x_main_rain, x_V_fea, x_H_fea]
        # x_main = x[0]
        # x_V = x[1]
        # x_H = x[2]
        res = self.body(x)
        #res = res + x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORSNet, self).__init__()
        act=nn.PReLU()
		
        self.shallow_feat1 = nn.Sequential(conv(3*4, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat4 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        
        self.orb = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
		
        self.tail_main_LL = conv(n_feat, 3, kernel_size, bias=bias)
        # self.tail_main_HH = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail_V_fea = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail_H_fea = conv(n_feat, 3, kernel_size, bias=bias)
        self.tail_HH_fea = conv(n_feat, 3, kernel_size, bias=bias)
        
    def forward(self, x):#x_LL, x_HL, x_LH, x_HH
        # H = x.size(2)
        # W = x.size(3)
		############ WaveletHaar2D, DWT
        x_LL = x[0]
        x_V = x[1]
        x_H = x[2]
        x_HH = x[3]
		############ wt
        # x_LL = x[:,0:3,:,:]#x[0]
        # x_V = x[:,3:6,:,:]#x[1]##Vertical
        # x_H = x[:,6:9,:,:]#x[2]##Horizontal
        # x_HH = x[:,9:12,:,:]#x[3]
		
        x_main_rain = self.shallow_feat1(torch.cat([x_LL, x_V, x_H, x_HH],1))
        x_V_fea = self.shallow_feat2(x_V)
        x_H_fea = self.shallow_feat3(x_H)
        x_HH_fea = self.shallow_feat4(x_HH)
        
        x_out= self.orb([x_main_rain, x_V_fea, x_H_fea, x_HH_fea])
		
        #x_main = self.tail_main_rain(x_out[0])
        x_LL_img = self.tail_main_LL(x_out[0])
        # x_HH_img = self.tail_main_HH(x_out[0])
        #x_LL_rain, x_HH_rain = torch.split(x_main, (3, 3), dim=1)

        x_V_img = self.tail_V_fea(x_out[1])
        x_H_img = self.tail_H_fea(x_out[2])
        x_HH_img = self.tail_HH_fea(x_out[3])
		
        x_cat = torch.cat((x_LL_img, x_V_img, x_H_img, x_HH_img), 1)
        return x_cat

##########################################################################
class DAWN(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_cab=15, bias=False):
        super(DAWN, self).__init__()

        act=nn.PReLU()
        #self.wavelet1 = WaveletHaar2D()
        self.dwt = DWT()
        self.orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.iwt = IWT()
		
    def forward(self, img): #####b,c,h,w
        #print(x_img.shape)
        #wave_out1 = wt(img)
        dwt_fea = self.dwt(img)
        #print(dwt_fea)
        #wave_out1 = self.wavelet1(img)
        orsnet_out = self.orsnet(dwt_fea)
        iwt_fea = self.iwt(orsnet_out)#iwt(orsnet_out)
        return iwt_fea#, orsnet_out[0], orsnet_out[0], orsnet_out[0]
