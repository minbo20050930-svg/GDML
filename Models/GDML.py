import torch.nn as nn
import torch.nn.functional as F
from kan import KANLinear
from archs import DW_bn_relu
from functools import partial
from .IIM_resnet34 import  IIM_resnet34
from timm.models.layers import trunc_normal_

nonlinearity = partial(F.relu, inplace=True)

import torch

from torch.nn.init import trunc_normal_


class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj =KANLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = DW_bn_relu(dim=576)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x.reshape(b*n,c))
        x = x.reshape(b,n,c).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x.reshape(b*n,c)).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v,H,W)

        x = self.proj(x.reshape(b*n,c))
        x = x.reshape(b,n,c)
        x = self.proj_drop(x)
        return x

class eca_layer(nn.Module):


    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def BNReLU(num_features):
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU()
    )


# ############################################## drop block ###########################################

class Drop(nn.Module):
    # drop_rate : 1-keep_prob  (all droped feature points)
    # block_size :
    def __init__(self, drop_rate=0.1, block_size=2):
        super(Drop, self).__init__()

        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):

        if not self.training:
            return x

        if self.drop_rate == 0:
            return x

        gamma = self.drop_rate / (self.block_size ** 2)
        # torch.rand(*sizes, out=None)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        mask = mask.to(x.device)

        # compute block mask
        block_mask = self._compute_block_mask(mask)
        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask



class IIM_DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, rla_channel=64, SE=False, ECA_size=5, reduction=16):
        super(IIM_DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels + rla_channel, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
        self.expansion = 1

        self.deconv_h = nn.ConvTranspose2d(rla_channel, rla_channel, 3, stride=2, padding=1, output_padding=1)
        self.deconv_x = nn.ConvTranspose2d(in_channels, n_filters, 3, stride=2, padding=1, output_padding=1)
        self.bn_=nn.BatchNorm2d(n_filters)
        self.relu_=nonlinearity
        self.se = None
        if SE:
            self.se = SELayer(n_filters * self.expansion, reduction)

        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(n_filters * self.expansion, int(ECA_size))

        self.conv_out = nn.Conv2d(n_filters, rla_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.recurrent_conv = nn.Conv2d(rla_channel, rla_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm4 = nn.BatchNorm2d(rla_channel)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        identity = x  # x.shape [2, 512, 7, 7],  print(h.shape) [4, 32, 7, 7]
        x = torch.cat((x, h), dim=1)  # [2, 576, 7, 7]

        out = self.conv1(x)  # [2, 128, 7, 7]
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.deconv2(out)  # [2, 128, 14, 14]
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)  # [2, 256, 14, 14]
        out = self.norm3(out)

        if self.se != None:
            out = self.se(out)

        if self.eca != None:
            out = self.eca(out)  # [2, 256, 14, 14]

        y = out  # [2, 256, 14, 14]

        identity = self.deconv_x(identity)  # [2, 512, 7, 7]--> [4, 256, 14, 14]
        identity = self.bn_(identity)
        identity = self.relu_(identity)
        out += identity
        out = self.relu3(out)  # [4, 256, 14, 14]

        y_out = self.conv_out(y)  # [4, 32, 14, 14]
        h = self.deconv_h(h)  # [4, 32, 14, 14]
        h = h + y_out  # [4, 32, 14, 14]
        h = self.norm4(h)
        h = self.tanh(h)
        # h = self.recurrent_conv(h)    # This convolution is not necessary

        return out, h


class GDML(nn.Module):
    def __init__(self, classes=2, channels=3):
        super(GDML, self).__init__()

        self.rla_channel = 32
        filters = [64, 128, 256, 512]
        self.model = IIM_resnet34()

        self.flat_layer = AgentAttention(dim=576,num_patches=49,num_heads=8)
        self.drop_block = Drop(drop_rate=0.2, block_size=2)

        self.decoder4 = IIM_DecoderBlock(512, filters[2], ECA_size=7)  # 7
        self.decoder3 = IIM_DecoderBlock(filters[2], filters[1], ECA_size=5)
        self.decoder2 = IIM_DecoderBlock(filters[1], filters[0], ECA_size=5)
        self.decoder1 = IIM_DecoderBlock(filters[0], filters[0], ECA_size=5)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0] + 64, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv3 = nn.Conv2d(4, classes, 1, stride=1)

        self.finaldeconv2 = nn.ConvTranspose2d(filters[0] , 32, 4, 2, 1)
        self.finalrelu2 = nonlinearity
        self.finalconv4 = nn.Conv2d(32, classes, 3, padding=1)
    def forward(self, x):
        # Encoder
        e1, e2, e3, e4, e_h1, e_h2, e_h3, e_h4 = self.model(x)
        # Center

        flat_feature = torch.cat((e4, e_h4), dim=1)
        B,C,H,W = flat_feature.size()
        flat_feature = flat_feature.view(B,-1,C)
        flat_feature = self.flat_layer(flat_feature,H,W)
        flat_feature = flat_feature.view(B,-1,H,W)
        flat_feature = self.drop_block(flat_feature)
        e4_flat, eh4_flat = torch.split(flat_feature, [512, 64], dim=1)

        # Decoder

        dh_4 = eh4_flat

        d3, dh_3 = self.decoder4(e4_flat, dh_4)
        d3 = d3 + e3
        dh_3 = dh_3 + e_h3

        d2, dh_2 = self.decoder3(d3, dh_3)
        d2 = d2 + e2
        dh_2 = dh_2 + e_h2

        d1, dh_1 = self.decoder2(d2, dh_2)
        d1 = d1 + e1
        dh_1 = dh_1 + e_h1

        d0, dh_0 = self.decoder1(d1, dh_1)
        out1 = d0  #[4,64,112,112]
        out2 = dh_0  #[4,64,112,112]
        d0_out = torch.cat((d0, dh_0), dim=1)#[4,128,112,112]


        out1 = self.finaldeconv2(out1)
        out1 = self.finalrelu2(out1)
        out1 = self.finalconv4(out1)

        out2 = self.finaldeconv2(out2)
        out2 = self.finalrelu2(out2)
        out2 = self.finalconv4(out2)

        out = torch.cat((out1, out2), dim=1)
        out = self.finalconv3(out)


        return F.sigmoid(out),F.sigmoid(out1),F.sigmoid(out2)

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224).cuda()
    model = GDML().cuda()
    out12,out121,out122 = model(input)
    print(out12.shape)
    print(out121.shape)
    print(out122.shape)