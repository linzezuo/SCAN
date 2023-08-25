# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath


class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        i_feats = 2 * n_feats
        self.scale = nn.Parameter(torch.zeros((1, i_feats, 1, 1)), requires_grad=True)
        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.act = nn.GELU()
        self.fc3 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)
        self.dw = nn.Conv2d(i_feats, i_feats, 3, 1, 1, groups=i_feats)
        self.fc2 = nn.Conv2d(i_feats, 1, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.scale * (x - self.act(self.fc2(x))) + x
        x = self.fc3(x)
        return x + shortcut


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GroupGLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 3 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA9 = nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3,
                              dilation=4)
        self.LKA7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3,
                              dilation=3)
        self.LKA5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3,
                              dilation=2)

        self.proj_first = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, 5, 1, (5 // 2), groups=self.n_feats))

        self.proj_last = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0))
        self.act = nn.SiLU()

        self.re = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0))

        self.last = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0))

        self.first = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, 1, 1, 0))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)

        x0 = self.first(x)
        x1 = self.gap(x0)
        x2 = self.scale * (x0 - x1) + x0
        x = self.gelu(x2)
        ###############################################################################33
        x3 = self.act(self.re(x))
        x = self.proj_first(x)
        a_1, a_2, a_3 = torch.split(x, [self.n_feats // 3, self.n_feats // 3, self.n_feats // 3], dim=1)
        a = torch.cat([self.LKA5(a_1), self.LKA7(a_2),
                       self.LKA9(a_3)], dim=1)
        x = self.act(self.proj_last(a)) * x3
        x = self.last(x) + shortcut
        return x


class MAB(nn.Module):
    def __init__(
            self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.LKA = GroupGLKA(self.n_feats)
        self.LFE = MLP(self.n_feats)

    def forward(self, x):
        # large kernel attention
        x = self.LKA(x)
        # local feature extraction
        x = self.LFE(x)

        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats):
        super(ResGroup, self).__init__()

        self.LKA9 = nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats,
                              dilation=4)
        self.body = nn.ModuleList([
            MAB(n_feats) \
            for i in range(n_resblocks)])

    def forward(self, x):
        res = x.clone()
        for i, block in enumerate(self.body):
            res = block(res)
        x = self.LKA9(res) + x

        return x


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


@ARCH_REGISTRY.register()
class SCAN(nn.Module):
    def __init__(self, n_resblocks=6, n_resgroups=6, n_colors=3, n_feats=180, scale=4):
        super(SCAN, self).__init__()

        # res_scale = res_scale
        self.n_resgroups = n_resgroups
        self.scale = scale
        self.sub_mean = MeanShift(1.0)
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # define body module
        self.body = nn.ModuleList([
            ResGroup(
                n_resblocks, n_feats)
            for i in range(n_resgroups)])

        self.body_t = nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats,
                                dilation=4)
        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        # self.tail = nn.Sequential(nn.Conv2d(n_feats, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        # self.upsample = Upsample(scale, 64)
        # self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.add_mean = MeanShift(1.0, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        y = F.interpolate(x, size=[x.shape[2] * self.scale, x.shape[3] * self.scale], mode="bilinear",
                          align_corners=False)
        x = self.head(x)
        res = x
        for i in self.body:
            res = i(res)
        res = self.body_t(res) + x

        x = self.tail(res)
        # x = self.conv_last(self.upsample(self.tail(res)))
        x = x + y
        x = self.add_mean(x)
        return x

    # def visual_feature(self, x):
    #     fea = []
    #     x = self.head(x)
    #     res = x
    #
    #     for i in self.body:
    #         temp = res
    #         res = i(res)
    #         fea.append(res)
    #
    #     res = self.body_t(res) + x
    #
    #     x = self.tail(res)
    #     return x, fea

# @ARCH_REGISTRY.register()
# class MAN_moga1(nn.Module):
#     def __init__(self, n_resblocks=6, n_resgroups=6, n_colors=3, n_feats=180, scale=4):
#         super(MAN_moga1, self).__init__()
#
#         # res_scale = res_scale
#         self.n_resgroups = n_resgroups
#         self.scale = scale
#         self.sub_mean = MeanShift(1.0)
#         self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
#
#         # define body module
#         self.body = nn.ModuleList([
#             ResGroup(
#                 n_resblocks, n_feats)
#             for i in range(n_resgroups)])
#
#         self.body_t = nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats,
#                                 dilation=4)
#         # define tail module
#         # self.tail = nn.Sequential(
#         #     nn.Conv2d(n_feats, n_colors * (scale ** 2), 3, 1, 1),
#         #     nn.PixelShuffle(scale)
#         # )
#         self.tail = nn.Sequential(nn.Conv2d(n_feats, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
#         self.upsample = Upsample(scale, 64)
#         self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
#         self.add_mean = MeanShift(1.0, sign=1)
#
#     def forward(self, x):
#         x = self.sub_mean(x)
#         y = F.interpolate(x, size=[x.shape[2] * self.scale, x.shape[3] * self.scale], mode="bilinear",
#                           align_corners=False)
#         x = self.head(x)
#         res = x
#         for i in self.body:
#             res = i(res)
#         res = self.body_t(res) + x
#         # x = self.tail(res)
#         x = self.conv_last(self.upsample(self.tail(res)))
#         x = x + y
#         x = self.add_mean(x)
#         return x
# #
# # def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
# #     tensor = tensor.float().cpu().clamp_(*min_max)
# #     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
# #     img_np = tensor.numpy()
# #     img_np = np.transpose(img_np, (1, 2, 0))
# #     if out_type == np.uint8:
# #         img_np = (img_np * 255.0).round()
# #
# #     return img_np.astype(out_type)
#
#
# if __name__ == '__main__':
#     # torch.cuda.empty_cache()
#     net = MAN_moga().cuda()
#     x = torch.randn((1, 3, 16, 16)).cuda()
#     x = net(x)
#     print(x.shape)
#     # import torch
#
#     # model_path = 'net_g_94000.pth'
#     # model = MAN().cuda()
#     # model.load_state_dict(torch.load(model_path)['params'])
#     # x = torch.randn((1, 3, 4, 4)).cuda()
#     # y = model(x)
#     # for parameters in model.parameters():
#     #     print(parameters,parameters.shape)
#
#     # path = 'net_g_94000.pth'
#     # LR = cv2.imread("baby.png", cv2.IMREAD_COLOR)
#     # LR = cv2.resize(LR, dsize=None, fx=0.25, fy=0.25)
#     # im_input = LR / 255.0
#     # im_input = np.transpose(im_input, (2, 0, 1))
#     # im_input = im_input[np.newaxis, ...]
#     # im_input = torch.from_numpy(im_input).float()
#     #
#     # # net = RepSR1(colors=3, module_nums=4, channel_nums=16, with_idt=False, act_type="prelu", scale=4)
#     # net = MAN()
#     # # net.load_state_dict(torch.load(path, map_location='cuda:0')["params"])
#     # net.load_state_dict(torch.load(path)['params'])
#     # # net.eval()
#     # SR = net(im_input)
#     # out_img = tensor2np(SR.detach()[0])
#     # cv2.imshow('sr', out_img)
#     # cv2.imwrite("2.png", out_img)
#     # cv2.waitKey(0)
