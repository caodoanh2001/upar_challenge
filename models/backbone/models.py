import warnings
from collections import OrderedDict
from copy import deepcopy
import logging

import math 
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
import cv2

from torch.nn import Module as BaseModule
from torch.nn import ModuleList
from torch.nn import Sequential
from torch.nn import Linear
from torch import Tensor
from mmcv.runner import load_checkpoint as _load_checkpoint

from itertools import repeat
import collections.abc
from einops import rearrange
from models.registry import BACKBONE
import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x, pre_logits=None):
        return x

class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x
    
class DefineModel(BaseModule):
    def __init__(self,):
        super(DefineModel, self).__init__()

        # N Transformer blocks
        vit = timm.create_model('vit_large_r50_s32_384', pretrained=True)
        self.vit_blocks = nn.ModuleList([*vit.blocks][-3:])
        
        # ConvNeXt
        self.cnn_model = timm.create_model('convnext_base.fb_in1k', pretrained=True, num_classes=0)
        self.cnn_model.head = Identity()
        
        # SwinT
        self.swint_model = timm.create_model('swinv2_base_window8_256.ms_in1k', img_size=(256, 128), pretrained=True, num_classes=0)
        self.swint_model.head = Identity()

        self.norm_1 = nn.BatchNorm2d(1024)
        self.norm_2 = nn.BatchNorm2d(1024)

    def forward(self, x):
        out = self.swint_model(x)
        out1 = self.cnn_model(x)
        h, w = out.shape[1:3]
        out = rearrange(out, 'b h w c -> b c h w', h=h, w=w)
                
        out = self.norm_1(out)
        out1 = self.norm_2(out1)

        out = rearrange(out, 'b c h w -> b (h w) c', h=h, w=w)
        out1 = rearrange(out1, 'b c h w -> b (h w) c', h=h, w=w)

        out2 = torch.cat([out, out1], dim=1)

        for block in self.vit_blocks:
            out2 = block(out2)

        return out2