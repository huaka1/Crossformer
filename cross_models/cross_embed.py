import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        # 分段 （32, 168, 7）->（6272, 6）
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment) #（6272, d_model）
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim) # (32, 7, 28, 256)
        
        return x_embed