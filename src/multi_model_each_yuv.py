# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




class Neulf4D_single(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, output_ch=1, skips=[4,8,12,16,20],depth_branch=False, input_dim = 4):
        """ 
        """
        #print("new model 1031!!!!!!!!!!!!!!")
        super(Neulf4D_single, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch
        self.skips = np.arange(4, D, 4)

        # New model
        self.input_net =  nn.Linear(input_dim, input_ch)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.rgb_linear = nn.Linear(W, 1)
        self.rgb_act   = nn.Sigmoid()

    def forward(self, x):
        input_pts = self.input_net(x)
        input_pts = F.relu(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)


        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)
        return rgb
    
class Neulf4D_rgb(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, output_ch=1, input_dim=4):
        super(Neulf4D_rgb, self).__init__()
        # 3개의 Neulf4D_single 인스턴스 생성
        self.single_models = nn.ModuleList([Neulf4D_single(D, W, input_ch, output_ch, input_dim) for _ in range(3)])
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=0.1 , betas=(0.9, 0.999)) for model in self.single_models]
        self.schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)  for optimizer in self.optimizers]


class GridYUVNetworks(nn.Module):
    def __init__(self, grid_size, D=4, W=64, input_ch=64, output_ch=1, input_dim=4):
        super(GridYUVNetworks,self).__init__()
        self.networks = nn.ModuleList([Neulf4D_rgb(D=D, W=W, input_ch=input_ch, output_ch=output_ch, input_dim=input_dim) for _ in range(grid_size**2)])
        
