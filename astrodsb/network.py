# --------------------------------------------------------------------------------------------------
# Core code for Astro-DSB for astrophysical observational inversion, for submission review only
# --------------------------------------------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion.script_util import create_model

from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

from ipdb import set_trace as debug


class Density128Net(torch.nn.Module):
    def __init__(self, log, noise_levels, cond=False):
        super(Density128Net, self).__init__()
        if cond == False:
            in_channels = 1
        else:
            in_channels = 2
        self.diffusion_model = create_model(image_size=128, num_channels=128, num_res_blocks=2, in_channels=in_channels, out_channels=1)
        log.info(f"[Net] Initialized network! Size={util.count_parameters(self.diffusion_model)}!")

        self.diffusion_model.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)
 


class MAG128Net(torch.nn.Module):
    def __init__(self, log, noise_levels, cond=False):
        super(MAG128Net, self).__init__()

        if cond == False:
            in_channels = 4
        else:
            in_channels = 8


        self.diffusion_model = create_model(image_size=128, num_channels=128, num_res_blocks=2, in_channels=in_channels, out_channels=4)
        log.info(f"[Net] Initialized network! Size={util.count_parameters(self.diffusion_model)}!")

        self.diffusion_model.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)        



