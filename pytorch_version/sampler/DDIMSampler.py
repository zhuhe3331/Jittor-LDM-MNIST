from typing import List, Optional

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from pytorch_version.sampler.DiffusionSampler import DiffusionSampler
from pytorch_version.LatentDiffusion import LatentDiffusion


class DDIMSampler(DiffusionSampler):
    def __init__(self, model: LatentDiffusion, n_steps: int, ddim_discretize: str = "uniform", ddim_eta: float = 0.):
        super().__init__(model)

        self.n_steps = model.n_steps

        if ddim_discretize == "uniform":
            c = self.n_steps // n_steps
            self.time_steps = np.asarray(list(range(0, self.n_steps, c))) 
        elif ddim_discretize == "quad":
            self.time_steps = (np.linspace(0, np.sqrt(self.n_steps * 0.8), n_steps) ** 2).astype(int) 
        else:
            raise NotImplementedError()

        with torch.no_grad():
            alpha_bar = self.model.alpha_bar
            self.alpha = alpha_bar.clone().to(torch.float32)
            self.alpha_sqrt = torch.sqrt(self.alpha)
            self.one_m_alpha_sqrt = torch.sqrt(1 - self.alpha)
            self.ddim_alpha = (alpha_bar[self.time_steps].clone()).to(torch.float32)
            self.ddim_alpha_prev = torch.concat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])

            self.ddim_sigma = (
                    ddim_eta *
                    ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha)) *
                    (1 - self.ddim_alpha / self.ddim_alpha_prev + 1e-7) ** 0.5
            )

            self.ddim_1m_alpha_sqrt = (1 - self.ddim_alpha) ** 0.5


    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)

        mean = self.alpha_sqrt[index].view(-1, 1, 1, 1)
        var = self.one_m_alpha_sqrt[index].view(-1, 1, 1, 1)

        return mean * x0 + var * noise
        # return self.ddim_alpha_sqrt[index] * x0 + self.ddim_1m_alpha_sqrt[index] * noise

    @torch.no_grad()
    def get_x_prev_and_pred_x0(self, e_t: torch.Tensor, index: int, x: torch.Tensor,
                               temperature: float = 1,
                               repeat_noise: bool = False,
                               ):
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_1m_alpha = self.ddim_1m_alpha_sqrt[index]

        # print(x* sqrt_1m_alpha* e_t, alpha ** 0.5)
        pred_x0 = (x - sqrt_1m_alpha * e_t) / (alpha ** 0.5 + 1e-8)
        # raise Exception()
        dir_xt = (1. - alpha - (sigma ** 2)).sqrt() * e_t
        # if True in np.isnan(dir_xt):
        #     print((1. - alpha - (sigma ** 2)), e_t)
        #     raise Exception()

        if sigma == 0:
            noise = 0
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature

        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        # if True in np.isnan(x_prev):
        #     print(alpha_prev, pred_x0, dir_xt, sigma)
        #     raise Exception()

        return x_prev, pred_x0

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int, index: int,
                 repeat_noise: bool = False,
                 temperature: float = 1,
                 uncond_scale: float = 1,
                 uncond_cond: Optional[torch.Tensor] = None
                 ):
        e_t = self.get_eps(x, t, c, uncond_scale, uncond_cond)
        # if True in np.isnan(e_t):
        #     print(x, t, c)
        #     raise Exception()
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x, temperature, repeat_noise)
        # if True in np.isnan(x_prev.numpy()):
        #     print(x, e_t, index, x_prev, pred_x0)
        #     raise Exception()

        return x_prev, pred_x0, e_t


    @torch.no_grad()
    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0
               ):
        bs = shape[0]
        x = x_last if x_last is not None else torch.randn(shape).to("cuda")

        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in tqdm(enumerate(time_steps)):
            index = len(time_steps) - i - 1
            ts = torch.full((bs, ), step, dtype=torch.int64, device="cuda")
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step, index,
                                            repeat_noise,
                                            temperature,
                                            uncond_scale,
                                            uncond_cond)
            # if True in np.isnan(x.numpy()):
            #     print(x, pred_x0, e_t, cond, ts, step)
            #     raise Exception()

        return x

    def loss(self, x0, c, noise=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,)).to(x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise)
        eps_theta = self.model(xt, t, c)

        loss = F.mse_loss(eps_theta, noise, reduction="mean")
        return loss