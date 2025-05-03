from typing import List, Optional

from tqdm import tqdm

import jittor as jt
from jittor import nn
import numpy as np

from jittor_version.sampler.DiffusionSampler import DiffusionSampler
from jittor_version.LatentDiffusion import LatentDiffusion


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

        with jt.no_grad():
            alpha_bar = self.model.alpha_bar
            self.alpha = jt.float32(alpha_bar).clone()
            self.alpha_sqrt = jt.sqrt(self.alpha)
            self.one_m_alpha_sqrt = jt.sqrt(1 - self.alpha)
            self.ddim_alpha = jt.float32(alpha_bar[self.time_steps].clone())
            self.ddim_alpha_prev = jt.concat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])
            # print(self.ddim_alpha[0])
            # print(self.ddim_alpha_prev[0])


            self.ddim_sigma = (
                    ddim_eta *
                    ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha)) *
                    (1 - self.ddim_alpha / self.ddim_alpha_prev + 1e-7) ** 0.5
            )

            self.ddim_1m_alpha_sqrt = (1 - self.ddim_alpha) ** 0.5


    @jt.no_grad()
    def q_sample(self, x0: jt.Var, index: int, noise: Optional[jt.Var] = None):
        if noise is None:
            noise = jt.randn_like(x0)

        mean = self.alpha_sqrt[index].view(-1, 1, 1, 1)
        var = self.one_m_alpha_sqrt[index].view(-1, 1, 1, 1)

        return mean * x0 + var * noise
        # return self.ddim_alpha_sqrt[index] * x0 + self.ddim_1m_alpha_sqrt[index] * noise

    @jt.no_grad()
    def get_x_prev_and_pred_x0(self, e_t: jt.Var, index: int, x: jt.Var,
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
            noise = jt.randn((1, *x.shape[1:]))
        else:
            noise = jt.randn(x.shape)

        noise = noise * temperature

        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        # if True in np.isnan(x_prev):
        #     print(alpha_prev, pred_x0, dir_xt, sigma)
        #     raise Exception()

        return x_prev, pred_x0





    @jt.no_grad()
    def p_sample(self, x: jt.Var, c: jt.Var, t: jt.Var, step: int, index: int,
                 repeat_noise: bool = False,
                 temperature: float = 1,
                 uncond_scale: float = 1,
                 uncond_cond: Optional[jt.Var] = None
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


    @jt.no_grad()
    def sample(self,
               shape: List[int],
               cond: jt.Var,
               repeat_noise: bool = False,
               temperature: float = 1,
               x_last: Optional[jt.Var] = None,
               uncond_scale: float = 1,
               uncond_cond: Optional[jt.Var] = None,
               skip_steps: int = 0
               ):
        bs = shape[0]
        x = x_last if x_last is not None else jt.randn(shape)

        time_steps = np.flip(self.time_steps)[skip_steps:]

        for i, step in tqdm(enumerate(time_steps)):
            index = len(time_steps) - i - 1
            ts = jt.full((bs, ), step, dtype=jt.int64)
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
        t = jt.randint(0, self.n_steps, (batch_size,))
        if noise is None:
            noise = jt.randn_like(x0)

        xt = self.q_sample(x0, t, noise)
        eps_theta = self.model(xt, t, c)

        loss = nn.mse_loss(eps_theta, noise, reduction="mean")
        return loss