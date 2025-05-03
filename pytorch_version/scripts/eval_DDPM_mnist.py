import sys
sys.path.append("/root/diffusion")

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from pytorch_version.model.UNet import UNetModel
from pytorch_version.LatentDiffusion import LatentDiffusion
from pytorch_version.sampler.DDPMSampler import DDPMSampler
from pytorch_version.utils import train_DDPM

# 加载配置文件
cfg = OmegaConf.load('./config/ddpm.yaml')

# 设置训练参数
device = "cuda" if torch.cuda.is_available() and cfg.training.use_cuda else "cpu"
image_size = cfg.training.image_size
model_path = cfg.training.model_path
d_cond = cfg.model.unet.d_cond

# 初始化模型
auto_encoder = nn.Identity()
context_embedder = nn.Identity()
unet = UNetModel(**cfg.model.unet)
ldm = LatentDiffusion(
    unet_model=unet,
    auto_encoder=auto_encoder,
    context_embedder=context_embedder,
    **cfg.model.latent_diffusion
).to(device)

ddpm = DDPMSampler(model=ldm)

ldm.load_state_dict(torch.load(model_path))

generated_images = ddpm.sample((10, 1, image_size, image_size), None)

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 5)

imgs = generated_images.reshape(2, 5, image_size, image_size).cpu().numpy()
for n_row in range(2):
    for n_col in range(5):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]), cmap="gray")
        f_ax.axis("off")

plt.show()
plt.savefig(f"./results/mnist_ddpm_{time.time()}.png")
plt.close()
