import sys
sys.path.append("/root/diffusion")

import time
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

from jittor_version.sampler.DDPMSampler import DDPMSampler
from jittor_version.LatentDiffusion import LatentDiffusion
from jittor_version.model.UNet import UNetModel

import jittor as jt
from jittor import nn

cfg = OmegaConf.load('./config/ddpm.yaml')

jt.flags.use_cuda = cfg.training.use_cuda
image_size = cfg.training.image_size
model_path = cfg.training.model_path

unet = UNetModel(**cfg.model.unet)

auto_encoder = nn.Identity()
context_embedder = nn.Identity()

ldm = LatentDiffusion(
    unet_model=unet,
    auto_encoder=auto_encoder,
    context_embedder=context_embedder,
    **cfg.model.latent_diffusion
)

ldm.load_state_dict(jt.load(model_path))

ddpm = DDPMSampler(ldm)

generated_images = ddpm.sample((10, 1, image_size, image_size), None)

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 5)

imgs = generated_images.reshape(2, 5, image_size, image_size).numpy()
for n_row in range(2):
    for n_col in range(5):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]), cmap="gray")
        f_ax.axis("off")

plt.show()
plt.savefig(f"./results/mnist_ddpm_{time.time()}.png")
plt.close()
