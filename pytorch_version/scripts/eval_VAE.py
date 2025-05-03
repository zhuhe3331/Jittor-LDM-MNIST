import sys
sys.path.append("/root/diffusion")

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import torch
from torch import nn
from torchvision import datasets, transforms

from pytorch_version.model.AutoEncoder import AutoEncoder, Encoder, Decoder, loss
from pytorch_version.utils import train_VAE

# 加载配置文件
cfg = OmegaConf.load('./config/vae.yaml')

# 设置训练参数
device = "cuda" if torch.cuda.is_available() and cfg.training.use_cuda else "cpu"
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
vae_model_path = cfg.training.model_path

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

dataset = datasets.MNIST('../mnist_data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

encoder = Encoder(**cfg.model.encoder).to(device)
decoder = Decoder(**cfg.model.decoder).to(device)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
).to(device)

auto_encoder.load_state_dict(torch.load(vae_model_path))

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(4, 4)

for pics, _ in train_loader:
    pics = pics.to(device)
    z = auto_encoder.encode(pics[:8])
    recon_images = auto_encoder.decode(z.sample())
    images = torch.cat([pics[:8], recon_images])
    images = images.reshape(4, 4, image_size, image_size).cpu().detach().numpy()

    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((images[n_row, n_col]), cmap="gray")
            f_ax.axis("off")

    plt.show()
    plt.savefig(f"./results/mnist_VAE_{time.time()}.png")
    plt.close()
    break