import sys
sys.path.append("/root/diffusion")

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import jittor as jt
from jittor.dataset.mnist import MNIST
from jittor import transform as trans

from jittor_version.model.AutoEncoder import AutoEncoder, Encoder, Decoder

# 加载配置文件
cfg = OmegaConf.load('./config/vae.yaml')

# 设置训练参数
jt.flags.use_cuda = cfg.training.use_cuda
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
vae_model_path = cfg.training.model_path

# 数据预处理
transform = trans.Compose([
    trans.Resize(image_size),
    trans.Gray(),
])

train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True, transform=transform)

# 初始化模型
encoder = Encoder(**cfg.model.encoder)
decoder = Decoder(**cfg.model.decoder)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
)

auto_encoder.load_state_dict(jt.load(vae_model_path))

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(4, 4)

for pics, _ in train_loader:
    gas = auto_encoder.encode(pics[:8])
    recon_images = auto_encoder.decode(gas.sample())
    images = jt.concat([pics[:8], recon_images])
    images = images.reshape(4, 4, image_size, image_size).numpy()
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((images[n_row, n_col]), cmap="gray")
            f_ax.axis("off")

    plt.show()
    plt.savefig(f"./results/mnist_VAE_{time.time()}.png")
    plt.close()
    break