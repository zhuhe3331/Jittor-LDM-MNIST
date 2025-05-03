import sys
sys.path.append("/root/diffusion")

import time
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

import jittor as jt
from jittor.dataset.mnist import MNIST
from jittor import transform as trans

from jittor_version.model.AutoEncoder import AutoEncoder, Encoder, Decoder, loss
from jittor_version.utils import train_VAE

# 加载配置文件
cfg = OmegaConf.load('./config/vae.yaml')

# 设置训练参数
jt.flags.use_cuda = cfg.training.use_cuda
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
vae_model_path = cfg.training.model_path
weight_decay = cfg.training.weight_decay
load_model = cfg.training.load_model

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

if load_model:
    auto_encoder.load_state_dict(jt.load(vae_model_path))

optimizer = jt.optim.AdamW(auto_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_history = []
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    avg_epoch_loss = train_VAE(train_loader, optimizer, auto_encoder, loss)
    loss_history.append(avg_epoch_loss)
    jt.save(auto_encoder.state_dict(), vae_model_path)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 50)

# Save the final model
print("Training completed!")
print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes.")

with open('./results/loss_vae.txt', 'w') as f:
    for i, loss_value in enumerate(loss_history, 1):
        f.write(f"Step {i}: Loss = {loss_value:.6f}\n")

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