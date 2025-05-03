import sys
sys.path.append("/root/diffusion")

import time
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

from jittor_version.sampler.DDPMSampler import DDPMSampler
from jittor_version.LatentDiffusion import LatentDiffusion
from jittor_version.model.UNet import UNetModel
from jittor_version.utils import train_DDPM

import jittor as jt
from jittor import nn
from jittor.dataset.mnist import MNIST
from jittor import transform as trans

cfg = OmegaConf.load('./config/ddpm_cond.yaml')

jt.flags.use_cuda = cfg.training.use_cuda
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
weight_decay = cfg.training.weight_decay
model_path = cfg.training.model_path
load_model = cfg.training.load_model
d_cond = cfg.model.unet.d_cond

transform = trans.Compose([
    trans.Resize(image_size),
    trans.Gray(),
])

train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True, transform=transform)

unet = UNetModel(**cfg.model.unet)

auto_encoder = nn.Identity()
context_embedder = nn.Identity()

ldm = LatentDiffusion(
    unet_model=unet,
    auto_encoder=auto_encoder,
    context_embedder=context_embedder,
    **cfg.model.latent_diffusion
)

if load_model:
    ldm.load_state_dict(jt.load(model_path))

ddpm = DDPMSampler(ldm)

optimizer = jt.optim.AdamW(ldm.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_history = []
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    avg_epoch_loss = train_DDPM(train_loader, optimizer, ddpm, d_cond)
    loss_history.append(avg_epoch_loss)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 50)
    jt.save(ldm.state_dict(), model_path)

print("Training completed!")
print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes.")

with open('./results/loss_ddpm_cond.txt', 'w') as f:
    for i, loss_value in enumerate(loss_history, 1):
        f.write(f"Step {i}: Loss = {loss_value:.6f}\n")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
plt.savefig(f"./losses/mnist_ddpm_cond_{time.time()}.png")
plt.close()

cond = jt.arange(start=0, end=10, step=1).repeat_interleave(d_cond, dim=0).reshape(10, 1, d_cond)
generated_images = ddpm.sample((10, 1, image_size, image_size), cond)

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 5)

imgs = generated_images.reshape(2, 5, image_size, image_size).numpy()
for n_row in range(2):
    for n_col in range(5):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]), cmap="gray")
        f_ax.axis("off")

plt.show()
plt.savefig(f"./results/mnist_ddpm_cond_{time.time()}.png")
plt.close()
