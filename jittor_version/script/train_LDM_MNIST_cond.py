import sys
sys.path.append("/root/diffusion")

import jittor as jt
from jittor import nn
from jittor.dataset.mnist import MNIST
from jittor import transform as trans

import time
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

from jittor_version.sampler.DDPMSampler import DDPMSampler
from jittor_version.LatentDiffusion import LatentDiffusion
from jittor_version.model.UNet import UNetModel
from jittor_version.model.AutoEncoder import AutoEncoder, Encoder, Decoder
from jittor_version.utils import train_LDM

cfg = OmegaConf.load('./config/ldm_cond.yaml')

mean = 0.5
std = 0.5

jt.flags.use_cuda = cfg.training.use_cuda

image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
model_path = cfg.training.model_path
vae_model_path = cfg.training.vae_model_path
weight_decay = cfg.training.weight_decay
load_model = cfg.training.load_model
d_cond = cfg.model.unet.d_cond

transform = trans.Compose([
    trans.Resize(image_size),
    trans.Gray(),
])

train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True,
                                                                transform=transform)

unet = UNetModel(
    **cfg.model.unet
)

encoder = Encoder(
    **cfg.model.encoder
)

decoder = Decoder(
    **cfg.model.decoder
)

auto_encoder = AutoEncoder(
    encoder=encoder,
    decoder=decoder,
    **cfg.model.auto_encoder
)

auto_encoder.load_state_dict(jt.load(vae_model_path))

context_embedder = nn.Identity()

ldm = LatentDiffusion(
    unet_model=unet,
    auto_encoder=auto_encoder,
    context_embedder=context_embedder,
    **cfg.model.ldm
)

if load_model:
    unet.load_state_dict(jt.load(model_path))

ddpm = DDPMSampler(ldm)

optimizer = jt.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_history = []
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    avg_epoch_loss = train_LDM(train_loader, optimizer, ldm, ddpm, d_cond)
    loss_history.append(avg_epoch_loss)
    jt.save(unet.state_dict(), model_path)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 50)

print("Training completed!")
print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes.")

with open('./results/loss_ldm_cond.txt', 'w') as f:
    for i, loss_value in enumerate(loss_history, 1):
        f.write(f"Step {i}: Loss = {loss_value:.6f}\n")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
plt.savefig(f"./losses/mnist_ldm_cond_{time.time()}.png")
plt.close()

cond = jt.arange(start=0, end=10, step=1).repeat_interleave(d_cond, dim=0).reshape(10, 1, d_cond)
z = ddpm.sample((10, 1, 8, 8), cond)
generated_images = ldm.autoencoder_decode(z)

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 5)

imgs = generated_images.reshape(2, 5, image_size, image_size).numpy()

for n_row in range(2):
    for n_col in range(5):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]), cmap="gray")
        f_ax.axis("off")

plt.show()
plt.savefig(f"./results/mnist_ldm_cond_{time.time()}.png")
plt.close()
