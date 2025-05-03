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
from pytorch_version.model.AutoEncoder import AutoEncoder, Encoder, Decoder
from pytorch_version.utils import train_LDM

# 加载配置文件
cfg = OmegaConf.load('./config/ldm.yaml')

# 设置训练参数
device = "cuda" if torch.cuda.is_available() and cfg.training.use_cuda else "cpu"
image_size = cfg.training.image_size
batch_size = cfg.training.batch_size
epochs = cfg.training.epochs
learning_rate = cfg.training.learning_rate
model_path = cfg.training.model_path
vae_model_path = cfg.training.vae_model_path
weight_decay = cfg.training.weight_decay
load_model = cfg.training.load_model
d_cond = cfg.model.unet.d_cond

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

auto_encoder.load_state_dict(torch.load(vae_model_path))

context_embedder = nn.Identity()

ldm = LatentDiffusion(
    unet_model=unet,
    auto_encoder=auto_encoder,
    context_embedder=context_embedder,
    **cfg.model.ldm
).to(device)

if load_model:
    unet.load_state_dict(torch.load(model_path))

ddpm = DDPMSampler(ldm)

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

loss_history = []
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    avg_epoch_loss = train_LDM(train_loader, optimizer, ldm, ddpm, d_cond, device)
    loss_history.append(avg_epoch_loss)
    torch.save(unet.state_dict(), model_path)
    print(f"Epoch {epoch + 1} Completed. Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 50)

print("Training completed!")
print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes.")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()
plt.savefig(f"./losses/mnist_ldm_{time.time()}.png")
plt.close()

with open('./results/loss_ldm.txt', 'w') as f:
    for i, loss_value in enumerate(loss_history, 1):
        f.write(f"Step {i}: Loss = {loss_value:.6f}\n")

generated_images = ddpm.sample((10, 1, 8, 8), None)
images = auto_encoder.decode(generated_images)

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 5)

imgs = images.reshape(2, 5, image_size, image_size).cpu().detach().numpy()
for n_row in range(2):
    for n_col in range(5):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]), cmap="gray")
        f_ax.axis("off")

plt.show()
plt.savefig(f"./results/mnist_ldm_{time.time()}.png")
plt.close()
