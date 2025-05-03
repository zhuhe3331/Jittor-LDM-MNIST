import torch

def train_DDPM(train_loader, optimizer, sampler, d_cond, device):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()
        if d_cond != 0:
            cond = torch.repeat_interleave(labels, d_cond, dim=0).reshape(-1, 1, d_cond).to(device).to(torch.float32)
        else:
            cond = None

        loss = sampler.loss(pics, cond)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    return avg_epoch_loss

def train_VAE(train_loader, optimizer, auto_encoder, loss, device):
    epoch_loss = []
    for step, (pics, _) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()

        z = auto_encoder.encode(pics)
        pics_hat = auto_encoder.decode(z.sample())

        recon_loss, kl_loss = loss(pics, pics_hat, z.mean, z.log_var, 0.)
        ls = recon_loss + kl_loss

        ls.backward()
        optimizer.step()

        epoch_loss.append(ls.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Recon Loss: {recon_loss.item():.4f} - KL Loss: {kl_loss.item():.4f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    return avg_epoch_loss


def train_LDM(train_loader, optimizer, ldm, sampler, d_cond, device):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        pics = pics.to(device)
        optimizer.zero_grad()

        z = ldm.autoencoder_encode(pics)
        if d_cond != 0:
            cond = torch.repeat_interleave(labels, d_cond, dim=0).reshape(-1, 1, d_cond).to(device).to(torch.float32)
        else:
            cond = None

        loss = sampler.loss(z, cond)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_epoch_loss = torch.tensor(epoch_loss).mean().item()
    return avg_epoch_loss
