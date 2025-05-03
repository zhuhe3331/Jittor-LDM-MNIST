import jittor as jt

def train_DDPM(train_loader, optimizer, sampler, d_cond):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        if d_cond != 0:
            cond = jt.repeat_interleave(labels, d_cond, dim=0).reshape(-1, 1, d_cond)
        else:
            cond = None
        loss = sampler.loss(pics, cond)
        optimizer.step(loss)
        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_epoch_loss = jt.array(epoch_loss).mean().item()
    return avg_epoch_loss

def train_VAE(train_loader, optimizer, vae, loss):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        z = vae.encode(pics)
        pics_hat = vae.decode(z.sample())
        recon_loss, kl_loss = loss(pics, pics_hat, z.mean, z.log_var, 0.)
        ls = recon_loss + kl_loss
        optimizer.step(ls)
        epoch_loss.append(ls.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {ls.item():.4f}")

    avg_epoch_loss = jt.array(epoch_loss).mean().item()
    return avg_epoch_loss

def train_LDM(train_loader, optimizer, ldm, sampler, d_cond):
    epoch_loss = []
    for step, (pics, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        z = ldm.autoencoder_encode(pics)
        if d_cond != 0:
            cond = jt.repeat_interleave(labels, d_cond, dim=0).reshape(-1, 1, d_cond)
        else:
            cond = None
        loss = sampler.loss(z, cond)

        optimizer.step(loss)
        epoch_loss.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_epoch_loss = jt.array(epoch_loss).mean().item()
    return avg_epoch_loss
