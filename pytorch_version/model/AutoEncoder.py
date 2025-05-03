from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', emb_channels: int, z_channels: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        z = self.encoder(img)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        z = self.post_quant_conv(z)
        return self.decoder(z)



class Encoder(nn.Module):
    """
    ## Encoder module
    """

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 in_channels: int, z_channels: int):
        """
        :param channels: is the number of channels in the first convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            subsequent blocks
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param in_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

        # Initial $3 \times 3$ convolution layer that maps the image to `channels`
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)

        # Number of channels in each top level block
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()
        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks
            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            #
            self.down.append(down)

        # Final ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # Map to embedding space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)

    def forward(self, img: torch.Tensor):
        """
        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        x = self.conv_in(img)

        for down in self.down:
            for block in down.block:
                # print(x.shape)
                # print(block)
                x = block(x)
            x = down.downsample(x)

        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        #
        return x


class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 out_channels: int, z_channels: int):
        """
        :param channels: is the number of channels in the final convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            previous blocks, in reverse order
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param out_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)

        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Initial $3 \times 3$ convolution layer that maps the embedding space to `channels`
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()
        up_block = []
        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks
            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            # Prepend to be consistent with the checkpoint
            up_block.append(up)

        self.up = nn.ModuleList(up_block[::-1])
        # Map to image space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map to `channels` with the initial convolution
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for up in reversed(self.up):
            # ResNet Blocks
            for block in up.block:
                h = block(h)
            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        #
        return img


class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # Group normalization
        self.norm = normalization(channels)
        # Query, key and value mappings
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        # Final $1 \times 1$ convolution layer
        self.proj_out = nn.Conv2d(channels, channels, 1)
        # Attention scaling factor
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        """
        :param x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.einsum('bij,bcj->bci', attn, v)

        # Reshape back to `[batch_size, channels, height, width]`
        out = out.view(b, c, h, w)
        # Final $1 \times 1$ convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out


class GaussianDistribution:
    """
    ## Gaussian Distribution
    """

    def __init__(self, parameters: torch.Tensor):
        """
        :param parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = x

        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


def normalization(channels: int):
    return nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-6)


def loss(y, y_hat, mean, log_var, kl_scale):
    recon_loss = F.mse_loss(y, y_hat)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var)))
    # print(kl_loss.shape)
    # loss = recon_loss + kl_loss * kl_scale
    # loss = recon_loss

    return recon_loss, kl_loss * kl_scale

def _test_():
    data = torch.randn((16, 1, 16, 16))
    encoder = Encoder(channels=32,
                      channel_multipliers=[1],
                      n_resnet_blocks=1,
                      in_channels=1,
                      z_channels=3)

    decoder = Decoder(channels=32,
                      channel_multipliers=[1],
                      n_resnet_blocks=1,
                      out_channels=1,
                      z_channels=3)

    ae = AutoEncoder(encoder, decoder, 3, 3)

    gs = ae.encode(data)
    z = gs.sample()
    recon_img = ae.decode(z)
    print(recon_img.shape)
    recon, kl = loss(data, recon_img, gs.mean, gs.log_var, 1.)
    ls = recon + kl
    print(ls)
    # print(recon_img)
    # print(recon_img.shape)

if __name__ == "__main__":
    _test_()
