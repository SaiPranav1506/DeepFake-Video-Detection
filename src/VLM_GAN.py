"""VLM-GAN: A lightweight GAN with optional Vision-Language conditioning.

This module provides:
- Generator: maps latent noise (and optional text embedding) -> RGB image (3xH x W)
- Discriminator: PatchGAN-style discriminator returning real/fake logits
- Conditioning helpers: project text embeddings (VLM) into conditioning vectors
- Loss helpers: adversarial loss (hinge & BCE), feature matching, L1
- Training step helpers: single-step update functions for generator/discriminator

Notes:
- This is a compact, well-documented reference implementation for experimentation.
- Image size defaults to 224x224 to match ViT/other backbones used in this repo.
- For serious training you should add spectral norm, progressive growing, or BigGAN-style blocks.

Usage example:
    from VLM_GAN import Generator, Discriminator, project_text, gan_g_step, gan_d_step
    G = Generator(latent_dim=256, cond_dim=128)
    D = Discriminator(cond_dim=128)
    text_proj = project_text(text_dim=768, cond_dim=128)

    # In training loop:
    z = torch.randn(B, 256)
    text_emb = ...  # optional (B,768)
    c = text_proj(text_emb) if text_emb is not None else None
    g_loss = gan_g_step(G, D, z, c, criterion='hinge')

"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    # optional integration with local ViT feature extractor
    from models import ViTFeatureExtractor
except Exception:
    ViTFeatureExtractor = None


# ----------------------------- Utilities ----------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode='nearest'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=up_mode)
        self.conv = ConvBlock(in_ch, out_ch, k=3, s=1, p=1, norm=True, activation=True)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# ----------------------------- Generator ----------------------------------
class Generator(nn.Module):
    """Simple upsampling generator. Optionally conditioned on a vector `cond`.

    Args:
        latent_dim: dimension of the noise vector z
        cond_dim: dimension of conditioning vector (set to 0 or None if unconditional)
        base_channels: channel multiplier for convolutional stacks
        out_channels: output image channels (3)
        img_size: output image size (assumed square)
    """

    def __init__(self, latent_dim=256, cond_dim: Optional[int] = 0, base_channels=64, out_channels=3, img_size=224):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim or 0
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.img_size = img_size

        # compute number of upsampling steps -> start from a small spatial size e.g. 7x7
        self.start_spatial = 7
        # feature map size after initial linear: base_channels * 8
        self.start_channels = base_channels * 8

        in_dim = latent_dim + (self.cond_dim if self.cond_dim > 0 else 0)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.start_channels * self.start_spatial * self.start_spatial),
            nn.ReLU(inplace=True)
        )

        # progressive upsampling blocks to reach desired img_size
        blocks = []
        channels = self.start_channels
        spatial = self.start_spatial
        while spatial < img_size:
            out_ch = max(base_channels, channels // 2)
            blocks.append(UpConvBlock(channels, out_ch))
            channels = out_ch
            spatial *= 2
            # safety break
            if spatial > img_size:
                break

        self.ups = nn.Sequential(*blocks)

        # final conv to produce RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate images.

        Args:
            z: (B, latent_dim)
            cond: optional conditioning vector (B, cond_dim)
        Returns:
            images: (B, out_channels, H, W) in range [-1,1]
        """
        B = z.shape[0]
        if self.cond_dim > 0 and cond is not None:
            x = torch.cat([z, cond], dim=1)
        else:
            x = z
        x = self.fc(x)
        x = x.view(B, self.start_channels, self.start_spatial, self.start_spatial)
        x = self.ups(x)
        img = self.to_rgb(x)
        return img


# --------------------------- Discriminator --------------------------------
class Discriminator(nn.Module):
    """PatchGAN-like discriminator with optional conditioning via projection.

    If cond_dim > 0, cond vector is spatially replicated and concatenated to input.
    """

    def __init__(self, in_channels=3, cond_dim: Optional[int] = 0, base_channels=64):
        super().__init__()
        self.cond_dim = cond_dim or 0
        input_ch = in_channels + (1 if self.cond_dim > 0 else 0)  # we'll use projected scalar map

        layers = []
        ch = input_ch
        out_ch = base_channels
        # Down blocks
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        ch = out_ch
        for _ in range(3):
            out_ch = min(ch * 2, 512)
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out_ch

        # final conv to 1-channel patch output
        layers.append(nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

        # if cond present, small linear to project cond->1 value per spatial location
        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(self.cond_dim, 1)
        else:
            self.cond_proj = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return patch logits (B, 1, H', W'). If cond provided, add projected cond map.
        """
        if self.cond_dim > 0 and cond is not None:
            # project cond to a single scalar per-sample and tile as extra channel
            p = self.cond_proj(cond)  # (B,1)
            # normalize to near-zero mean
            p = p.unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)
            pmap = p.expand(-1, -1, x.shape[2], x.shape[3])
            x_in = torch.cat([x, pmap], dim=1)
        else:
            x_in = x
        logits = self.net(x_in)
        return logits


# --------------------------- Conditioning helper --------------------------
class TextProjector(nn.Module):
    """Project text embeddings into conditioning vector used by G and D.

    Simple MLP: text_dim -> cond_dim
    """

    def __init__(self, text_dim: int = 768, cond_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, txt: torch.Tensor) -> torch.Tensor:
        return self.net(txt)


def project_text(text_dim: int = 768, cond_dim: int = 128) -> TextProjector:
    return TextProjector(text_dim=text_dim, cond_dim=cond_dim)


# --------------------------- Loss helpers ---------------------------------

def adversarial_loss_d(logits_real: torch.Tensor, logits_fake: torch.Tensor, loss_type: str = 'hinge') -> torch.Tensor:
    """Discriminator loss. Supports 'hinge' and 'bce'.
    logits_real/logits_fake: raw outputs from discriminator.
    """
    if loss_type == 'hinge':
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        return 0.5 * (loss_real + loss_fake)
    elif loss_type == 'bce':
        real_target = torch.ones_like(logits_real)
        fake_target = torch.zeros_like(logits_fake)
        loss = F.binary_cross_entropy_with_logits(logits_real, real_target) + F.binary_cross_entropy_with_logits(logits_fake, fake_target)
        return 0.5 * loss
    else:
        raise ValueError('Unknown loss_type')


def adversarial_loss_g(logits_fake: torch.Tensor, loss_type: str = 'hinge') -> torch.Tensor:
    if loss_type == 'hinge':
        return -torch.mean(logits_fake)
    elif loss_type == 'bce':
        target = torch.ones_like(logits_fake)
        return F.binary_cross_entropy_with_logits(logits_fake, target)
    else:
        raise ValueError('Unknown loss_type')


# --------------------------- Training steps -------------------------------

def gan_d_step(D: nn.Module, G: nn.Module, real_imgs: torch.Tensor, z: torch.Tensor, cond: Optional[torch.Tensor], optimizer_d: torch.optim.Optimizer, loss_type: str = 'hinge', device: torch.device = torch.device('cpu')) -> dict:
    """Single discriminator step: update D to distinguish real vs fake.

    Returns dict of losses and logits for debugging.
    """
    D.train()
    G.eval()
    optimizer_d.zero_grad()

    real_imgs = real_imgs.to(device)
    z = z.to(device)
    if cond is not None:
        cond = cond.to(device)

    with torch.no_grad():
        fake = G(z, cond)

    logits_real = D(real_imgs, cond)
    logits_fake = D(fake, cond)

    # flatten logits to scalar per sample by averaging spatial map
    lr = torch.mean(logits_real.view(logits_real.size(0), -1), dim=1, keepdim=True)
    lf = torch.mean(logits_fake.view(logits_fake.size(0), -1), dim=1, keepdim=True)

    loss_d = adversarial_loss_d(lr, lf, loss_type=loss_type)
    loss_d.backward()
    optimizer_d.step()

    return {'loss_d': loss_d.item(), 'logit_real_mean': lr.mean().item(), 'logit_fake_mean': lf.mean().item()}


def gan_g_step(D: nn.Module, G: nn.Module, z: torch.Tensor, cond: Optional[torch.Tensor], optimizer_g: torch.optim.Optimizer, loss_type: str = 'hinge', device: torch.device = torch.device('cpu'), perceptual_net: Optional[nn.Module] = None, lambda_l1: float = 0.0) -> dict:
    """Single generator step: update G to fool D. Optionally include perceptual/L1 losses.
    """
    D.eval()
    G.train()
    optimizer_g.zero_grad()

    z = z.to(device)
    if cond is not None:
        cond = cond.to(device)

    fake = G(z, cond)
    logits_fake = D(fake, cond)
    lf = torch.mean(logits_fake.view(logits_fake.size(0), -1), dim=1, keepdim=True)

    loss_g = adversarial_loss_g(lf, loss_type=loss_type)

    # L1 pixel loss to encourage similarity to target if available via cond (not typical)
    if lambda_l1 > 0 and getattr(cond, 'shape', None) is not None:
        # placeholder: user can compute L1 against a provided target
        pass

    # optional perceptual loss
    if perceptual_net is not None:
        # user-supplied network should accept images scaled to [0,1] or [-1,1] depending on implementation
        pass

    loss_g.backward()
    optimizer_g.step()

    return {'loss_g': loss_g.item(), 'logit_fake_mean': lf.mean().item()}


# ---------------------- Image-conditional helpers ------------------------
def _extract_features(feat_extractor: nn.Module, imgs: torch.Tensor, device: torch.device):
    """Run a ViTFeatureExtractor on images and return per-sample feature vectors.

    Args:
        feat_extractor: instance of ViTFeatureExtractor (must output (B, F))
        imgs: (B,3,H,W) tensor in [0,1] or [0,255] depending on extractor expectations
        device: torch.device
    Returns:
        features tensor (B, F)
    """
    feat_extractor.eval()
    imgs = imgs.to(device)
    with torch.no_grad():
        feats = feat_extractor(imgs)
    return feats


def gan_d_step_image(D: nn.Module, G: nn.Module, real_imgs: torch.Tensor, z: torch.Tensor, cond_imgs: torch.Tensor, feat_extractor: nn.Module, optimizer_d: torch.optim.Optimizer, loss_type: str = 'hinge', device: torch.device = torch.device('cpu')) -> dict:
    """Discriminator step where conditioning is provided as images processed by feat_extractor."""
    cond = _extract_features(feat_extractor, cond_imgs, device)
    return gan_d_step(D, G, real_imgs, z, cond, optimizer_d, loss_type=loss_type, device=device)


def gan_g_step_image(D: nn.Module, G: nn.Module, z: torch.Tensor, cond_imgs: torch.Tensor, feat_extractor: nn.Module, optimizer_g: torch.optim.Optimizer, loss_type: str = 'hinge', device: torch.device = torch.device('cpu'), perceptual_net: Optional[nn.Module] = None, lambda_l1: float = 0.0) -> dict:
    """Generator step where conditioning is provided as images processed by feat_extractor."""
    cond = _extract_features(feat_extractor, cond_imgs, device)
    return gan_g_step(D, G, z, cond, optimizer_g, loss_type=loss_type, device=device, perceptual_net=perceptual_net, lambda_l1=lambda_l1)


def create_image_conditioned_gan(latent_dim: int = 256, text_cond_dim: Optional[int] = None, base_channels: int = 64, img_size: int = 224, device: torch.device = torch.device('cpu')) -> Tuple[Generator, Discriminator, Optional[nn.Module]]:
    """Factory that creates a Generator, Discriminator and a ViTFeatureExtractor for image conditioning.

    If local `models.ViTFeatureExtractor` is available it will be used. The returned
    Generator/Discriminator are configured with `cond_dim` equal to the ViT output dim
    (or to `text_cond_dim` if provided).
    """
    feat_extractor = None
    cond_dim = text_cond_dim
    if ViTFeatureExtractor is not None:
        feat_extractor = ViTFeatureExtractor()
        try:
            cond_dim = feat_extractor.out_dim
        except Exception:
            # infer with dummy
            with torch.no_grad():
                dummy = torch.zeros(1, 3, img_size, img_size)
                od = feat_extractor(dummy)
                cond_dim = od.shape[-1]
        feat_extractor.to(device)
    if cond_dim is None:
        cond_dim = 128

    G = Generator(latent_dim=latent_dim, cond_dim=cond_dim, base_channels=base_channels, img_size=img_size)
    D = Discriminator(in_channels=3, cond_dim=cond_dim, base_channels=base_channels)
    G.to(device)
    D.to(device)
    return G, D, feat_extractor


# --------------------------- Convenience factories ------------------------
def create_generator(latent_dim=256, cond_dim=128, base_channels=64, img_size=224) -> Generator:
    return Generator(latent_dim=latent_dim, cond_dim=cond_dim, base_channels=base_channels, img_size=img_size)


def create_discriminator(cond_dim=128, base_channels=64) -> Discriminator:
    return Discriminator(in_channels=3, cond_dim=cond_dim, base_channels=base_channels)


# --------------------------- Save/Load helpers ----------------------------
def save_checkpoint(path: str, G: nn.Module, D: nn.Module, opt_g=None, opt_d=None, extra: dict = None):
    state = {
        'G_state': G.state_dict(),
        'D_state': D.state_dict()
    }
    if opt_g is not None:
        state['opt_g'] = opt_g.state_dict()
    if opt_d is not None:
        state['opt_d'] = opt_d.state_dict()
    if extra:
        state['extra'] = extra
    torch.save(state, path)


def load_checkpoint(path: str, G: Optional[nn.Module] = None, D: Optional[nn.Module] = None, map_location='cpu') -> dict:
    ckpt = torch.load(path, map_location=map_location)
    if G is not None and 'G_state' in ckpt:
        G.load_state_dict(ckpt['G_state'])
    if D is not None and 'D_state' in ckpt:
        D.load_state_dict(ckpt['D_state'])
    return ckpt


# End of file
