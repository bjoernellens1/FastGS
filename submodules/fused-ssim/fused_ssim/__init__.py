from typing import NamedTuple
import torch.nn as nn
import torch

# Try to load the compiled CUDA/ROCm extension.  If it is not available (e.g.
# on a CPU-only machine) we fall back to a pure-PyTorch implementation so the
# rest of the codebase can run without modification.
try:
    from fused_ssim_cuda import fusedssim, fusedssim_backward
    _HAS_FUSED_CUDA = True
except ImportError:
    _HAS_FUSED_CUDA = False

allowed_padding = ["same", "valid"]


# ---------------------------------------------------------------------------
# Pure-PyTorch SSIM helpers (used when the CUDA extension is unavailable)
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    import math
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d = _gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim_map_pytorch(C1: float, C2: float, img1: torch.Tensor,
                      img2: torch.Tensor) -> torch.Tensor:
    """Differentiable per-pixel SSIM map using standard convolutions."""
    import torch.nn.functional as F
    B, CH, H, W = img1.shape
    window_size = 11
    window = _create_window(window_size, CH).to(img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=CH)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=CH)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=CH) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=CH) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=CH) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


# ---------------------------------------------------------------------------
# CUDA-backed autograd Function (used when the extension is compiled)
# ---------------------------------------------------------------------------

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None


def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    if _HAS_FUSED_CUDA:
        map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    else:
        # Pure-PyTorch path (CPU, or GPU without the compiled extension)
        map = _ssim_map_pytorch(C1, C2, img1, img2)
        if padding == "valid":
            map = map[:, :, 5:-5, 5:-5]
    return map.mean()


def fused_ssim_(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    if _HAS_FUSED_CUDA:
        map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    else:
        map = _ssim_map_pytorch(C1, C2, img1, img2)
        if padding == "valid":
            map = map[:, :, 5:-5, 5:-5]
    return map.squeeze(0).mean(0)
