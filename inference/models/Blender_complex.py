import torch.nn as nn
import torch
import torch.nn.functional as F


class BlendedModel(nn.Module):
    def __init__(self, ks=9):
        super(BlendedModel, self).__init__()
        self.sigma_k = nn.Parameter(torch.rand(1))
        self.sigma_size = ks

        self.alpha_r = nn.Parameter(torch.rand(1))
        self.b_r = nn.Parameter(torch.rand(1))
        self.alpha_g = nn.Parameter(torch.rand(1))
        self.b_g = nn.Parameter(torch.rand(1))
        self.alpha_b = nn.Parameter(torch.rand(1))
        self.b_b = nn.Parameter(torch.rand(1))

    def gaussian_kernel(self, dep):
        B, _, H, W = dep.shape
        kernel_size = self.sigma_size
        r = kernel_size // 2
        sigma_k = torch.relu(self.sigma_k + 1e-3)
        sigma_map = sigma_k * dep

        kernels = torch.zeros((B, 1, kernel_size, kernel_size, H, W), device=sigma_map.device)

        y, x = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)], indexing='ij')
        y, x = y.to(sigma_map.device), x.to(sigma_map.device)
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernels[:, :, i, j, :, :] = torch.exp(-((x[i, j] ** 2 + y[i, j] ** 2) / (2 * sigma_map ** 2)))

        kernels = kernels / kernels.sum(dim=(2, 3), keepdim=True)
        return kernels

    def forward(self, x, dep, noise):
        B, C, H, W = x.shape
        alpha_r = torch.sigmoid(self.alpha_r)
        b_r = torch.sigmoid(self.b_r)
        alpha_g = torch.sigmoid(self.alpha_g)
        b_g = torch.sigmoid(self.b_g)
        alpha_b = torch.sigmoid(self.alpha_b)
        b_b = torch.sigmoid(self.b_b)

        t_r = torch.exp(- alpha_r * dep)
        t_g = torch.exp(- alpha_g * dep)
        t_b = torch.exp(- alpha_b * dep)

        r_channel = x[:, 2, :, :].unsqueeze(dim=1)
        g_channel = x[:, 1, :, :].unsqueeze(dim=1)
        b_channel = x[:, 0, :, :].unsqueeze(dim=1)

        br_expanded = b_r.expand_as(r_channel)
        bg_expanded = b_g.expand_as(g_channel)
        bb_expanded = b_b.expand_as(b_channel)

        out1_b_r = br_expanded * (1 - t_r)
        out1_b_g = bg_expanded * (1 - t_g)
        out1_b_b = bb_expanded * (1 - t_b)
        clear_back = torch.stack([out1_b_b, out1_b_g, out1_b_r], dim=1).squeeze(dim=2)

        b_r = (br_expanded * (1-noise) + noise) * (1 - t_r)
        b_g = (bg_expanded * (1-noise) + noise) * (1 - t_g)
        b_b = (bb_expanded * (1-noise) + noise) * (1 - t_b)

        kernel = self.gaussian_kernel(dep)

        out1_r_channel = r_channel * t_r
        out1_g_channel = g_channel * t_g
        out1_b_channel = b_channel * t_b
        clear = torch.stack([out1_b_channel, out1_g_channel, out1_r_channel], dim=1).squeeze(dim=2)

        ks = self.sigma_size
        r_channel = F.unfold(r_channel, kernel_size=(ks, ks), padding=(ks//2, ks//2)).view(B, 1, ks, ks, H, W)
        g_channel = F.unfold(g_channel, kernel_size=(ks, ks), padding=(ks//2, ks//2)).view(B, 1, ks, ks, H, W)
        b_channel = F.unfold(b_channel, kernel_size=(ks, ks), padding=(ks//2, ks//2)).view(B, 1, ks, ks, H, W)

        r_channel = (r_channel * kernel).sum(dim=[2, 3])
        g_channel = (g_channel * kernel).sum(dim=[2, 3])
        b_channel = (b_channel * kernel).sum(dim=[2, 3])
        r_channel = r_channel * t_r
        g_channel = g_channel * t_g
        b_channel = b_channel * t_b

        back = torch.stack([b_b, b_g, b_r], dim=1).squeeze(dim=2)
        x = torch.stack([b_channel, g_channel, r_channel], dim=1).squeeze(dim=2)

        clear = clear + clear_back
        clear_forward = x + clear_back
        x = x + back
        return x, clear, clear_forward
