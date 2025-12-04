import torch, torch.nn as nn

class BatchRenorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.01, rmax=3.0, dmax=5.0):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.rmax, self.dmax = rmax, dmax
        self.register_buffer('rmax_buf', torch.tensor(1.0))
        self.register_buffer('dmax_buf', torch.tensor(0.0))

    @torch.no_grad()
    def set_rd(self, r, d):
        self.rmax_buf.fill_(r); self.dmax_buf.fill_(d)

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=[0,2,3,4])
            var  = x.var (dim=[0,2,3,4], unbiased=False)
            running_std = torch.sqrt(self.running_var + self.eps)
            batch_std   = torch.sqrt(var + self.eps)
            r = (batch_std / running_std).clamp(1.0/self.rmax_buf.item(), self.rmax_buf.item())
            d = ((mean - self.running_mean) / running_std).clamp(-self.dmax_buf.item(), self.dmax_buf.item())
            x_hat = (x - mean[None,:,None,None,None]) / batch_std[None,:,None,None,None]
            y = x_hat * r[None,:,None,None,None] + d[None,:,None,None,None]

            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
            
            if self.affine:
                y = y * self.weight[None,:,None,None,None] + self.bias[None,:,None,None,None]
            return y
        else:
            mean = self.running_mean[None,:,None,None,None]
            std  = torch.sqrt(self.running_var + self.eps)[None,:,None,None,None]
            y = (x - mean) / std
            if self.affine:
                y = y * self.weight[None,:,None,None,None] + self.bias[None,:,None,None,None]
            return y
    

class ChannelRenorm3d_Flexible(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.01, rmax=3.0, dmax=5.0):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum
        self.rmax = rmax
        self.dmax = dmax

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1, 1))
        self.register_buffer("running_std",  torch.ones(1, num_channels, 1, 1, 1))

    def _norm_core(self, x_ncdhw):
        # x_ncdhw: (N, C, D, H, W)
        mean = x_ncdhw.mean(dim=(0, 2, 3, 4), keepdim=True)
        std  = x_ncdhw.std(dim=(0, 2, 3, 4), keepdim=True, unbiased=False)

        if self.training:
            r = (std / (self.running_std + self.eps)).clamp(1 / self.rmax, self.rmax)
            d = ((mean - self.running_mean) / (self.running_std + self.eps)).clamp(-self.dmax, self.dmax)

            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
            self.running_std.mul_(1 - self.momentum).add_(self.momentum * std.detach())

            y = (x_ncdhw - mean) / (std + self.eps)
            y = r * y + d
        else:
            y = (x_ncdhw - self.running_mean) / (self.running_std + self.eps)

        y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"ChannelRenorm3d_Flexible expects 5D input, got {tuple(x.shape)}")

        N, *rest = x.shape
        C = self.num_channels

        if x.shape[1] == C:  # channels_first
            return self._norm_core(x)
        elif x.shape[-1] == C:  # channels_last
            # NDHWC -> NCDHW
            x_cf = x.permute(0, 4, 1, 2, 3).contiguous()
            y_cf = self._norm_core(x_cf)
            # NCDHW -> NDHWC
            y = y_cf.permute(0, 2, 3, 4, 1).contiguous()
            return y
        else:
            raise RuntimeError(f"Cannot infer channel axis for shape {tuple(x.shape)} with C={C}")