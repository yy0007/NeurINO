import torch
import torch.nn as nn
import torch.nn.functional as F


class MedNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=4, 
                kernel_size:int=7, 
                do_res:int=True,
                norm_type:str = 'group',
                n_groups:int or None = None,
                dim = '3d',
                grn = False
                ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

 
    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1  
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=False, norm_type = 'group', dim='3d', grn=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size, 
                        do_res = False, norm_type = norm_type, dim=dim,
                        grn=grn)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=False, norm_type = 'group', dim='3d', grn = False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type = norm_type, dim=dim,
                         grn=grn)

        self.resample_do_res = do_res
        
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:            
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )


    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1,0,1,0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim):
        super().__init__()
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        
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

         
if __name__ == "__main__":


    # network = nnUNeXtBlock(in_channels=12, out_channels=12, do_res=False).cuda()

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 8, 8, 8)).cuda()
    #     print(network(x).shape)

    # network = DownsampleBlock(in_channels=12, out_channels=24, do_res=False)

    # with torch.no_grad():
    #     print(network)
    #     x = torch.zeros((2, 12, 128, 128, 128))
    #     print(network(x).shape)

    network = MedNeXtBlock(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').cuda()
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 12, 64, 64, 64)).cuda()
        print(network(x).shape)
