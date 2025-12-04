import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

class LayerNorm3d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, (tuple, list)):
            normalized_shape = normalized_shape[0]
        self.num_channels = int(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))
        self.eps = eps

    def forward(self, x):
        C = self.num_channels
        nd = x.ndim
        if nd not in (4, 5):
            raise ValueError(f"LayerNorm3d expects 4D/5D input, got {nd}D.")

        if x.shape[1] == C:  # channels_first
            u = x.mean(dim=1, keepdim=True)
            s = (x - u).pow(2).mean(dim=1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            shape = [1] * nd
            shape[1] = C
            return self.weight.view(*shape) * x + self.bias.view(*shape)

        elif x.shape[-1] == C:  # channels_last
            return F.layer_norm(x, (C,), self.weight, self.bias, self.eps)
        else:
            raise RuntimeError(f"Cannot infer channel axis: x.shape={x.shape}, C={C}")


# # ============ inflate ============

def inflate_conv2d_to_3d(conv2d, kernel_depth=3, method="center", is_downsample="normal"):
    # print('kernel_depth:', kernel_depth)
    pH, pW = conv2d.padding

     # ✅ 区分 padding 逻辑
    if is_downsample == "normal":
        stride = (conv2d.stride[0], *conv2d.stride)
        pad = (pH, pH, pW)  
    elif is_downsample == "change_to_double":
        stride = (2, 2, 2)
        pad = (1, 1, 1)  
    elif is_downsample == "No":
        stride = (conv2d.stride[0], *conv2d.stride)
        pad = (kernel_depth // 2, pH, pW)  
        # print('~~~ Stage kernel_depth // 2:', kernel_depth // 2)
        
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_size=(kernel_depth, conv2d.kernel_size[0], conv2d.kernel_size[1]),
        stride=stride, 
        # padding=(kernel_depth // 2, *conv2d.padding),
        padding=pad,
        bias=(conv2d.bias is not None),
        groups=conv2d.groups,
    )

    with torch.no_grad():
        w2d = conv2d.weight.data  # [out, in, kH, kW]
        if method == "center":
            conv3d.weight.zero_()
            center = kernel_depth // 2
            conv3d.weight[:, :, center, :, :] = w2d
        elif method == "average":
            conv3d.weight[:] = w2d.unsqueeze(2).repeat(1, 1, kernel_depth, 1, 1) / kernel_depth
        else:
            raise ValueError(f"Unknown inflate method: {method}")

        if conv2d.bias is not None:
            conv3d.bias.data = conv2d.bias.data.clone()

    return conv3d


def adapt_input_conv(conv, in_channels):  
    old_weight = conv.weight.data
    out_channels, old_in, kD, kH, kW = old_weight.shape

    new_conv = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=(kD, kH, kW),
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
        groups=conv.groups,
    )

    with torch.no_grad():
        if in_channels == old_in:
            new_conv.weight.copy_(old_weight)
        elif in_channels == 1:
            new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
        elif in_channels > old_in:
            repeat = (in_channels + old_in - 1) // old_in
            new_conv.weight.copy_(old_weight.repeat(1, repeat, 1, 1, 1)[:, :in_channels])
        else:
            new_conv.weight.copy_(old_weight[:, :in_channels])

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


def inflate_convnext2d_to_3d(model2d, methods=None, input_channels=3, downsample_0_to_double=False, stage_kernel_depth=3):
    if methods is None:
        methods = {"downsample": "center", "stage": "center"}

    model3d = copy.deepcopy(model2d)

    for i, stage in enumerate(model3d.stages):
        # downsampling  
        if hasattr(stage, "downsample_layers"):
            for j, layer in enumerate(stage.downsample_layers):
                if isinstance(layer, nn.Conv2d):
                    
                    if downsample_0_to_double and i == 0:
                        mode = "change_to_double"
                    else:
                        mode = "normal"
                        
                    conv3d = inflate_conv2d_to_3d(
                        layer,
                        kernel_depth=4 if i == 0 else 2,
                        method=methods["downsample"],
                        is_downsample=mode,  
                    )
                    stage.downsample_layers[j] = conv3d

        # Stage  
        for j, block in enumerate(stage.layers):
            for name, sublayer in block.named_modules():
                if isinstance(sublayer, nn.Conv2d):
                    parent = block
                    *path, last = name.split(".")
                    for p in path:
                        parent = getattr(parent, p)
                    setattr(parent, last, inflate_conv2d_to_3d(
                        sublayer,
                        kernel_depth=3,
                        method=methods["stage"], 
                        # is_downsample=False 
                        is_downsample="No" 
                    ))

    
    for name, module in list(model3d.named_modules()):
        if isinstance(module, nn.LayerNorm):
            if "stages" in name:  
                ln3d = LayerNorm3d(module.normalized_shape, eps=module.eps)
                ln3d.weight.data.copy_(module.weight.data)
                ln3d.bias.data.copy_(module.bias.data)
                parent = model3d
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, ln3d)
    print("✅ Converted all LayerNorm → LayerNorm3d.")

    
    first_conv = None
    for layer in model3d.stages[0].downsample_layers:
        if isinstance(layer, nn.Conv3d):  
            first_conv = layer
            break

    if first_conv is not None and input_channels != first_conv.in_channels:
        print(f"Adapting downsample_layers[0] conv channels: {first_conv.in_channels} -> {input_channels}")
        new_conv = adapt_input_conv(first_conv, in_channels=input_channels)
        for i, layer in enumerate(model3d.stages[0].downsample_layers):
            if layer is first_conv:
                model3d.stages[0].downsample_layers[i] = new_conv
                break

    print("✅ Model inflated to 3D successfully.")
    return model3d