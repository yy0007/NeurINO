import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from types import MethodType
from transformers import AutoConfig, AutoModel, AutoImageProcessor 

from nnunet_mednext.network_architecture.neurino.blocks import *
from nnunet_mednext.network_architecture.neurino.inflate_dino_convnext2d_to_3d import *
from transformers.models.dinov3_convnext.modeling_dinov3_convnext import DINOv3ConvNextLayer

def dinov3_convnext_layer_forward_3d(self, features: torch.Tensor) -> torch.Tensor:
    if features.dim() != 5:
        raise ValueError(f"Expected 5D tensor [B, C, D, H, W], got shape: {features.shape}")

    residual = features                        # [B, C, D, H, W]
    features = self.depthwise_conv(features)   # [B, C, D, H, W]
    features = features.permute(0, 2, 3, 4, 1)
    features = self.layer_norm(features)       # [B, D, H, W, C]
    features = self.pointwise_conv1(features)  # [B, D, H, W, 4C]
    features = self.activation_fn(features)
    features = self.pointwise_conv2(features)  # [B, D, H, W, C]
    features = features * self.gamma
    
    # back to [B, C, D, H, W]
    features = features.permute(0, 4, 1, 2, 3)
    features = residual + self.drop_path(features)

    return features

def patch_dinov3_blocks_to_3d(model3d):
    for module in model3d.modules():
        if isinstance(module, DINOv3ConvNextLayer):
            module.forward = MethodType(dinov3_convnext_layer_forward_3d, module)
    print("✅ Patched all DINOv3ConvNextLayer.forward → 3D version.")
    return model3d

def forward_features_multi(self, x):
    features = []
    for stage in self.stages:
        # print('forward_features_multi shape:', x.shape) 
        x = stage(x)
        features.append(x)
    return features 


class NeurINO(nn.Module):    

    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        use_dino_encoder: bool = False,   # Use dino encoder or not 
        dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",   
        freeze_dino: bool = False,   
        dino_train_from_scratch: bool = False,   
        dino_stage_kernelDepth: int = 3,
        dino_downsample0_to_double: bool = False, 
        inflation_type: str = 'center', 
        use_mednext_bottleneck: bool = False,  
        batchrenorm3d_replace_scope: str = None,  # all, dino_only, all_except_dino, decoder_only
        norm_replace_type: str = 'BatchRenorm3d', 
        rmax: float = 3.,
        dmax: float = 5.,
        momentum: float = 0.01, 
        use_skeleton_loss: bool = False,     
        use_skeleton_graph_loss: bool = False,     
        skeleton_loss_weight: float = 0.2,   
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet 
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
    ):

        super().__init__()

        self.do_ds = deep_supervision
        self.dino_downsample0_to_double = dino_downsample0_to_double
        self.use_mednext_bottleneck = use_mednext_bottleneck
        self.is_val = True  
        self.use_skeleton_loss = use_skeleton_loss
        self.use_skeleton_graph_loss = use_skeleton_graph_loss 
        
        self.skeleton_loss_weight = skeleton_loss_weight 
        
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']
        
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]
            
            
        # -----------------------------
        # 🔹 DINOv3 encoder
        # -----------------------------
        if use_dino_encoder:
            print(f"⚙️  Loading pretrained DINOv3 backbone: {dino_model_name}")
            # dino2d = Dinov3Model.from_pretrained(dino_model_name) 
            if dino_train_from_scratch: 
                print(f"⚙️ Training DINOv3 {dino_model_name} from scratch")
                config = AutoConfig.from_pretrained(dino_model_name)
                dino2d = AutoModel.from_config(config)
                if hasattr(dino2d, "layer_norm"):  
                    del dino2d.layer_norm
                if hasattr(dino2d, "pool"):      
                    del dino2d.pool
                print("✅ Removed layer_norm and pool from DINO backbone.")
            else:
                print(f"⚙️ Transfer pretrained DINOv3 backbone: {dino_model_name}")   
                dino2d = AutoModel.from_pretrained(dino_model_name)                         
            # dino2d = AutoModel.from_pretrained(dino_model_name)   
            dino3d = inflate_convnext2d_to_3d(dino2d, methods={'downsample': inflation_type, 'stage': inflation_type}, input_channels=in_channels, downsample_0_to_double=dino_downsample0_to_double, stage_kernel_depth=dino_stage_kernelDepth)
            dino3d.forward_features_multi = MethodType(forward_features_multi, dino3d)
            print('✅ dino3d framework:', dino3d)
            self.encoder = dino3d  

            # freeze dino 
            if freeze_dino:
                for p in self.encoder.parameters():
                    p.requires_grad = False

            dino_dims = [96, 192, 384, 768]
            # target_dims = [n_channels, 2*n_channels, 4*n_channels, 8*n_channels]
            target_dims = [2*n_channels, 4*n_channels, 8*n_channels, 8*n_channels]

        else:
            raise NotImplementedError("No DINOv3 encoder。")
        

        if use_mednext_bottleneck: 
            self.bottleneck = nn.Sequential(*[
                MedNeXtBlock(  
                    in_channels=1024 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 768,  # 768:用mednext bottleneck，除非前面先upsample了.   Defau: n_channels*16 --> 512 
                    out_channels=1024 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 768 ,  #  768:用mednext bottleneck.   Defau: n_channels*16 --> 512
                    exp_r=exp_r[4],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,  # # "layer" 
                    dim=dim,
                    grn=grn
                    )
                for i in range(block_counts[4])]
            )
        

        self.up_3 = MedNeXtUpBlock(
            in_channels=1024 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 768,  # 768 / 8*n_channels: 如果用dino值做 bottleneck。 768: 用 mednext bottleneck。Defau: n_channels*16 
            out_channels=512 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 384,  # 8*n_channels 
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,  # "layer"
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(  
                in_channels=512 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 384,  # n_channels*8
                out_channels=512 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 384,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[5])]
        )
        

        self.up_2 = MedNeXtUpBlock(
            in_channels=512 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 384,  # 8*n_channels
            out_channels=256 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 192,  #  4*n_channels 
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(  
                in_channels=256 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 192,  # 4*n_channels
                out_channels=256 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 192,  # 4*n_channels
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=256 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 192,  # 4*n_channels 
            out_channels=128 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 96,  # 2*n_channels
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock( 
                in_channels=128 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 96,  # 2*n_channels
                out_channels=128 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 96,  # 4*n_channels
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=128 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 96,  # 2*n_channels
            out_channels=64 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 48,  # n_channels
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(  
                in_channels=64 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 48,  # n_channels
                out_channels=64 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 48,  # n_channels
                exp_r=exp_r[8], 
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[8])]
        )
        
        if not dino_downsample0_to_double: 
            print('~~~~~ We have up_extra!!!')
            self.up_extra = MedNeXtUpBlock(
                in_channels=64 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 48,
                out_channels=32 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 24,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res_up_down,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )


        if dino_downsample0_to_double: 
            self.out_0 = OutBlock(in_channels=64 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 48, n_classes=n_classes, dim=dim)  # n_channels
        else:
            self.out_0 = OutBlock(in_channels=32 if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m" else 24, n_classes=n_classes, dim=dim)  # n_channels

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)  

        if deep_supervision:           
            if dino_downsample0_to_double: 
                print('~~~~~ dino_downsample0_to_double !!!')
                if dino_model_name == "facebook/dinov3-convnext-base-pretrain-lvd1689m": 
                    self.out_1 = OutBlock(in_channels=128, n_classes=n_classes, dim=dim)
                    self.out_2 = OutBlock(in_channels=256, n_classes=n_classes, dim=dim)
                    self.out_3 = OutBlock(in_channels=512, n_classes=n_classes, dim=dim)
                    self.out_4 = OutBlock(in_channels=1024, n_classes=n_classes, dim=dim)
                else:
                    self.out_1 = OutBlock(in_channels=96, n_classes=n_classes, dim=dim)
                    self.out_2 = OutBlock(in_channels=192, n_classes=n_classes, dim=dim)
                    self.out_3 = OutBlock(in_channels=384, n_classes=n_classes, dim=dim)
                    self.out_4 = OutBlock(in_channels=768, n_classes=n_classes, dim=dim) 
            else:
                self.out_1 = OutBlock(in_channels=48, n_classes=n_classes, dim=dim)
                self.out_2 = OutBlock(in_channels=96, n_classes=n_classes, dim=dim)
                self.out_3 = OutBlock(in_channels=192, n_classes=n_classes, dim=dim)
                self.out_4 = OutBlock(in_channels=384, n_classes=n_classes, dim=dim) 

        self.block_counts = block_counts
        
        if batchrenorm3d_replace_scope:
            print(f"✅ Applying {norm_replace_type} (scope: {batchrenorm3d_replace_scope})") 
            self._replace_norm_layers(
                scope=batchrenorm3d_replace_scope,   # 'decoder_only' / 'all_except_dino' / 'dino_only' / 'all'
                factory=norm_replace_type,
                rmax=rmax, dmax=dmax, momentum=momentum
            )
            
     
    def _replace_norm_layers(
        self,
        module=None,
        parent_name="",
        scope="all_except_dino",
        factory="BatchRenorm3d",
        rmax=3.0,
        dmax=5.0,
        momentum=0.01,
    ):
        import torch.nn as nn

        if module is None:
            module = self
            print(f"✅ Applying {factory} (scope: {scope})")
            print(f"🔧 [Normalization Replacement] scope={scope}, type={factory}")

        def _get_num_channels(m):
            if hasattr(m, "num_features"):
                return int(m.num_features)
            if hasattr(m, "num_channels"):
                return int(m.num_channels)
            norm_shape = getattr(m, "normalized_shape", None)
            if isinstance(norm_shape, (tuple, list)) and len(norm_shape) > 0:
                return int(norm_shape[0])
            if hasattr(m, "weight") and m.weight is not None and len(m.weight.shape) > 0:
                return int(m.weight.shape[0])
            return None

        def _is_norm_like(m):
            cname = m.__class__.__name__.lower()
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm3d, nn.InstanceNorm3d, nn.LayerNorm)):
                return True
            if "layernorm" in cname or "norm3d" in cname:
                return True
            return False

        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            # --- scope filter ---
            if scope == "decoder_only":
                if not any(tag in full_name for tag in ("dec_", "up_", "out_")):
                    self._replace_norm_layers(child, full_name, scope, factory, rmax, dmax, momentum)
                    continue
            elif scope == "all_except_dino":
                if full_name.startswith("encoder"):
                    self._replace_norm_layers(child, full_name, scope, factory, rmax, dmax, momentum)
                    continue
            elif scope == "dino_only":
                if not full_name.startswith("encoder"):
                    self._replace_norm_layers(child, full_name, scope, factory, rmax, dmax, momentum)
                    continue

            # --- replace Norm ---
            if _is_norm_like(child):
                C = _get_num_channels(child)
                if C is None:
                    print(f"⚠️ Skipped {full_name}: unable to infer channels.")
                    continue

                cname = child.__class__.__name__.lower()

                if "layernorm3d" in cname:
                    new_norm = ChannelRenorm3d_Flexible(C, momentum=momentum, rmax=rmax, dmax=dmax)
                    tag = "ChannelRenorm3d_Flexible"
                else:
                    new_norm = BatchRenorm3d(C, momentum=momentum, rmax=rmax, dmax=dmax)
                    tag = "BatchRenorm3d"

                print(f"🔁 Replacing {full_name}: {child.__class__.__name__} → {tag}({C})")
                setattr(module, name, new_norm)
            else:
                self._replace_norm_layers(child, full_name, scope, factory, rmax, dmax, momentum)
            
            
    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x)  # self.dummy_tensor
        return x
        
        
    def forward(self, x, return_hard_tp_fp_fn=False):  
        """
        Forward pass with MoE load balance loss collection.
        """
        all_load_balance_losses = []  # For encoder with moe 
        print('training x shape:', x.shape)

        # --- Encoder ---
        feats = self.encoder.forward_features_multi(x)
        
        x_res_1, x_res_2, x_res_3, x_res_4 = feats
        print('feats x_res_3 shape:', x_res_3.shape)
        print('feats[-1] shape:', feats[-1].shape)
        x = feats[-1]   # [1, 768, 1, 1, 1] 

        if self.training and self.outside_block_checkpointing:
            
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            # Decoder stages
            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            res = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            res = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            res = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            res = self.iterative_checkpoint(self.dec_block_0, dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:            
            # bottleneck  *******
            if self.use_mednext_bottleneck:
                print('~~~~~ use_mednext_bottleneck output')  
                res = self.bottleneck(x)
                if isinstance(res, tuple):
                    x, lb_loss = res
                    all_load_balance_losses.append(lb_loss)
                else:
                    x = res

            if self.do_ds:
                if self.dino_downsample0_to_double:
                    x_ds_4 = self.out_4(x)   


            x_up_3 = self.up_3(x)   # x_res_1, x_res_2, x_res_3, x_res_4 = feats
            dec_x = x_res_3 + x_up_3
            res = self.dec_block_3(dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            if self.do_ds:
                if self.dino_downsample0_to_double:
                    x_ds_3 = self.out_3(x)  # ********
                    # print('x_res_3 x shape:', x.shape)
                else:
                    x_ds_4 = self.out_4(x)
                    # print('x_ds_4 shape:', x_ds_4.shape)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            res = self.dec_block_2(dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res

            if self.do_ds:
                if self.dino_downsample0_to_double:
                    x_ds_2 = self.out_2(x)  # *******
                else:
                    x_ds_3 = self.out_3(x)   
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            res = self.dec_block_1(dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            if self.do_ds:
                if self.dino_downsample0_to_double:
                    x_ds_1 = self.out_1(x) # ******** 
                else:
                    x_ds_2 = self.out_2(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_up_0
            res = self.dec_block_0(dec_x)
            if isinstance(res, tuple):
                x, lb_loss = res
                all_load_balance_losses.append(lb_loss)
            else:
                x = res
            if self.do_ds:
                if self.dino_downsample0_to_double:
                   pass
                else:
                    x_ds_1 = self.out_1(x) 
            del x_up_0, dec_x
            

            if self.dino_downsample0_to_double:
                pass
            else:
                x = self.up_extra(x)

            x = self.out_0(x)
            
        # if self.training:   # *******
        if self.training:
            if self.do_ds:
                return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4], all_load_balance_losses
            else:
                return x, all_load_balance_losses
            
        elif self.is_val:
            # --- val ---
            if self.do_ds:
                return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4], all_load_balance_losses   
            else:
                return x, all_load_balance_losses                                     
       
        else:
            if self.do_ds:
                return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]  # , []   
            else:
                return x  # , [] 


if __name__ == "__main__":
    
    network = NeurINO(  # DinoEncBot_MedNeXt, MedNeXt
            in_channels = 1, 
            n_channels = 32,
            n_classes = 2,   
            exp_r = 2,
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True, 
            use_dino_encoder = True, 
            dino_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",  # tiny, small, base
            dino_train_from_scratch = False, 
            dino_downsample0_to_double = True,  
            use_mednext_bottleneck = True,  
            batchrenorm3d_replace_scope  = 'all_except_dino',  # all, dino_only, all_except_dino, decoder_only
            block_counts = [2,2,2,2,2,2,2,2,2],
            checkpoint_style = None,
            dim = '3d',   # 2d
            grn=True
            
        ).cuda().eval()
    

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1,1,64,64,64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())
    
    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)
