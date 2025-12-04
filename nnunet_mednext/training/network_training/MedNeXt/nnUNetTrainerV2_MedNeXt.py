import torch
import os
import torch.nn as nn
# from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
from nnunet_mednext.network_architecture.mednextv1_neurino.NeurINO import NeurINO as NeurINO_Orig
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class NeurINO(NeurINO_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['n_classes']
        # self.do_ds = False        Already added this in the main class

class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainerV2):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]    
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None


class NeurINO_CenterInfla_SGL_T_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = NeurINO(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            use_dino_encoder = True,   
            freeze_dino = False,
            dino_downsample0_to_double = True,  
            inflation_type = 'center',
            use_mednext_bottleneck = True,   
            batchrenorm3d_replace_scope = 'all_except_dino', 
            use_skeleton_graph_loss = True,     
            skeleton_graph_loss_weight = 0.1,  
            block_counts = [2,2,2,2,2,2,2,2,2], 
            # checkpoint_style = 'outside_block'  # no 
        )

        if torch.cuda.is_available():
            self.network.cuda()
            
            
class NeurINO_AvgInfla_SGL_T_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = NeurINO(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            use_dino_encoder = True,   
            freeze_dino = False,
            dino_downsample0_to_double = True,  
            inflation_type = 'average',
            use_mednext_bottleneck = True,   
            batchrenorm3d_replace_scope = 'all_except_dino', 
            use_skeleton_graph_loss = True,     
            skeleton_graph_loss_weight = 0.1,  
            block_counts = [2,2,2,2,2,2,2,2,2], 
            # checkpoint_style = 'outside_block'  # no 
        )

        if torch.cuda.is_available():
            self.network.cuda()
            

class NeurINO_CenterInfla_SGL_S_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = NeurINO(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            use_dino_encoder = True,  
            dino_model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m",  
            freeze_dino = False,
            dino_downsample0_to_double = True,  
            inflation_type = 'center',
            use_mednext_bottleneck = True,   
            batchrenorm3d_replace_scope = 'all_except_dino', 
            use_skeleton_graph_loss = True,     
            skeleton_graph_loss_weight = 0.1,  
            block_counts = [2,2,2,2,2,2,2,2,2], 
            # checkpoint_style = 'outside_block'  # no 
        )

        if torch.cuda.is_available():
            self.network.cuda()
            
            
class NeurINO_AvgInfla_SGL_S_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = NeurINO(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            use_dino_encoder = True,   
            dino_model_name = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            freeze_dino = False,
            dino_downsample0_to_double = True,  
            inflation_type = 'average',
            use_mednext_bottleneck = True,   
            batchrenorm3d_replace_scope = 'all_except_dino', 
            use_skeleton_graph_loss = True,     
            skeleton_graph_loss_weight = 0.1,  
            block_counts = [2,2,2,2,2,2,2,2,2], 
            # checkpoint_style = 'outside_block'  # no 
        )

        if torch.cuda.is_available():
            self.network.cuda()
