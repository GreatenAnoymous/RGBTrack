import torch
import torch.onnx
from onnx import load_model, save_model
import onnx
from omegaconf import OmegaConf
from onnxmltools.utils import float16_converter
import os
import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
print(code_dir)
sys.path.append(code_dir)
sys.path.append(f'/mnt/ssd_990/teng/ycb/fp/FoundationPose')
from refine_network import RefineNet
import numpy as np
import onnxscript
from onnxscript.onnx_opset import opset17 as op
from score_network import ScoreNetMultiPair
print(torch.__version__)

class RefinerConverter:
    def __init__(self):

        self.amp = True
        self.run_name = "2023-10-28-18-33-37"
        model_name = 'model_best.pth'

        ckpt_dir = f'../../weights/{self.run_name}/{model_name}'

        self.cfg = OmegaConf.load(f'../../weights/{self.run_name}/config.yml')

        self.cfg['ckpt_dir'] = ckpt_dir
        self.cfg['enable_amp'] = True

        ########## Defaults, to be backward compatible
        if 'use_normal' not in self.cfg:
            self.cfg['use_normal'] = False
        if 'use_mask' not in self.cfg:
            self.cfg['use_mask'] = False
        if 'use_BN' not in self.cfg:
            self.cfg['use_BN'] = False
        if 'c_in' not in self.cfg:
            self.cfg['c_in'] = 4
        if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
            self.cfg['crop_ratio'] = 1.2
        if 'n_view' not in self.cfg:
            self.cfg['n_view'] = 1
        if 'trans_rep' not in self.cfg:
            self.cfg['trans_rep'] = 'tracknet'
        if 'rot_rep' not in self.cfg:
            self.cfg['rot_rep'] = 'axis_angle'
        if 'zfar' not in self.cfg:
            self.cfg['zfar'] = 3
        if 'normalize_xyz' not in self.cfg:
            self.cfg['normalize_xyz'] = False
        if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
            self.cfg['zfar'] = np.inf
        if 'normal_uint8' not in self.cfg:
            self.cfg['normal_uint8'] = False
        self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
        ckpt= torch.load(ckpt_dir)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        self.model.load_state_dict(ckpt)
        self.model.cuda().eval()
        
        

    def convert(self):
        
        # Define example input (assuming A and B are input tensors)
        # Example shapes for A and B: batch_size=1, channels=4, height=224, width=224
        A = torch.randn(252, 6, 160, 160).cuda()  # Example tensor for A
        B = torch.randn(252, 6, 160, 160).cuda() # Example tensor for B

        # Export the model to ONNX format
        onnx_path = "refinenet.onnx"
        torch.onnx.export(
            self.model,                           # Model to be exported
            (A, B),                          # Example input tuple
            onnx_path,                       # Path to save the model
            input_names=["A", "B"],          # Name of input nodes\
            export_params=True,              # Export model parameters
            opset_version=17,                # ONNX version (adjust if needed)
            custom_opsets = {"torch.onnx": 17},  
            
            do_constant_folding=True,        # Fold constant values into the model
            output_names=["trans", "rot"],   # Name of output nodes              # ONNX version (adjust if needed)
            dynamic_axes={"A": {0: "batch_size"}, "B": {0: "batch_size"}, "trans": {0: "batch_size"}, "rot": {0: "batch_size"}}
        )

        print(f"Model successfully exported to {onnx_path}")
        
        onnx_model = load_model(onnx_path)
        onnx.checker.check_model(onnx_model)
        trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
        
        save_model(trans_model, "refinenet.onnx")


class ScorerConverter:
    def __init__(self):
        self.amp = True
        self.run_name = "2024-01-11-20-02-45"
        model_name = 'model_best.pth'

        ckpt_dir = f'../../weights/{self.run_name}/{model_name}'

        self.cfg = OmegaConf.load(f'../../weights/{self.run_name}/config.yml')

        self.cfg['ckpt_dir'] = ckpt_dir
        self.cfg['enable_amp'] = True

        ########## Defaults, to be backward compatible
        if 'use_normal' not in self.cfg:
            self.cfg['use_normal'] = False
        if 'use_mask' not in self.cfg:
            self.cfg['use_mask'] = False
        if 'use_BN' not in self.cfg:
            self.cfg['use_BN'] = False
        if 'c_in' not in self.cfg:
            self.cfg['c_in'] = 4
        if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
            self.cfg['crop_ratio'] = 1.2
        if 'n_view' not in self.cfg:
            self.cfg['n_view'] = 1
        if 'trans_rep' not in self.cfg:
            self.cfg['trans_rep'] = 'tracknet'
        if 'rot_rep' not in self.cfg:
            self.cfg['rot_rep'] = 'axis_angle'
        if 'zfar' not in self.cfg:
            self.cfg['zfar'] = 3
        if 'normalize_xyz' not in self.cfg:
            self.cfg['normalize_xyz'] = False
        if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
            self.cfg['zfar'] = np.inf
        if 'normal_uint8' not in self.cfg:
            self.cfg['normal_uint8'] = False
        self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()
        ckpt= torch.load(ckpt_dir)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.inference_mode()
    def convert(self, export_path="score_network.onnx"):
        # Create dummy input for ONNX export, matching the expected input size
        # You may need to adjust the input shape according to your model's expected dimensions
        dummy_A = torch.randn(252, 6, 160, 160).cuda()  # Example dimensions
        dummy_B = torch.randn(252, 6, 160, 160).cuda()  # Example dimensions


        # Define dynamic axes for the ONNX model
        # dynamic_axes = {
        #     'A': {0: 'batch_size'},
        #     'B': {0: 'batch_size'},
        #     "score_logit": {1: 'batch_size'}
        #     # 'L': {0: 'batch_size'}
        # }

        # Export the model to ONNX format
        torch.onnx.export(
            self.model,
            (dummy_A, dummy_B),
            export_path,
            input_names=['A', 'B'],
            output_names=['score_logit'],
            dynamic_axes=None,
            # dynamic_axes=dynamic_axes,
            opset_version=17,  # You can adjust the opset version if needed
            verbose=True
        )
        onnx_model = load_model(export_path)
        trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=True)
        
        save_model(trans_model, "refinenet.onnx")
        print(f"Model has been successfully exported to {export_path}")

if __name__ == "__main__":
    converter = RefinerConverter()
    # converter= ScorerConverter()
    converter.convert()