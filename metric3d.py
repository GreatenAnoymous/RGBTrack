import os

dependencies = ['torch', 'torchvision']
import matplotlib.pyplot as plt
import os
import sys
import cv2
import torch
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
    
## add Metric3D to sys.path, modify the path to your Metric3D directory
sys.path.append("/mnt/ssd_990/teng/BinPicking/Metric3D/") 
metric3d_dir="/mnt/ssd_990/teng/BinPicking/Metric3D/"
from mono.model.monodepth_model import get_configured_monodepth_model



MODEL_TYPE = {
    'ConvNeXt-Tiny': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
    },
    'ConvNeXt-Large': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
    },
    'ViT-Small': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
    },
    'ViT-Large': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
    },
    'ViT-giant2': {
        'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
    },
}



def metric3d_vit_large(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
        pretrain (bool): whether to load pretrained weights.
    Returns:
        model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
        torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
        strict=False,
        )
    return model



def metric3d_vit_small(pretrain=False, **kwargs):
    '''
    Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
    For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
    Args:
        pretrain (bool): whether to load pretrained weights.
    Returns:
        model (nn.Module): a Metric3D model.
    '''
    cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
    ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

    cfg = Config.fromfile(cfg_file)
    model = get_configured_monodepth_model(cfg)
    if pretrain:
        model.load_state_dict(
        torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
        strict=False,
        )
    return model


class Metric3D:
    def __init__(self, model_type='ViT-Large', pretrain=False):
        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        self.model.cuda().eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, rgb_origin, intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]):
        rgb_origin=rgb_origin[:,:,::-1]
        with torch.no_grad():
            input_size = (616, 1064) # for vit model
        
            h, w = rgb_origin.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            # remember to scale intrinsic, hold depth
            intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
            # padding to input_size
            padding = [123.675, 116.28, 103.53]
            h, w = rgb.shape[:2]
            pad_h = input_size[0] - h
            pad_w = input_size[1] - w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
            pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

            #### normalize
            mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
            std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
            rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
            rgb = torch.div((rgb - mean), std)
            rgb = rgb[None, :, :, :].cuda()

            
            
            ###################### canonical camera space ######################
            # inference
            
            with torch.no_grad():
                pred_depth, confidence, output_dict = self.model.inference({'input': rgb})

            # un pad
            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
            
            # upsample to original size
            pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
            ###################### canonical camera space ######################

            #### de-canonical transform
            canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300)

        return pred_depth.cpu().numpy()
    
    def infer_pil(self, rgb_origin, intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]):
        return self.predict(rgb_origin, intrinsic)



    def __call__(self, rgb):
        return self.predict(rgb)

    def __repr__(self):
        return f'Metric3D(model={self.model.__class__.__name__})'

    def __str__(self):
        return self.__repr__()
    



class ZoedepthWrapper:
    def __init__(self):
        repo = "isl-org/ZoeDepth"
        self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, rgb):
        with torch.no_grad():
            depth_numpy=self.model.infer_pil(rgb)
        return depth_numpy

if __name__ == "__main__":
    # model = Metric3D()
    # rgb_file = '/mnt/ssd_990/teng/BinPicking/DPT_transparent_objects/metric_depth/data/nyu/test/010/90_color.png'
    # img = cv2.imread(rgb_file)
    # depth = model.predict(img)
    # plt.imshow(depth, cmap='jet')
    # plt.colorbar()
    # plt.savefig('pred_depth.png')


    model=ZoedepthWrapper()
    rgb_file = '/mnt/ssd_990/teng/BinPicking/DPT_transparent_objects/metric_depth/data/nyu/test/010/90_color.png'
    img = cv2.imread(rgb_file)
    depth = model.predict(img)
    plt.imshow(depth, cmap='jet')
    plt.colorbar()
    plt.savefig('pred_depth.png')