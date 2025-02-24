import matplotlib.cm as cm
import time
from estimater import *
from datareader import *
import argparse
import pyrender
import trimesh
from tools import get_3d_points
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

XMEM_PATH = '/mnt/ssd_990/teng/XMem'
sys.path.append(XMEM_PATH)
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

# default configuration
config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}



from progressbar import progressbar
torch.set_grad_enabled(False)
torch.cuda.empty_cache()


class yolo_wrapper:
    def __init__(self):
        self.model= YOLO("yolo11n-seg.pt")  # segmentation model

    def initialize(self, rgb, mask):
        annotator=Annotator(rgb, line_width=2)