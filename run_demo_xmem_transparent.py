# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import matplotlib.cm as cm
import time
from estimater import *
from datareader import *
import argparse
import pyrender
import trimesh
from tools import get_3d_points
USE_XMEM = True
SAVE_VIDEO=False
XMEM_PATH = '/mnt/ssd_990/teng/XMem'
sys.path.append('/mnt/ssd_990/teng/SuperGluePretrainedNetwork/')
from models.matching import Matching
# from nets.patchnet import Quad_L2Net 
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
torch.set_grad_enabled(False)
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



import numpy as np
import trimesh
from trimesh.scene import Scene
from trimesh.viewer import SceneViewer
green_color = np.array([0.0, 1.0, 0.0, 1.0])
green_color2 =np.array([0,255,0],dtype=np.uint8)
repo = "isl-org/ZoeDepth"

# Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# zoe = model_zoe_n.to(DEVICE)



def render_rgbd(cad_model, object_pose, K, W, H):
    pose_tensor= torch.from_numpy(object_pose).float().to("cuda")

    mesh_tensors = make_mesh_tensors(cad_model)
   
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    rgb_r = rgb_r.squeeze().cpu().numpy()
    depth_r = depth_r.squeeze().cpu().numpy()
    mask_r = (depth_r > 0)
    return rgb_r, depth_r, mask_r

def render_cad_depth(pose, mesh_model,K):
    h=480
    w=640
    # Load the mesh model and apply the center pose transformation
    
    vertices = np.array(mesh_model.vertices)
    
    # Transform vertices with the center pose
    transformed_vertices = (pose @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T).T[:, :3]

    # Project vertices to the 2D plane using the intrinsic matrix K
    projected_points = (K @ transformed_vertices.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # Normalize by z

    # Initialize a depth map and project depth values into it
    image_size = (h, w)
    depth_map = np.zeros(image_size, dtype=np.float32)
    for i, point in enumerate(projected_points):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            depth_map[y, x] = transformed_vertices[i, 2]
    return depth_map

import numpy as np
import trimesh
import pyrender
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def load_super_glue():
    # Load SuperGlue
    superglue = Matching()
    superglue.eval()
    superglue = superglue.cuda()
    return superglue


def binary_search_depth(est, rgb, mask, K, depth_min=0.2, depth_max=20):
    low=depth_min
    high=depth_max
    w,h=544, 960
    while low <= high:
        mid= (low+high)/2
        depth= np.ones_like(mask)*mid
        pose= est.register(K, rgb, depth, mask, 5)
        rgb_r, depth_r, mask_r= render_rgbd(mesh, pose, K, w, h)
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_r)
        plt.subplot(1, 2, 2)
        rgb_copy= rgb.copy()
        rgb_copy[mask==0]=0
        plt.imshow(rgb_copy)
        plt.savefig(f"tmp/debug_{mid}.png")
        if abs(high-low)<0.001:
            return pose
        
        if  abs(np.sum(mask_r)-np.sum(mask))<10:
            logging.info(f"mid: {mid}")
            a=input("Press enter to continue")
            return pose
        if np.sum(mask_r)>np.sum(mask):
            low=mid
        elif np.sum(mask_r)<np.sum(mask):
            high=mid
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        # default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
        default="/mnt/ssd_990/teng/ycb/FoundationPose/demo_data/bottle/mesh/ee4ae2457ec14af388c90e1561f68ac9.obj",
    )
    parser.add_argument(
        # "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/mustard0"
        "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/bottle"
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=3)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    # Remove any existing texture/material by resetting the visual attribute
    mesh.visual = trimesh.visual.ColorVisuals(mesh)
    mesh.visual.vertex_colors = green_color

    mesh.show()
    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam"
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("estimator initialization done")

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf
    )
    
    
    if USE_XMEM:
        network = XMem(config, f'{XMEM_PATH}/saves/XMem.pth').eval().to("cuda")
        processor=InferenceCore(network, config=config)
        processor.set_all_labels(range(1,2))
        #You can change these values to get different results
        frames_to_propagate = 1000
    # last_depth = reader.get_depth(0)
    # last_rgb = reader.get_color(0)
    
    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_color(i)
        color_copy= color.copy()
        if USE_XMEM:
            frame_torch, _ = image_to_torch(color, device="cuda")
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            last_mask= mask
            
            color[mask]=green_color2
            t1=time.time()
            # initial_depth= reader.get_depth(0)
            # depth_numpy= zoe.infer_pil(color)*0.2
            # depth_numpy=1.9*np.ones_like(initial_depth)
            pose= binary_search_depth(est, color, mask, reader.K)
            
            # pose = est.register(
            #     K=reader.K,
            #     rgb=color,
            #     depth=depth_numpy,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            
            logging.info(f"Initial pose:\n{pose}")
            # coarse_estimate(est, mesh, color,mask,reader.K, to_origin, bbox)
            
            
            # rgb, depth, mask = render_rgbd(mesh, pose,reader.K, 640, 480)
            # logging.info(f"depth shape: {depth.shape}, np.max(depth): {np.max(depth)}")

       
        
            t2=time.time()
            
            if USE_XMEM:
                mask_png= reader.get_mask(0)
                mask_torch= index_numpy_to_one_hot_torch(mask_png, 2).to("cuda")
                prediction= processor.step(frame_torch, mask_torch[1:])


            if SAVE_VIDEO:
                output_video_path = "foundation_pose.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video
                # Assuming 'color' is the image shape (height, width, channels)
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))
        else:
            t1=time.time()
            if USE_XMEM:
                prediction = processor.step(frame_torch)
                
                prediction= torch_prob_to_numpy_mask(prediction)
                mask= (prediction==1)
                color[mask]=green_color2

            else:
                prediction= last_mask
            # last_depth= optical_flow_get_depth(last_rgb.copy(), last_depth, last_mask, color.copy(), prediction)
            last_depth = np.zeros_like(last_mask)
           
            t3=time.time()
            pose = est.track_one(
                rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter
            )
            t4=time.time()
            logging.info(f"time for xmem: {t3-t1}, time for track_one: {t4-t3}")
            
            t2=time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color_copy=cv2.putText(color_copy, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            vis = draw_posed_3d_box(
                reader.K, img=color_copy, ob_in_cam=center_pose, bbox=bbox
            )
            vis = draw_xyz_axis(
                color_copy,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            

            cv2.imshow("1", vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f"{debug_dir}/track_vis", exist_ok=True)
            imageio.imwrite(f"{debug_dir}/track_vis/{reader.id_strs[i]}.png", vis)
        if SAVE_VIDEO:
            video_writer.write(vis[..., ::-1])