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

SAVE_VIDEO=False

sys.path.append('/mnt/ssd_990/teng/SuperGluePretrainedNetwork/')
from models.matching import Matching
# from nets.patchnet import Quad_L2Net 
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
torch.set_grad_enabled(False)


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

SAVE_VIDEO = True


import numpy as np
import trimesh
from trimesh.scene import Scene
from trimesh.viewer import SceneViewer


# repo = "isl-org/ZoeDepth"

# # Zoe_N
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

repo = "isl-org/ZoeDepth"

# Zoe_N
model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id, type='mask_visib')
    valid = mask>0
  elif detect_type=='cnos':   #https://github.com/nv-nguyen/cnos
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cnos'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError

  return valid


def create_foundation_pose(mesh):
    glctx = dr.RasterizeCudaContext()
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir="./debug",
        debug=1,
        glctx=glctx,
    )
    return est


def binary_search_scale(est,mesh, rgb,depth, mask, K, scale_min=0.2, scale_max=5):
    low=scale_min
    high=scale_max
    while low<=high:
        mid= (low+high)/2
        mesh_c=mesh.copy()
        mesh_c.apply_scale(mid)
        est.reset_object(model_pts=mesh_c.vertices.copy(), model_normals=mesh_c.vertex_normals.copy(), mesh=mesh_c)
        pose= est.register(K, rgb,depth, mask, 5)
        rgb_r, depth_r, mask_r= render_rgbd(mesh_c, pose, K, 640, 480)
        binary_mask = (mask_r > 0).astype(np.uint8)
    
        # Calculate the bounding box
        x, y, width, height = cv2.boundingRect(binary_mask)
        
        # Calculate the area of the bounding box
        area = width * height
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_r)
        # plt.subplot(1, 2, 2)
        # rgb_copy= rgb.copy()
        # rgb_copy[mask==0]=0
        # plt.imshow(rgb_copy)
        # plt.savefig(f"tmp2/debug_{mid}.png")

        if abs(high-low)<0.01:
            break
        if  abs(area-np.sum(mask))<20:
            break
        if area>np.sum(mask):
            high=mid
        elif area<np.sum(mask):
            low=mid
    return pose, mid





def binary_search_depth(est,mesh, rgb, mask, K, depth_min=0.2, depth_max=10, zoe_depth=None):
    low=depth_min
    high=depth_max
    best_pose=None
    while low <= high:
        mid= (low+high)/2
        # depth= np.ones_like(mask)*mid
        if zoe_depth is not None:
            depth= zoe_depth*mid
        else:
            depth= np.ones_like(mask)*mid
  
        pose= est.register(K, rgb, depth, mask, 5)
        
        best_pose=pose
        rgb_r, depth_r, mask_r= render_rgbd(mesh, pose, K, 640, 480)
        binary_mask = (mask_r > 0).astype(np.uint8)
    
        # Calculate the bounding box
        x, y, width, height = cv2.boundingRect(binary_mask)
        
        # Calculate the area of the bounding box
        area = width * height
            # depth=depth_r
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_r)
        # plt.subplot(1, 2, 2)
        # rgb_copy= rgb.copy()
        # rgb_copy[mask==0]=0
        # plt.imshow(rgb_copy)
        # plt.savefig(f"tmp2/debug_{mid}.png")
        if abs(high-low)<0.01:
            break
        if  abs(area-np.sum(mask))<20:
            break
        if area>np.sum(mask):
            low=mid
        elif area<np.sum(mask):
            high=mid
    logging.info(f"mid:{mid}")
    # exit(0)
    return best_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
    )
    parser.add_argument(
        "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/mustard0"
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    ycbv_dir="/mnt/ssd_990/teng/ycb/FoundationPose/ycbv"
    video_dirs = sorted(glob.glob(f'{ycbv_dir}/test/*'))
    os.environ["YCB_VIDEO_DIR"] = ycbv_dir
    reader_tmp = YcbVideoReader(video_dirs[0])
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    ob_id = 15
    mesh = reader_tmp.get_gt_mesh(ob_id)

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

    for video_dir in video_dirs:
        try:
            logging.info(f"video_dir: {video_dir}")
            reader= YcbVideoReader(video_dir, zfar=1.5)
            scene_ob_ids = reader.get_instance_ids_in_image(0)
            if ob_id in scene_ob_ids:
                break
        except:
            pass
    logging.info(f"video_dir: {video_dir}")
    
    last_depth = reader.get_depth(0)
    last_rgb = reader.get_color(0)
    
    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_color(i)

        if i == 0:
            # mask = reader.get_mask(0).astype(bool)
            mask= get_mask(reader, 0, ob_id, detect_type='box')
            last_mask= mask
            t1=time.time()
            initial_depth= reader.get_depth(0)

            # depth_numpy= zoe.infer_pil(color)
            # initial_depth= depth_numpy*10
            depth_numpy= None
            pose= binary_search_depth(est, mesh, color, mask, reader.K, zoe_depth=depth_numpy)
            # pose, scale= binary_search_scale(est,mesh, color, initial_depth, mask, reader.K)
            # mesh.apply_scale(scale)
            est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), mesh=mesh)
            est.register(K=reader.K, rgb=color, depth=initial_depth, ob_mask=mask, iteration=args.est_refine_iter)
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
            # initial_depth=np.ones_like(mask)*1.0423754453659058
            # pose = est.register(
            #     K=reader.K,
            #     rgb=color,
            #     depth=depth_numpy,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            logging.info(f"Initial pose:\n{pose}")
            t2=time.time()
          

            if SAVE_VIDEO:
                output_video_path = "foundation_pose.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video
                # Assuming 'color' is the image shape (height, width, channels)
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))
        else:
            t1=time.time()
          
            prediction= last_mask
            # last_depth= optical_flow_get_depth(last_rgb.copy(), last_depth, last_mask, color.copy(), prediction)
            last_depth = np.zeros_like(initial_depth)
    


            pose = est.track_one(
                rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter
            )
            
            t2=time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color=cv2.putText(color, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            vis = draw_posed_3d_box(
                reader.K, img=color, ob_in_cam=center_pose, bbox=bbox
            )
            vis = draw_xyz_axis(
                color,
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