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
import trimesh
from tools import get_3d_points
from binary_search_adjust import *
import numpy as np
import pyrender
import open3d as o3d
from scipy.spatial.transform import Rotation as R

SAVE_VIDEO=True
# repo = "isl-org/ZoeDepth"

# # Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# zoe = model_zoe_n.to(DEVICE)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
        # default="/mnt/ssd_990/teng/ycb/FoundationPose/demo_data/cola_can/mesh/32429d6d7cd54f8392f9b3056a1f26c3.obj",
    )
    parser.add_argument(
        "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/mustard0"
        # "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/cola_can"
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

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
    
    
    
    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_color(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            last_mask= mask
            t1=time.time()
            # initial_depth= reader.get_depth(0)
            # depth_numpy= zoe.infer_pil(color)
            depth_numpy=np.ones_like(mask)
            pose= binary_search_depth(est, mesh, color, mask, reader.K,debug=True)
            # pose = est.register(
            #     K=reader.K,
            #     rgb=color,
            #     depth=depth_numpy,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            # pose = est.register_without_depth(
            #     K=reader.K,
            #     rgb=color,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            logging.info(f"Initial pose:\n{pose}")
            # coarse_estimate(est, mesh, color,mask,reader.K, to_origin, bbox)
            # rgb, depth, mask = render_rgbd(mesh, pose,reader.K, 640, 480)
            # logging.info(f"depth shape: {depth.shape}, np.max(depth): {np.max(depth)}")
            t2=time.time()
            if SAVE_VIDEO:
                output_video_path = "fp_nodepth_improved.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video
                # Assuming 'color' is the image shape (height, width, channels)
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))
        else:
            t1=time.time()
            
            # last_depth= optical_flow_get_depth(last_rgb.copy(), last_depth, last_mask, color.copy(), prediction)
            last_depth = np.zeros_like(last_mask)
    

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