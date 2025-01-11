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
from binary_search_adjust import binary_search_depth
SAVE_VIDEO = False


torch.set_grad_enabled(False)


# default configuration
config = {
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "num_prototypes": 128,
    "min_mid_term_frames": 5,
    "max_mid_term_frames": 10,
    "max_long_term_elements": 10000,
}


torch.set_grad_enabled(False)
torch.cuda.empty_cache()

SAVE_VIDEO = True


import numpy as np
import trimesh


# repo = "isl-org/ZoeDepth"

# # Zoe_N
# model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# zoe = model_zoe_n.to(DEVICE)



import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R



def get_mask(reader, i_frame, ob_id, detect_type):
    if detect_type == "box":
        mask = reader.get_mask(i_frame, ob_id)
        H, W = mask.shape[:2]
        vs, us = np.where(mask > 0)
        umin = us.min()
        umax = us.max()
        vmin = vs.min()
        vmax = vs.max()
        valid = np.zeros((H, W), dtype=bool)
        valid[vmin:vmax, umin:umax] = 1
    elif detect_type == "mask":
        mask = reader.get_mask(i_frame, ob_id, type="mask_visib")
        valid = mask > 0
    elif detect_type == "cnos":  # https://github.com/nv-nguyen/cnos
        mask = cv2.imread(reader.color_files[i_frame].replace("rgb", "mask_cnos"), -1)
        valid = mask == ob_id
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
    ycbv_dir = "/mnt/ssd_990/teng/ycb/FoundationPose/ycbv"
    video_dirs = sorted(glob.glob(f"{ycbv_dir}/test/*"))
    os.environ["YCB_VIDEO_DIR"] = ycbv_dir
    reader_tmp = YcbVideoReader(video_dirs[0])
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    ob_id = 2
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
            reader = YcbVideoReader(video_dir, zfar=1.5)
            scene_ob_ids = reader.get_instance_ids_in_image(0)
            if ob_id in scene_ob_ids:
                break
        except:
            pass
    logging.info(f"video_dir: {video_dir}")

    last_depth = reader.get_depth(0)
    last_rgb = reader.get_color(0)

    for i in range(len(reader.color_files)):

        color = reader.get_color(i)

        if i == 0:
            # mask = reader.get_mask(0).astype(bool)
            mask = get_mask(reader, 0, ob_id, detect_type="box")
            last_mask = mask
            t1 = time.time()
            initial_depth = reader.get_depth(0)

            # depth_numpy= zoe.infer_pil(color)
            # initial_depth= depth_numpy*10
            depth_numpy = None
            pose = binary_search_depth(
                est, mesh, color, mask, reader.K,debug=True
            )
            # pose, scale= binary_search_scale(est,mesh, color, initial_depth, mask, reader.K)
            # mesh.apply_scale(scale)
            est.reset_object(
                model_pts=mesh.vertices.copy(),
                model_normals=mesh.vertex_normals.copy(),
                mesh=mesh,
            )
            est.register(
                K=reader.K,
                rgb=color,
                depth=initial_depth,
                ob_mask=mask,
                iteration=args.est_refine_iter,
            )
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
            t2 = time.time()

            if SAVE_VIDEO:
                output_video_path = (
                    "foundation_pose.mp4"  # Specify the output video filename
                )
                fps = 30  # Frames per second for the video
                # Assuming 'color' is the image shape (height, width, channels)
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for .avi format
                video_writer = cv2.VideoWriter(
                    output_video_path, fourcc, fps, (640, 480)
                )
        else:
            t1 = time.time()
            depth= reader.get_depth(i)
            prediction = last_mask
            # last_depth= optical_flow_get_depth(last_rgb.copy(), last_depth, last_mask, color.copy(), prediction)
            last_depth = np.zeros_like(initial_depth)

            pose = est.track_one(
                rgb=color,
                depth=depth,
                K=reader.K,
                iteration=args.track_refine_iter,
            )

            t2 = time.time()
        os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)
        np.savetxt(f"{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt", pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            color = cv2.putText(
                color,
                f"fps {int(1/(t2-t1))}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
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
