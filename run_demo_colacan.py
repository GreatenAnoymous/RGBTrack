# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from tools import *
SAVE_VIDEO=False
repo = "isl-org/ZoeDepth"

model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Configure logging
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, 
                    default=f"{code_dir}/demo_data/cola_can/mesh/32429d6d7cd54f8392f9b3056a1f26c3.obj"
    )
    parser.add_argument(
        "--test_scene_dir", type=str, default=f"{code_dir}/demo_data/cola_can"
    )
    parser.add_argument('--scale_recovery', type=bool, default=True)
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=3)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    use_scale_recovery = args.scale_recovery
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        if i==0:
            depth= zoe.infer_pil(color)
            mask = reader.get_mask(0).astype(bool)
            if use_scale_recovery:
                pose, scale=binary_search_scale(est, mesh, color, depth, mask, reader.K, w=544, h=960, debug=False)
                mesh.apply_scale(scale)
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            else:
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            if SAVE_VIDEO:
                output_video_path = "colacan_no_recovery.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video
                # Assuming 'color' is the image shape (height, width, channels)
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (544, 960))
        else:
            last_depth=np.zeros_like(depth)
            pose = est.track_one(rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter)
        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)
        if SAVE_VIDEO:
            video_writer.write(vis[..., ::-1])

