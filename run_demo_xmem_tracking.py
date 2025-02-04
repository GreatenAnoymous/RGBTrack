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
from tools import get_3d_points, evaluate_metrics
from xmem_wrapper import *

from binary_search_adjust import *
import numpy as np

from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        # default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
        default=f"{code_dir}/data/CADmodels/021_bleach_cleanser/textured_simple.obj",
        # default=f"{code_dir}/data/CADmodels/003_cracker_box/textured_simple.obj",
        # default=f"{code_dir}/data/CADmodels/004_sugar_box/textured_simple.obj"
        # default=f"{code_dir}/data/CADmodels/005_tomato_soup_can/textured_simple.obj"
        
    )
    parser.add_argument(
        # "--test_scene_dir", type=str, default=f"{code_dir}/data/mustard0"
        "--test_scene_dir", type=str, default=f"{code_dir}/data/bleach0"
        # "--test_scene_dir", type=str, default=f"{code_dir}/data/mustard_easy_00_02"
        # "--test_scene_dir", type=str, default=f"{code_dir}/data/cracker_box_reorient"
        # "--test_scene_dir", type=str, default=f"{code_dir}/data/tomato_soup_can_yalehand0"
        # "--test_scene_dir", type=str, default=f"{code_dir}/data/sugar_box_yalehand0"

    )
    parser.add_argument("--use_xmem", type=bool, default=False)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()
    USE_XMEM = args.use_xmem
    SAVE_VIDEO = args.save_video
    print(f"USE_XMEM: {USE_XMEM}")
    print(f"SAVE_VIDEO: {SAVE_VIDEO}")
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


    reader = YcbineoatReader(
        video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf
    )
    
    
    if USE_XMEM:
        network = XMem(config, f'{XMEM_PATH}/saves/XMem.pth').eval().to("cuda")
        processor=InferenceCore(network, config=config)
        processor.set_all_labels(range(1,2))
        #You can change these values to get different results
        frames_to_propagate = 1000

    kf=PoseTracker()
    history_poses=[]
    for i in range(len(reader.color_files)):
        logging.info(f"i:{i}")
        color = reader.get_color(i)
        depth= reader.get_depth(i)
        if USE_XMEM:
            frame_torch, _ = image_to_torch(color, device="cuda")
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            last_mask= mask
            t1=time.time()
            initial_depth= reader.get_depth(0)
            # pose= binary_search_depth(est,mesh, color, mask, reader.K, depth_max=2)
    
            # pose = est.register(
            #     K=reader.K,
            #     rgb=color,
            #     depth=initial_depth,
            #     ob_mask=mask,
            #     iteration=args.est_refine_iter,
            # )
            pose= reader.get_gt_pose(0)
            est.pose_last=torch.from_numpy(pose).float().cuda()
            xyz=pose[:3,3].reshape(3,1)
            r=R.from_matrix(pose[:3,:3])
            angles=r.as_euler("xyz").reshape(3, 1)
            kf.initialize(xyz, angles)
            measurement = np.concatenate((xyz, angles)).reshape(6, 1)
            
            for k in range(10):
                kf.update(measurement)
                # pose=kf.predict_next_pose()
            
        
            logging.info(f"Initial pose:\n{pose}")
        
            t2=time.time()
            
            if USE_XMEM:
                mask_png= reader.get_mask(0)
                mask_torch= index_numpy_to_one_hot_torch(mask_png, 2).to("cuda")
                prediction= processor.step(frame_torch, mask_torch[1:])
                
            if SAVE_VIDEO:
                output_video_path = "kalman_filter.mp4"  # Specify the output video filename
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
                mask=(prediction==1)
            # prediction= last_mask
        
            # last_depth=render_cad_depth(pose, mesh, reader.K, 640, 480)
            # predict=kf.predict_next_pose()
            # xyz_predict = np.squeeze(predict["position"])
            # orientation= predict["orientation"]
            # predict_pose = np.eye(4)
            # predict_pose[:3, 3] = xyz_predict
            # matrix = R.from_euler("xyz", np.squeeze(orientation)).as_matrix()
            # predict_pose[:3, :3] = matrix


            last_depth = np.zeros_like(last_mask)

            pose = est.track_one(
                rgb=color, depth=last_depth, K=reader.K, iteration=args.track_refine_iter
            )
            # pose=est.track_one_new_without_depth(rgb=color,K=reader.K,mask=mask, iteration=args.track_refine_iter)
            # xyz=pose[:3,3]
            # xyz=xyz.reshape(3,1)
            # r=R.from_matrix(pose[:3,:3])
            # angles=r.as_euler("xyz").reshape(3, 1)
            # measurement = np.concatenate((xyz, angles)).reshape(6, 1)
            
            # kf.update(measurement)
            # kf_pose=kf.get_current_pose()
            # est.pose_last=torch.from_numpy(kf_pose).float().cuda()
            # pose=est.track_one_new_without_depth(rgb=color,
            #                     K=reader.K,mask=prediction, iteration=args.track_refine_iter)
            t2=time.time()
        history_poses.append(pose)


        color_copy=color.copy()
        # color_copy[mask]=np.array([0,255,0])
        if debug >= 1:
            color_copy=cv2.putText(color_copy, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            center_pose = pose @ np.linalg.inv(to_origin)
            # color_copy=cv2.putText(color_copy, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
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
    data=evaluate_metrics(history_poses, reader, mesh)
    header=["object","ADD", "ADD-S", "rotation_error_deg", "translation_error", "recall"]
    with open(f"tmp.csv", "w") as f:
        f.write(",".join(header)+"\n")
        for key in header[1:]:
            f.write(f",{data[key]}")
        f.write("\n")

