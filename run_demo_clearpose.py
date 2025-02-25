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
import os
from tools import *
import cv2
import numpy as np
from xmem_wrapper import *
USE_RECOVER = True
SAVE_VIDEO= False
from torch.cuda.amp import autocast


def infer_online(args):
    global USE_RECOVER
    if args.recover:
        USE_RECOVER = True
    else:
        USE_RECOVER = False
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh_file)
    green_color = np.array([0.0, 1.0, 0.0, 1.0])  # RGB + Alpha (fully opaque)
        # Apply green color to all vertices
    # mesh.visual.vertex_colors = green_color
    # mesh.show()
    if USE_RECOVER:
        # network = XMem(config, f'{XMEM_PATH}/saves/XMem-with-mose.pth').eval().to("cuda")
        network = XMem(config, f'{XMEM_PATH}/saves/XMem.pth').eval().to("cuda")
        network=network
        processor=InferenceCore(network, config=config)
        processor.set_all_labels(range(1,2))
        #You can change these values to get different results
        frames_to_propagate = 2000

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(
        f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

   
    object_id = 202
 
    reader = ClosePoseReader(
        video_dir=args.test_scene_dir, object_id=object_id , shorter_side=None, zfar=np.inf)
    reader.color_files=sorted(reader.color_files)
    start=00
    for i in range(start,len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if USE_RECOVER:
            frame_torch, _ = image_to_torch(color, device="cuda")
            frame_torch = frame_torch

        green_color =np.array([0,255,0],dtype=np.uint8)
        t1= time.time()
        if i == start:
            if SAVE_VIDEO:
                # Initialize VideoWriter
                output_video_path = "tracking_no_depth.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video

                # Assuming 'color' is the image shape (height, width, channels)
                height, width, layers = color.shape

                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            
            mask = reader.get_mask(start).astype(bool)
            
            if USE_RECOVER:
                # color[mask]= green_color
                mask_png= reader.get_mask(start)
                mask_png[mask_png>250]=1

                mask_torch= index_numpy_to_one_hot_torch(mask_png, 2).to("cuda")
                mask_torch= mask_torch
                # resize mask to half size
                # mask_torch= F.interpolate(mask_torch.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
                # frame_torch= F.interpolate(frame_torch.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
                with autocast():
                    prediction= processor.step(frame_torch, mask_torch[1:])
            t0= time.time()
            pose = est.register(K=reader.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=args.est_refine_iter)
            reader.initial_pose=pose
            # pose= binary_search_depth(est, mesh, color, mask, reader.K,depth_min=0.2, depth_max=2)

            t1=time.time()
            logging.info(f"runnning time: {t1-t0}") 
        else:
            if USE_RECOVER:
                # frame_torch= F.interpolate(frame_torch.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
                with autocast():
                    prediction = processor.step(frame_torch)
                # prediction= F.interpolate(prediction.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
                prediction= torch_prob_to_numpy_mask(prediction)
                mask=(prediction==1)
                if args.mode==0:
                    pose = est.track_one_new_without_depth(rgb=color,
                                        K=reader.K,mask=mask, iteration=args.track_refine_iter)
                elif args.mode==1:
                    pose = est.track_one_new(rgb=color, depth=depth, 
                                K=reader.K,mask=mask, iteration=args.track_refine_iter)
            else:
                pose= est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
            #get the euler angles from the pose
            r=R.from_matrix(pose[:3,:3])
            angles=r.as_euler("xyz").reshape(3, 1)
            print(f"angles: {np.squeeze(angles)}")
            # print(f"pose: {pose[:3,3]}")
            print("=====================================")
        t2= time.time()
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        color_copy=color.copy()
        # color_copy[mask]=green_color
        if SAVE_VIDEO:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(
                reader.K, img=color_copy, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_copy, ob_in_cam=center_pose, scale=0.1,
                                K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            video_writer.write(vis[..., ::-1])
        elif debug >= 1:
            center_pose = pose@np.linalg.inv(to_origin)
            # color_copy=cv2.putText(color_copy, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            vis = draw_posed_3d_box(
                reader.K, img=color_copy, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_copy, ob_in_cam=center_pose, scale=0.1,
                                K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[..., ::-1])
            cv2.waitKey(1) 

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(
                f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    if SAVE_VIDEO:
        video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/006_mustard_bottle/006_mustard_bottle.obj')
    parser.add_argument('--test_scene_dir', type=str,
                        default=f'{code_dir}/demo_data/closepose/set8/scene1/')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--mode', type=int, default=0, help='0: no depth, 1:  depth')
    parser.add_argument("--recover", type=bool, default=True, help='whether to use recovery mechanism')
    args = parser.parse_args()

    infer_online(args)

    