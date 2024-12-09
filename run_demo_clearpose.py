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
from os import path
from argparse import ArgumentParser
import shutil
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from xmem_wrapper import *
USE_XMEM = True
SAVE_VIDEO= True



def infer_online(args):
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh_file)
    green_color = np.array([0.0, 1.0, 0.0, 1.0])  # RGB + Alpha (fully opaque)
        # Apply green color to all vertices
    mesh.visual.vertex_colors = green_color
    # mesh.show()
    if USE_XMEM:
        network = XMem(config, f'{XMEM_PATH}/saves/XMem.pth').eval().to("cuda")
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

    #object_id = 202, 55
    # object_id = 200
    object_id = 202
    # object_id = 60
    reader = ClosePoseReader(
        video_dir=args.test_scene_dir, object_id=object_id , shorter_side=None, zfar=np.inf)
    

    for i in range(len(reader.color_files)):
        # if i>200:
        #     break
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if USE_XMEM:
            frame_torch, _ = image_to_torch(color, device="cuda")
        green_color =np.array([0,255,0],dtype=np.uint8)
        t1= time.time()
        if i == 0:
            if SAVE_VIDEO:
                # Initialize VideoWriter
                output_video_path = "fp_tracking_unimproved.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video

                # Assuming 'color' is the image shape (height, width, channels)
                height, width, layers = color.shape

                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            
            mask = reader.get_mask(0).astype(bool)
            
            if USE_XMEM:
                # color[mask]= green_color
                mask_png= reader.get_mask(0)
                mask_png[mask_png>250]=1
          
                mask_torch= index_numpy_to_one_hot_torch(mask_png, 2).to("cuda")
                prediction= processor.step(frame_torch, mask_torch[1:])
            pose = est.register(K=reader.K, rgb=color, depth=depth,
                                ob_mask=mask, iteration=args.est_refine_iter)
            # show the mask
            # pose= binary_search_depth(est, mesh, color, mask, reader.K)
            # Plot the mask

            # plt.imshow(reader.get_mask(0), cmap='gray')  # Use 'gray' colormap for better visibility

            # # Add a title
            # plt.title('Mask Visualization')
            # plt.show()
            # plt.close()
            # plt.imshow(color)
            # plt.show()
            # plt.close()


            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(
                    f'{debug_dir}/scene_complete.ply', pcd)
        else:
            if USE_XMEM:
                prediction = processor.step(frame_torch)
                prediction= torch_prob_to_numpy_mask(prediction)
                mask=(prediction==1)
                # color[prediction==1]= green_color
            # pose = est.track_one(rgb=color, depth=depth, 
            #                      K=reader.K, iteration=args.track_refine_iter)
            pose = est.track_one_new(rgb=color, depth=depth, 
                                 K=reader.K,mask=mask, iteration=args.track_refine_iter)
        t2= time.time()
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        color_copy=color.copy()
        # color_copy[mask]=green_color
        # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
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
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/wine_cup_1/wine_cup_1.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/bottle_3/bottle_3.obj')
    parser.add_argument('--test_scene_dir', type=str,
                        default=f'{code_dir}/demo_data/closepose/set8/scene1/')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    infer_online(args)

    