
from estimater import *
from datareader import *
import argparse
import os
from binary_search_adjust import *
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
SAVE_VIDEO= False
from torch.cuda.amp import autocast


code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/mycpp/build")
import yaml
import json
import pandas as pd
import numpy as np
from tools import *
from scipy.spatial.distance import cdist
import json

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



def run_estimation(args, choice):
    global USE_XMEM
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh_file)
    if USE_XMEM:
        network = XMem(config, f'{XMEM_PATH}/saves/XMem.pth').eval().to("cuda")
        network=network
        # for name, param in network.named_parameters():
        #     print(f"Parameter: {name}, Data type: {param.dtype}")
        processor=InferenceCore(network, config=config)
        processor.set_all_labels(range(1,2))
        #You can change these values to get different results
        frames_to_propagate = 2000

    names=["foundation_pose", "tracking", "tracking_no_depth"]

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, debug_dir="./debug", debug=0, glctx=glctx)
    logging.info("estimator initialization done")

    #object_id = 202, 55
    object_id = 202
    # object_id = 201
    # object_id = 60
    reader = ClosePoseReader(
        video_dir=args.test_scene_dir, object_id=object_id , shorter_side=None, zfar=np.inf)
    reader.color_files=sorted(reader.color_files)
    start=0
    history_poses=[]
    for i in range(start,len(reader.color_files)):
    
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if USE_XMEM:
            frame_torch, _ = image_to_torch(color, device="cuda")
            frame_torch = frame_torch

        green_color =np.array([0,255,0],dtype=np.uint8)
        t1= time.time()
        if i == start:
            if SAVE_VIDEO:
                # Initialize VideoWriter
                output_video_path = "fp_tracking_improved.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video

                # Assuming 'color' is the image shape (height, width, channels)
                height, width, layers = color.shape

                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            
            mask = reader.get_mask(start).astype(bool)
            
            if USE_XMEM:
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
            # show the mask
            
            # pose= binary_search_depth(est, mesh, color, mask, reader.K,depth_min=0.2, depth_max=2)

            # pose= est.register_without_depth(K=reader.K, rgb=color, ob_mask=mask, iteration=args.est_refine_iter,low=0.2,high=3)
            t1=time.time()
            logging.info(f"runnning time: {t1-t0}") 
            logging.info(f"Initial pose:\n{pose}")
        else:
            if USE_XMEM:
                # frame_torch= F.interpolate(frame_torch.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
                with autocast():

                    prediction = processor.step(frame_torch)
                # prediction= F.interpolate(prediction.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)
                prediction= torch_prob_to_numpy_mask(prediction)
                mask=(prediction==1)
            # pose = est.track_one_new_without_depth(rgb=color,
            #                     K=reader.K,mask=mask, iteration=args.track_refine_iter)
            if choice==0:
                pose = est.track_one(rgb=color, depth=depth, 
                                    K=reader.K, iteration=args.track_refine_iter)
            elif choice==1:
                pose = est.track_one_new(rgb=color, depth=depth, 
                                    K=reader.K,mask=mask, iteration=args.track_refine_iter)
            else:
                pose = est.track_one_new_without_depth(rgb=color,
                            K=reader.K,mask=mask, iteration=args.track_refine_iter)
            
            history_poses.append(pose)
            pose_predict=est.tracker.predict_next_pose()
            r=R.from_matrix(pose[:3,:3])
            angles=r.as_euler("xyz").reshape(3, 1)

        t2= time.time()
        center_pose = pose@np.linalg.inv(to_origin)
        color_copy=color.copy()
        # color_copy=cv2.putText(color_copy, f"fps {int(1/(t2-t1))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        vis = draw_posed_3d_box(
            reader.K, img=color_copy, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color_copy, ob_in_cam=center_pose, scale=0.1,
                            K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[..., ::-1])
        cv2.waitKey(1) 
        # color_copy[mask]=green_color
        # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
    data,data2=evaluate_metrics(history_poses, reader, mesh, traj=True)
    header=["object","ADD", "ADD-S", "rotation_error_deg", "translation_error", "mspd","mssd","recall", "AR_mspd", "AR_mssd","AR_vsd"]
    with open(f"{names[choice]}.csv", "w") as f:
        f.write(",".join(header)+"\n")
        for key in header[1:]:
            f.write(f",{data[key]}")
        f.write("\n")
    with open(f"{names[choice]}_traj.csv", "w") as f:
        f.write(",".join(header)+"\n")
        for i in range(len(data2["ADD"])):
            f.write(f",{data2['ADD'][i]},{data2['ADD-S'][i]},{data2['rotation_error_deg'][i]},{data2['translation_error'][i]},{data2['mspd'][i]},{data2['mssd'][i]},{data2['recall'][i]},{data2['AR_mspd'][i]},{data2['AR_mssd'][i]},{data2['AR_vsd'][i]}\n")
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/wine_cup_1/wine_cup_1.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/005_tomato_soup_can/005_tomato_soup_can.obj')
    parser.add_argument('--test_scene_dir', type=str,
                        default=f'{code_dir}/demo_data/closepose/set8/scene1/')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--use_xmem', type=str, default=True)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()
    
 
    USE_XMEM = args.use_xmem
    # run_estimation(args, choice=0)
    # run_estimation(args, choice=1)
    run_estimation(args, choice=2)
