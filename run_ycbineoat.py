# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json, uuid, joblib, os, sys, argparse
from datareader import *
from estimater import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/mycpp/build")
import yaml
import json
import pandas as pd
import numpy as np
from tools import *
from scipy.spatial.distance import cdist
import json


def load_json_data():
    # Path to the JSON file
    json_file_path = "./data/ycbineoat.json"
    # Read and parse the JSON file
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            print("Parsed JSON data:")
            print(data)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return data


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

def run_pose_unkown_scale(choice):
    json_data=load_json_data()
    final_data={}
    # name=["foundation_pose","binary_search_depth"]
    name=["foundation_pose","binary_search"]
    k=0
    for dataset, cad_model in json_data.items():
        test_scene_dir="./data/"+dataset
        print("test_scene_dir:",test_scene_dir)
        cad_model_path="./data/CADmodels/"+cad_model+"/textured_simple.obj"
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        mesh=trimesh.load(cad_model_path)
        vertices = mesh.vertices
        pairwise_distances = cdist(vertices, vertices)  # Use scipy.spatial.distance.cdist
        diameter_exact = np.max(pairwise_distances)
        print("diameter_exact:", diameter_exact)
        original_mesh=mesh.copy()
        mesh.apply_scale(3)
        
        # diameter_exact=diameters[k]
        k+=1
        est=FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug_dir="./debug", debug=0)
        reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
        final_data[dataset]={}
 
        for i in range(len(reader.color_files)):
            color=reader.get_color(i)
            depth=reader.get_depth(i)
            
            if i==0:
                mask=reader.get_mask(i).astype(bool)
                if choice==0:
                    pose=est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
                elif choice==1:
                    pose,scale= binary_search_scale(est, mesh,color, depth, mask, reader.K, scale_min=0.2, scale_max=5,debug=False)

                
            else:
                pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)
            gt_pose=reader.get_gt_pose(i)
            
            tmp=evaluate_pose(gt_pose, pose, original_mesh, diameter_exact,reader.K)
            
            for key in tmp:
                if key not in final_data[dataset]:
                    final_data[dataset][key]=0
                final_data[dataset][key]+=tmp[key]
        for key in final_data[dataset]:
            final_data[dataset][key]/=len(reader.color_files)
    #dump to csv
        csv_file=f"./tmp/{name[choice]}.csv"
        header=["object","ADD", "ADD-S", "rotation_error_deg", "translation_error", "mspd","mssd","recall", "AR_mspd", "AR_mssd","AR_vsd"]
        # Check if the file exists to avoid writing the header multiple times
        # write_header = not os.path.exists(csv_file)

        with open(csv_file, "w") as f:  # Open in append mode
            # if write_header:
            f.write(",".join(header) + "\n")  # Write header only if the file is new

            for obj in final_data:
                f.write(str(obj))
                for key in header[1:]:
                    f.write(f",{final_data[obj][key]}")
                f.write("\n")


def run_pose_estimation_only(choice):
    json_data=load_json_data()
    final_data={}
    # name=["foundation_pose","binary_search_depth"]
    name=["binary_search_depth10","binary_search_depth15"]
    k=0
    for dataset, cad_model in json_data.items():
        test_scene_dir="./data/"+dataset
        print("test_scene_dir:",test_scene_dir)
        cad_model_path="./data/CADmodels/"+cad_model+"/textured_simple.obj"
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        mesh=trimesh.load(cad_model_path)
        vertices = mesh.vertices
        pairwise_distances = cdist(vertices, vertices)  # Use scipy.spatial.distance.cdist
        diameter_exact = np.max(pairwise_distances)
        print("diameter_exact:", diameter_exact)
        # diameter_exact=diameters[k]
        k+=1
        est=FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug_dir="./debug", debug=0)
        reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
        final_data[dataset]={}
 
        for i in range(len(reader.color_files)):
            color=reader.get_color(i)
            depth=reader.get_depth(i)
            mask=reader.get_mask(i).astype(bool)

            # if choice==0:
            #     pose=est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)            
            # elif choice==1:
            #     pose= binary_search_depth(est, mesh,color, mask, reader.K, depth_min=0.3, depth_max=2.0,debug=False)
            if choice==0:
                pose= binary_search_depth(est, mesh,color, mask, reader.K, depth_min=0.4, depth_max=2.0, iteration=10,debug=False)
            elif choice==1:
                pose= binary_search_depth(est, mesh,color, mask, reader.K, depth_min=0.4, depth_max=2.0, iteration=15,debug=False)
            gt_pose=reader.get_gt_pose(i)
            
            tmp=evaluate_pose(gt_pose, pose, mesh, diameter_exact,reader.K)
            
            for key in tmp:
                if key not in final_data[dataset]:
                    final_data[dataset][key]=0
                final_data[dataset][key]+=tmp[key]
        for key in final_data[dataset]:
            final_data[dataset][key]/=len(reader.color_files)
    #dump to csv
        csv_file=f"{name[choice]}.csv"
        header=["object","ADD", "ADD-S", "rotation_error_deg", "translation_error", "mspd","mssd","recall", "AR_mspd", "AR_mssd","AR_vsd"]
        # Check if the file exists to avoid writing the header multiple times
        # write_header = not os.path.exists(csv_file)

        with open(csv_file, "w") as f:  # Open in append mode
            # if write_header:
            f.write(",".join(header) + "\n")  # Write header only if the file is new

            for obj in final_data:
                f.write(str(obj))
                for key in header[1:]:
                    f.write(f",{final_data[obj][key]}")
                f.write("\n")
        



def run_pose_estimation(choice):
    json_data=load_json_data()
    final_data={}
    name=["foundation_pose", "foundation_pose_with_zero_depth", "foundation_pose_with_kf", "foundation_pose_last_depth", "foundation_pose_with_binary_search","foundation_pose_gt"]
    k=0
    for dataset, cad_model in json_data.items():
        test_scene_dir="./data/"+dataset
        print("test_scene_dir:",test_scene_dir)
        cad_model_path="./data/CADmodels/"+cad_model+"/textured_simple.obj"
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        mesh=trimesh.load(cad_model_path)
        vertices = mesh.vertices
        pairwise_distances = cdist(vertices, vertices)  # Use scipy.spatial.distance.cdist
        diameter_exact = np.max(pairwise_distances)
        print("diameter_exact:", diameter_exact)
        # diameter_exact=diameters[k]
        k+=1
        est=FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx, debug_dir="./debug", debug=0)
        reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
        final_data[dataset]={}
        if choice==2:
            kf=PoseTracker()
        for i in range(len(reader.color_files)):
            color=reader.get_color(i)
            depth=reader.get_depth(i)
            if i==0:
                mask=reader.get_mask(0).astype(bool)
                if choice in [5,1,2,3]:
                    # pose=est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
                    pose=reader.get_gt_pose(0)
                    est.pose_last=torch.from_numpy(pose).cuda()
                if choice==2:
                    xyz=pose[:3,3].reshape(3,1)
                    r=R.from_matrix(pose[:3,:3])
                    angles=r.as_euler("xyz").reshape(3, 1)
                    kf.initialize(xyz, angles)
                    measurement = np.concatenate((xyz, angles)).reshape(6, 1)
                    for k in range(10):
                        kf.update(measurement)
                if choice==0:
                    pose=est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
                if choice==4:
                    pose= binary_search_depth(est, mesh,color, mask, reader.K, depth_max=3,debug=False)
            else:
                if choice==1 or choice==4:
                    last_depth=np.zeros_like(depth)
                elif choice==3:
                    last_depth=render_cad_depth(pose, est.mesh, reader.K, 640, 480)
                elif choice==2:
                    predict=kf.predict_next_pose()
                    xyz_predict = np.squeeze(predict["position"])
                    orientation= predict["orientation"]
                    predict_pose = np.eye(4)
                    predict_pose[:3, 3] = xyz_predict
                    matrix = R.from_euler("xyz", np.squeeze(orientation)).as_matrix()
                    predict_pose[:3, :3] = matrix
                    last_depth=render_cad_depth(pose, mesh, reader.K, 640, 480)
                elif choice==0 or choice==5:
                    last_depth=depth
                pose = est.track_one(rgb=color, depth=last_depth, K=reader.K, iteration=2)
                if choice==2:
                    xyz=pose[:3,3].reshape(3,1)
                    r=R.from_matrix(pose[:3,:3])
                    angles=r.as_euler("xyz").reshape(3, 1)
                    kf.update(np.concatenate((xyz, angles)).reshape(6, 1))
            gt_pose=reader.get_gt_pose(i)
            
            tmp=evaluate_pose(gt_pose, pose, mesh, diameter_exact,reader.K)
            
            for key in tmp:
                if key not in final_data[dataset]:
                    final_data[dataset][key]=0
                final_data[dataset][key]+=tmp[key]
        for key in final_data[dataset]:
            final_data[dataset][key]/=len(reader.color_files)
    #dump to csv
        header=["object","ADD", "ADD-S", "rotation_error_deg", "translation_error", "mspd","mssd","recall", "AR_mspd", "AR_mssd", "AR_vsd"]
        with open(f"{name[choice]}.csv", "w") as f:
            f.write(",".join(header)+"\n")
            for obj in final_data:
                f.write(str(obj))
                for key in header[1:]:
                    f.write(f",{final_data[obj][key]}")
                f.write("\n")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--ycbv_dir",
        type=str,
        default="/mnt/ssd_990/teng/ycb/FoundationPose/ycbv",
        help="data dir",
    )
    parser.add_argument("--use_reconstructed_mesh", type=int, default=0)
    parser.add_argument(
        "--ref_view_dir",
        type=str,
        default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16",
    )
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    opt = parser.parse_args()
    os.environ["YCB_VIDEO_DIR"] = opt.ycbv_dir
    logging.disable(logging.INFO)
    set_seed(0)

    detect_type = "mask"  # mask / box / detected
    # choice=0
    # run_pose_estimation(choice)
    # choice=1
    # run_pose_estimation(choice)
    # choice=2
    # run_pose_estimation(choice)
    # choice=3
    # run_pose_estimation(choice)
    # choice=4
    # run_pose_estimation(choice)
    # choice=5
    # run_pose_estimation(choice)

    # choice=0
    # # run_pose_estimation_only(choice)
    # run_pose_unkown_scale(choice)
    choice=1
    # run_pose_estimation_only(choice)
    run_pose_unkown_scale(choice)

    choice=0
    # run_pose_estimation_only(choice)
    run_pose_unkown_scale(choice)
