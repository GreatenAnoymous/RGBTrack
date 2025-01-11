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


def run_pose_estimation_worker(
    reader, i_frames, est: FoundationPose, debug=False, ob_id=None, device: int = 0
):

    torch.cuda.set_device(device)
    est.to_device(f"cuda:{device}")
    est.glctx = dr.RasterizeCudaContext(device)
    debug_dir = est.debug_dir
    model_points= est.mesh.vertices
    # Compute the pairwise distances between all vertices
    diameter = reader.get_model_diameter(ob_id)
    # print(i_frames, "debug")
    data={"ADD":0, "ADD-S":0, "ADD_AUC":0, "ADD-S_AUC":0, "rotation_error_deg":0, "translation_error":0, "recall":0}

    for i in range(len(i_frames)):
        i_frame = i_frames[i]
        id_str = reader.id_strs[i_frame]

        color = reader.get_color(i_frame)
        depth = reader.get_depth(i_frame)

        H, W = color.shape[:2]
        scene_ob_ids = reader.get_instance_ids_in_image(i_frame)
        video_id = reader.get_video_id()

        logging.info(f"video:{reader.get_video_id()}, id_str:{id_str}, ob_id:{ob_id}")
        if ob_id not in scene_ob_ids:
            logging.info(f"skip {ob_id} as it does not exist in this scene")
            continue
        ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)

        est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
        pose= binary_search_depth(est, est.mesh, color, ob_mask, reader.K,debug=True)
        
        # pose = est.register(
        #     K=reader.K,
        #     rgb=color,
        #     depth=depth,
        #     ob_mask=ob_mask,
        #     ob_id=ob_id,
        #     iteration=5,
        # )
        tmp_data=evaluate_pose(est.gt_pose, pose, model_points, diameter)
    
        for key in data:
            data[key]+=tmp_data[key]
    
        for key in data:
            data[key]/=len(i_frames)
        # logging.info(f"pose:\n{pose}")


    
    return data




def run_pose_estimation():
    wp.force_load(device="cuda")
    video_dirs = sorted(glob.glob(f"{opt.ycbv_dir}/test/*"))
    debug = opt.debug
    use_reconstructed_mesh = opt.use_reconstructed_mesh
    debug_dir = opt.debug_dir

    reader_tmp = YcbVideoReader(video_dirs[0])
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
    est = FoundationPose(
        model_pts=mesh_tmp.vertices.copy(),
        model_normals=mesh_tmp.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh_tmp,
        scorer=None,
        refiner=None,
        glctx=glctx,
        debug_dir=debug_dir,
        debug=debug,
    )

    ob_ids = reader_tmp.ob_ids
    final_data={}

    for ob_id in ob_ids:

        
        print(f"Processing object {ob_id}")
        final_data[ob_id]={"ADD":0, "ADD-S":0, "ADD_AUC":0, "ADD-S_AUC":0, "rotation_error_deg":0, "translation_error":0, "recall":0}
        if use_reconstructed_mesh:
            mesh = reader_tmp.get_reconstructed_mesh(
                ob_id, ref_view_dir=opt.ref_view_dir
            )
        else:
            mesh = reader_tmp.get_gt_mesh(ob_id)
        
        symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]

        est.reset_object(
            model_pts=mesh.vertices.copy(),
            model_normals=mesh.vertex_normals.copy(),
            symmetry_tfs=symmetry_tfs,
            mesh=mesh,
        )

        # args = []
        final_data[ob_id]={"ADD":0, "ADD-S":0, "ADD_AUC":0, "ADD-S_AUC":0, "rotation_error_deg":0, "translation_error":0, "recall":0}
        count=0
        zero_depth=np.zeros_like(reader_tmp.get_depth(0))
        
        for video_dir in video_dirs:
            reader = YcbVideoReader(video_dir, zfar=1.5)
            scene_ob_ids = reader.get_instance_ids_in_image(0)
            if ob_id not in scene_ob_ids:
                continue
            video_id = reader.get_video_id()

            for i in range(len(reader.color_files)):
                if i==0:
                    ob_mask = get_mask(reader, i, ob_id, detect_type=detect_type)
                    color=reader.get_color(i)
                    depth=reader.get_depth(i)
                    pose=est.register(reader.K, color,depth, ob_mask, iteration=5)
                    # print(reader.K, ob_mask.shape, color.shape, depth.shape)
                    # pose=binary_search_depth(est, mesh,color , ob_mask, reader.K,debug=True)
                else:
                    depth=reader.get_depth(i)
                    
                    pose=est.track_one(
                        reader.get_color(i),
                        depth,
                        reader.K,
                        iteration=3,
                    )
                gt_pose=reader.get_gt_pose(i, ob_id)
                tmp_data=evaluate_pose(gt_pose, pose, mesh.vertices, reader.get_model_diameter(ob_id))
                for key in final_data[ob_id]:
                    final_data[ob_id][key]+=tmp_data[key]
                count+=1
            
                # if not reader.is_keyframe(i):
                #     continue
                # args.append((reader, [i], est, debug, ob_id, 0))
            for key in final_data[ob_id]:
                final_data[ob_id][key]/=count
        
        # print(len(args))
        # count=0
        # for arg in args:
        #     tmp_data= run_pose_estimation_worker(*arg)
        #     count+=1
        #     for key in final_data[ob_id]:
        #         final_data[ob_id][key]+=tmp_data[key]
        #     for key in final_data[ob_id]:
        #         final_data[ob_id][key]/=count
        # final_data["mean"]={"ADD":0, "ADD-S":0, "ADD_AUC":0, "ADD-S_AUC":0, "rotation_error_deg":0, "translation_error":0, "recall":0}
        # for obj in final_data:
        #     if obj=="mean":
        #         continue
        #     for sub_key in final_data[obj]:
        #         final_data["mean"][sub_key]+=final_data[obj][sub_key]
        # for key in final_data["mean"]:
        #     final_data["mean"][key]/=len(final_data)-1

    #dump to csv
        header=["object","ADD", "ADD-S", "ADD_AUC", "ADD-S_AUC", "rotation_error_deg", "translation_error", "recall"]
        with open("data.csv", "w") as f:
            f.write(",".join(header)+"\n")
            for obj in final_data:
                f.write(str(obj))
                for key in header[1:]:
                    f.write(f",{final_data[obj][key]}")
                f.write("\n")
        

        # for out in outs:
        #     for video_id in out:
        #         for id_str in out[video_id]:
        #             if video_id not in res:
        #                 res[video_id] = {}
        #             if id_str not in res[video_id]:
        #                 res[video_id][id_str] = {}
        #             # print(out[video_id][id_str][ob_id])
        #             res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id].tolist()



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

    run_pose_estimation()
    # evaluate_ycb()
