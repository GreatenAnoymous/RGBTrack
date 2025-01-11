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
from binary_search_adjust import *
def generate_checkerboard_texture(width, height, block_size=40, color1=(255, 255, 255), color2=(0, 0, 0)):
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if (x // block_size + y // block_size) % 2 == 0:
                texture[y:y + block_size, x:x + block_size] = color1
            else:
                texture[y:y + block_size, x:x + block_size] = color2

    texture_image = Image.fromarray(texture)
    return texture_image

# Step 1: Create a texture
def generate_gradient_texture(width, height, color1=(0, 255, 255), color2=(0, 0, 0)):
    texture = np.zeros((height, width, 3), dtype=np.uint8)

    # Create horizontal gradient from color1 to color2
    for x in range(width):
        r = int(color1[0] + (color2[0] - color1[0]) * x / width)
        g = int(color1[1] + (color2[1] - color1[1]) * x / width)
        b = int(color1[2] + (color2[2] - color1[2]) * x / width)
        texture[:, x] = [r, g, b]
    
    texture_image = Image.fromarray(texture)
    return texture_image
# Step 2: Apply texture to OBJ file
def apply_texture_to_obj(mesh, texture_image):
    # Convert texture_image to numpy array if it's a PIL Image
    if isinstance(texture_image, Image.Image):
        texture_array = texture_image
    else:
        texture_array = texture_image
    
    # Generate UV coordinates
    vertices = mesh.vertices
    u = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
    v = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
    uv = np.column_stack((u, v))
    
    # Create material
    material = trimesh.visual.material.SimpleMaterial(
        image=texture_array,
        diffuse=[255, 255, 255, 255]
    )
    
    # Create TextureVisuals
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=material
    )
    return mesh




# Step 3: Overlay texture on the RGB image
def overlay_texture_on_image(image, texture, mask=None):
    """
    Overlay a texture onto an image with 100% opacity in the masked regions
    
    Parameters:
    image (numpy.ndarray): Base image to overlay texture onto
    texture (numpy.ndarray or PIL.Image): Texture to overlay
    mask (numpy.ndarray, optional): Binary mask where texture should be applied
    
    Returns:
    numpy.ndarray: Image with texture overlay
    """
    # Convert texture to numpy array if it's a PIL Image
    if isinstance(texture, Image.Image):
        texture = np.array(texture)
    
    # Create a copy of the image to avoid modifying the original
    result = image.copy()
    
    if mask is not None:
        # Resize the texture to match the mask's size
        texture = cv2.resize(texture, (mask.shape[1], mask.shape[0]))
        # Apply texture directly to masked regions (100% opacity)
        result[mask > 0] = texture[mask > 0]
    else:
        # If no mask is provided, overlay the texture onto the full image
        texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
        result = texture.copy()  # Full opacity means using texture directly
    
    return result

def infer_online(args):
    set_logging_format()
    set_seed(0)
    texture= generate_gradient_texture(512, 512)
    plt.imshow(texture)
    plt.show()
    mesh = trimesh.load(args.mesh_file)
    # mesh= apply_texture_to_obj(mesh, texture)
    
    # mesh.show()
    green_color = np.array([0.0, 1.0, 0.0, 1.0])  # RGB + Alpha (fully opaque)
        # Apply green color to all vertices
    # mesh.visual.vertex_colors = green_color
    apply_texture_to_obj(mesh, texture)
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
    # object_id = 202
    object_id =  55
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
        if True:
            if SAVE_VIDEO:
                # Initialize VideoWriter
                output_video_path = "fp_tracking_transparent.mp4"  # Specify the output video filename
                fps = 30  # Frames per second for the video

                # Assuming 'color' is the image shape (height, width, channels)
                height, width, layers = color.shape

                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            
            mask = reader.get_mask(i).astype(bool)
            
            if USE_XMEM:
                color_copy=color.copy()
                color=overlay_texture_on_image(color, texture, mask)
                mask_png= reader.get_mask(i)
                mask_png[mask_png>250]=1
        
                mask_torch= index_numpy_to_one_hot_torch(mask_png, 2).to("cuda")
                # prediction= processor.step(frame_torch, mask_torch[1:])
            # pose = est.register(K=reader.K, rgb=color, depth=depth,
            #                     ob_mask=mask, iteration=args.est_refine_iter)
            # show the mask
            pose= binary_search_depth(est, mesh, color, mask, reader.K)
            # Plot the mask



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
                # prediction = processor.step(frame_torch)
                # prediction= torch_prob_to_numpy_mask(prediction)
                # mask=(prediction==1)
                mask = reader.get_mask(i).astype(bool)
                color_copy=color.copy()
                # color=overlay_texture_on_image(color, texture, mask)
            
            pose = est.track_one(rgb=color, depth=depth, 
                                 K=reader.K, iteration=args.track_refine_iter)
            # pose = est.track_one_new(rgb=color, depth=depth, 
            #                      K=reader.K,mask=mask, iteration=args.track_refine_iter)
        t2= time.time()
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        
        # color_copy[mask]=green_color
        # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
        if SAVE_VIDEO:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(
                reader.K, img=color_copy, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_copy, ob_in_cam=center_pose, scale=0.1,
                                K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            video_writer.write(vis[..., ::-1])


        if debug >= 1:
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
    parser.add_argument('--tex_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/wine_cup_1/wine_cup_1_texture.mtl')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/wine_cup_1/wine_cup_1.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/closepose/closepose_model/bottle_3/bottle_3.obj')
    parser.add_argument('--test_scene_dir', type=str,
                        default=f'{code_dir}/demo_data/closepose/set8/scene1/')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=3)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    infer_online(args)

    