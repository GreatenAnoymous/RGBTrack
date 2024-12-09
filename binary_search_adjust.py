import numpy as np
import matplotlib.pyplot as plt
import logging
from tools import render_rgbd
import cv2

def binary_search_depth( est,mesh, rgb, mask, K, depth_min=0.2, depth_max=20,w=640, h=480, debug=False):
    low=depth_min
    high=depth_max
    while low <= high:
        mid= (low+high)/2
        depth= np.ones_like(mask)*mid
        pose= est.register(K, rgb, depth, mask, 5)
        rgb_r, depth_r, mask_r= render_rgbd(mesh, pose, K, w, h)
        if debug:
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_r)
            plt.subplot(1, 2, 2)
            rgb_copy= rgb.copy()
            rgb_copy[mask==0]=0
            plt.imshow(rgb_copy)
            plt.savefig(f"tmp/debug_{mid}.png")
        if abs(high-low)<0.001:
            return pose
        
        if  abs(np.sum(mask_r)-np.sum(mask))<10:
            logging.info(f"mid: {mid}")
            a=input("Press enter to continue")
            return pose
        if np.sum(mask_r)>np.sum(mask):
            low=mid
        elif np.sum(mask_r)<np.sum(mask):
            high=mid
            
            

def binary_search_scale(est,mesh, rgb,depth, mask, K, scale_min=0.2, scale_max=5,w=640, h=480, debug=False):
    low=scale_min
    high=scale_max
    while low<=high:
        mid= (low+high)/2
        mesh_c=mesh.copy()
        mesh_c.apply_scale(mid)
        est.reset_object(model_pts=mesh_c.vertices.copy(), model_normals=mesh_c.vertex_normals.copy(), mesh=mesh_c)
        pose= est.register(K, rgb,depth, mask, 5)
        rgb_r, depth_r, mask_r= render_rgbd(mesh_c, pose, K, 640, 480)
        binary_mask = (mask_r > 0).astype(np.uint8)
    
        # Calculate the bounding box
        x, y, width, height = cv2.boundingRect(binary_mask)
        
        # Calculate the area of the bounding box
        area = width * height
        if debug:
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_r)
            plt.subplot(1, 2, 2)
            rgb_copy= rgb.copy()
            rgb_copy[mask==0]=0
            plt.imshow(rgb_copy)
            plt.savefig(f"tmp/debug_{mid}.png")
        if abs(high-low)<0.01:
            break
        if  abs(area-np.sum(mask))<20:
            break
        if area>np.sum(mask):
            high=mid
        elif area<np.sum(mask):
            low=mid
    return pose, mid

