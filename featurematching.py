
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from tools import get_3d_points
sys.path.append('/mnt/ssd_990/teng/SuperGluePretrainedNetwork/')

from models.matching import Matching
# from nets.patchnet import Quad_L2Net 
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
torch.set_grad_enabled(False)
def load_super_glue():
    # Load SuperGlue
    superglue = Matching()
    superglue.eval()
    superglue = superglue.cuda()
    return superglue


class FeatureMathcerPoseEstimator:
    def __init__(self, to_origin=None, bbox=None, K=None) -> None:
        self.model= load_super_glue()
        self.to_origin=to_origin
        self.bbox = bbox
        self.w=640
        self.K=K
        self.h=480

    def estimate_pose(self,points_3d, points_2d, camera_matrix, dist_coeffs=None):
        # Use PnP algorithm to estimate camera pose
        # _, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, camera_matrix, dist_coeffs)
    
        return rvec, tvec, inliers
    
    def extract_features(self, image, mask=None):
        object_id=1
        # Using ORB feature detector
        orb = cv2.ORB_create()
        #convert mask to binary mask
        binary_mask = np.where(mask == object_id, 255, 0).astype(np.uint8) 
        keypoints, descriptors = orb.detectAndCompute(image, binary_mask)
        return keypoints, descriptors
    
    def estimated_pose_from_images_superglue(self,rgb1, label1, depth1, rgb2, label2, camera_matrix, last_pose=None):
        superglue_model=self.model
        rgb1= rgb1*255.0
        rgb1=cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
    
        h,w,_= rgb1.shape
        gray1= cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)

        gray2= cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)


        # Perform dilation
        mask1= label1
        mask2= label2

        mask1= mask1 &(depth1>0)


        crop_gray1, x1, y1, w1, h1 = self.crop_image(gray1, cv2.resize(mask1.astype(np.uint8), (w, h)))
        crop_gray2, x2, y2, w2, h2 = self.crop_image(gray2, cv2.resize(mask2.astype(np.uint8), (w, h)))



        frame1= frame2tensor(crop_gray1, device='cuda')
        frame2= frame2tensor(crop_gray2, device='cuda')

        
        keypoints1, keypoints2, matches, confidence = self.superglue_extract_features(superglue_model,frame1, frame2)

        # Combine keypoints1, matches, and confidence into a single list of tuples
        data = list(zip(keypoints1, matches, confidence))


        # Sort the data based on confidence scores in descending order
        sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
        
        keypoints1, matches, confidence = zip(*sorted_data)

        keypoints1 = np.array(keypoints1)
        keypoints2 = np.array(keypoints2)
        matches = np.array(matches)
        confidence = np.array(confidence)
        #sort keypoints1 and matches based on confidence
        
        
        keypoints1[:, 0] += x1
        keypoints1[:, 1] += y1
        keypoints2[:, 0] += x2
        keypoints2[:, 1] += y2
        keypoints1_copy= keypoints1.copy()
        keypoints2_copy= keypoints2.copy()
        keypoints1_downscale=[]

        keypoints2_downscale=[]
        for kpt in keypoints1:
            keypoints1_downscale.append([kpt[0]/w*self.w, kpt[1]/h*self.h])
        for kpt in keypoints2:
            keypoints2_downscale.append([kpt[0]/w*self.w, kpt[1]/h*self.h])
        keypoints1= np.array(keypoints1_downscale)
        # print(keypoints1_downscale)
        keypoints2= np.array(keypoints2_downscale)
        # Get 2D points from matched keypoints
        valid=(matches>-1) 
        
        mkpts0= keypoints1[valid]
        mkpts1= keypoints2[matches[valid]]
        mkpts0_copy= keypoints1_copy[valid][0:30]
        mkpts1_copy= keypoints2_copy[matches[valid]][0:30]
        # filtered_pts0=[]
        # filtered_pts1=[]
        # for i in range(len(mkpts0)):
        #     if mask1[int(mkpts0[i][1]), int(mkpts0[i][0])]==1 and mask2[int(mkpts1[i][1]), int(mkpts1[i][0])]==1 and depth1[int(mkpts0[i][1]), int(mkpts0[i][0])]>0:
        #         filtered_pts0.append(mkpts0[i])
        #         filtered_pts1.append(mkpts1[i])
        
        
    
        # mkpts0= np.array(filtered_pts0)
        # mkpts1= np.array(filtered_pts1)
        # mkpts0= mkpts0[0:20]
        # mkpts1= mkpts1[0:20]
    
        
        pts1 = np.float32(mkpts0).reshape(-1, 1, 2)
        pts2 = np.float32(mkpts1).reshape(-1, 1, 2)

        #visualize the matches
        points_3d = get_3d_points(depth1, mkpts0, camera_matrix)    
        rvec, tvec, inliers = self.estimate_pose(points_3d, pts2, camera_matrix)
        R, _ = cv2.Rodrigues(rvec)

        # Construct pose matrix for the second frame
        pose2 = np.eye(4)
        pose2[:3, :3] = R
        pose2[:3, 3] = tvec.squeeze()

        # center_pose2 = (pose2 @ last_pose)@np.linalg.inv(self.to_origin)
        # center_pose1 = last_pose @ np.linalg.inv(self.to_origin)
        # gray1_copy= gray1.copy()
        # gray2_copy= gray2.copy()

        # gray1_copy= cv2.resize(gray1, (self.w, self.h))
        # gray2_copy= cv2.resize(gray2, (self.w, self.h))
        Km=self.K
        # Km= np.eye(3)
        # Km[0, 0]= self.K[0, 0]*3
        # Km[1, 1]= self.K[1, 1]*3
        # Km[0, 2]= self.K[0, 2]*3
        # Km[1, 2]= self.K[1, 2]*3
        # color = cm.jet(confidence)
        # vis1 = draw_posed_3d_box(Km, img=gray1_copy, ob_in_cam=center_pose1, bbox=self.bbox)
        # vis1 = draw_xyz_axis(gray1_copy, ob_in_cam=center_pose1, scale=0.1, K=Km, thickness=3, transparency=0, is_input_rgb=True)

        # vis2 = draw_posed_3d_box(Km, img=gray2_copy, ob_in_cam=center_pose2, bbox=self.bbox)
        # vis2 = draw_xyz_axis(gray2_copy, ob_in_cam=center_pose2, scale=0.1, K=Km, thickness=3, transparency=0, is_input_rgb=True)

        # out = make_matching_plot_fast(gray1_copy, gray2_copy, keypoints1_copy, keypoints2_copy, mkpts0_copy, mkpts1_copy, color, "",
        #         path=None, show_keypoints=True)
        # cv2.imwrite(f"./tmp/debug_matching.png", out)

        # projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
        
        # # Compute the Euclidean distance between actual 2D points and projected points
        # projected_points = projected_points.reshape(-1, 2)
        # pts2 = pts2.reshape(-1, 2)

        
        # # Calculate the reprojection error for each point
        # errors = np.linalg.norm(pts2 - projected_points, axis=1)
        
        # # Compute the mean error
        # mean_error = np.mean(errors)

        return pose2
    
    def crop_image(self, rgb, object_mask):
        x, y, w, h = cv2.boundingRect(object_mask.astype(np.uint8))
        cropped_rgb = rgb[y:y+h, x:x+w]

        return cropped_rgb,x,y,w,h

    def superglue_extract_features(self,superglue_model,image1_tensor, image2_tensor, device='cuda'):
        # Load the images as tensors
        with torch.no_grad():
            pred = superglue_model({'image0': image1_tensor, 'image1': image2_tensor})
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        return kpts0, kpts1, matches, confidence