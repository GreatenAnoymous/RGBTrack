
import mycpp
import torch
from scipy.spatial.transform import Rotation as R
from Utils import *
from tools import *
from estimater import *
from datareader import *
import sys
from kornia.feature import LoFTR
sys.path.append('/mnt/ssd_990/teng/SuperGluePretrainedNetwork/')
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
torch.set_grad_enabled(False)
def load_super_glue():
    # Load SuperGlue
    superglue = Matching()
    superglue.eval()
    superglue = superglue.cuda()
    return superglue
class FeatureMatchingEstimater(object):
    def __init__(self, cad_model, K, to_origin=None, bbox=None,w=640,h=480) -> None:
        self.cad_model = cad_model
        self.K = K
        self.to_origin = to_origin
        self.bbox = bbox
        self.w = w
        self.h = h
        
        # self.super_glue = load_super_glue()
        self.loftr=LoFTR(pretrained="indoor").eval().to("cuda")
        self.pose_last = None
        self.reset_object(
            cad_model.vertices, cad_model.vertex_normals, symmetry_tfs=None, mesh=mesh
        )
        self.make_rotation_grid(min_n_views=40, inplane_step=60)
        

    def generate_template(self, num_views=20,z=0.5):
        poses = self.rot_grid.clone()
        # random choose num_views from poses
        poses = poses[torch.randperm(poses.shape[0])[:num_views]]
        # poses[:, :3, 3] = np.array([0, 0, z])
        poses[:, :3, 3] = torch.tensor([0, 0, z], device="cuda", dtype=torch.float)
        rgbs=[]
        masks=[]
        depths=[]
        for i in range(num_views):
            pose = poses[i]
            rgb, mask, depth = render_rgbd(self.cad_model, pose, self.K, self.w, self.h)
            rgbs.append(rgb)
            # plt.imshow(rgb.cpu().numpy())
            # plt.savefig(f"./tmp/rgb{i}.png")
            # plt.close()
            masks.append(mask)
            depths.append(depth)
        self.templates ={"rgbs":rgbs, "masks":masks, "depths":depths, "poses":poses}


    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)

        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)

        rot_grid = mycpp.cluster_poses(
            30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy()
        )
        rot_grid = np.asarray(rot_grid)

        self.rot_grid = torch.as_tensor(rot_grid, device="cuda", dtype=torch.float)

    def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)
        model_pts = mesh.vertices
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        logging.info(f"self.diameter:{self.diameter}, vox_size:{self.vox_size}")
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        pcd = toOpen3dCloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(
            np.asarray(pcd.points), dtype=torch.float32, device="cuda"
        )
        self.normals = F.normalize(
            torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device="cuda"),
            dim=-1,
        )
        logging.info(f"self.pts:{self.pts.shape}")
        self.mesh_path = None
        self.mesh = mesh
        # if self.mesh is not None:
        #   self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
        #   self.mesh.export(self.mesh_path)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(
                symmetry_tfs, device="cuda", dtype=torch.float
            )

        logging.info("reset done")

    def superglue_extract_features(self,superglue_model,image1_tensor, image2_tensor, device='cuda'):
        # Load the images as tensors
        with torch.no_grad():
            pred = superglue_model({'image0': image1_tensor, 'image1': image2_tensor})
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        return kpts0, kpts1, matches, confidence
    

    def loftr_extract_features(self,loftr_model, image1_tensor, image2_tensor, device='cuda'):
        # Load the images as tensors
        input_dict = {'image0': image1_tensor, 'image1': image2_tensor,}
        with torch.no_grad():
            pred = loftr_model(input_dict)
        kpts0 = pred['keypoints0'].cpu().numpy()
        kpts1 = pred['keypoints1'].cpu().numpy()
        confidence = pred['confidence'][0].cpu().numpy()
        return kpts0, kpts1,  confidence
    
    # def extract_features(self, image, mask=None):

    #     # Using ORB feature detector
    #     orb = cv2.ORB_create()
    #     #convert mask to binary mask
    #     binary_mask = np.where(mask == object_id, 255, 0).astype(np.uint8) 
    #     keypoints, descriptors = orb.detectAndCompute(image, binary_mask)
    #     return keypoints, descriptors

    def crop_image(self, rgb, object_mask):
        x, y, w, h = cv2.boundingRect(object_mask.astype(np.uint8))
        cropped_rgb = rgb[y:y+h, x:x+w]

        return cropped_rgb,x,y,w,h
    

    def estimated_pose_from_images_loftr(self,loftr_model,rgb1, label1, depth1, rgb2, label2, debug=True):
        rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
        
        rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB)

        frame1= kornia.image_to_tensor(rgb1, False).float() / 255.
        frame2= kornia.image_to_tensor(rgb2, False).float() / 255.

        frame1  = kornia.color.rgb_to_grayscale(frame1)
        frame2  = kornia.color.rgb_to_grayscale(frame2)

        frame1 = frame1.to("cuda")
        frame2 = frame2.to("cuda")
        
        keypoints1, keypoints2, confidence = self.loftr_extract_features(loftr_model,frame1, frame2)
    
        # Get 2D points from matched keypoints
        valid=(confidence>0.1)
        mkpts0= keypoints1[valid].squeeze()
        mkpts1= keypoints2[valid].squeeze()
        filtered_pts0=[]
        filtered_pts1=[]

        for i in range(len(mkpts0)):
            if label1[int(mkpts0[i][1]), int(mkpts0[i][0])]==1 and label2[int(mkpts1[i][1]), int(mkpts1[i][0])]==1 and depth1[int(mkpts0[i][1]), int(mkpts0[i][0])]>0:
                filtered_pts0.append(mkpts0[i])
                filtered_pts1.append(mkpts1[i])
        mkpts0= np.array(filtered_pts0)
        mkpts1= np.array(filtered_pts1)
        # color = cm.jet(confidence)
        
        pts1 = np.float32(mkpts0).reshape(-1, 1, 2)
        pts2 = np.float32(mkpts1).reshape(-1, 1, 2)

        #visualize the matches
        points_3d = get_3d_points(depth1, mkpts0, self.K)    
        rvec, tvec, inliers = self.estimate_pose(points_3d, pts2, self.K)
        # inlier_matches = [matches[i] for i in inliers.flatten()]
        
        mkpts0 = np.array([mkpts0[i] for i in inliers.flatten()])
        mkpts1 = np.array([mkpts1[i] for i in inliers.flatten()])
        R, _ = cv2.Rodrigues(rvec)

        # Construct pose matrix for the second frame
        pose2 = np.eye(4)
        pose2[:3, :3] = R
        pose2[:3, 3] = tvec.squeeze()
        if debug:
            # center_pose2 = (pose2 @ self.history_pose)@np.linalg.inv(to_origin)
            # center_pose1 = self.history_pose @ np.linalg.inv(to_origin)
            rgb1_copy= rgb1.copy()
            rgb2_copy= rgb2.copy()
            # vis1 = draw_posed_3d_box(self.K, img=rgb1_copy, ob_in_cam=center_pose1, bbox=bbox)
            # vis1 = draw_xyz_axis(rgb1_copy, ob_in_cam=center_pose1, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)

            # vis2 = draw_posed_3d_box(self.K, img=rgb2_copy, ob_in_cam=center_pose2, bbox=bbox)
            # vis2 = draw_xyz_axis(rgb2_copy, ob_in_cam=center_pose2, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)

            out = self.draw_matches(rgb1_copy, rgb2_copy, mkpts0, mkpts1)
            cv2.imwrite(f"./tmp/matches_{self.idx}.png", out)
        return pose2
    
    def estimated_pose_from_images_super_glue(self,superglue_model,rgb1, mask1, depth1, rgb2, mask2):
        mask1=(depth1>0)
        h,w,_= rgb1.shape
        gray1= cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
        gray2= cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
        crop_gray1, x1, y1, w1, h1 = self.crop_image(gray1, cv2.resize(mask1.astype(np.uint8), (w, h)))
        crop_gray2, x2, y2, w2, h2 = self.crop_image(gray2, cv2.resize(mask2.astype(np.uint8), (w, h)))
        frame1= frame2tensor(crop_gray1, device='cuda')
        frame2= frame2tensor(crop_gray2, device='cuda')        
        keypoints1, keypoints2, matches, confidence = self.superglue_extract_features(superglue_model, frame1, frame2)
        # Combine keypoints1, matches, and confidence into a single list of tuples
        data = list(zip(keypoints1, matches, confidence))

        # Sort the data based on confidence scores in descending order
        sorted_data = sorted(data, key=lambda x: x[2], reverse=True)

        # Unzip and convert back to lists
        keypoints1, matches, confidence = map(list, zip(*sorted_data))

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
        pts2 = np.float32(mkpts1).reshape(-1, 1, 2)
        # Get 3D points
        points_3d = get_3d_points(depth1, mkpts0  ,self.K)

        # Estimate pose
        rvec, tvec, inliers = self.estimate_pose(points_3d, pts2, self.K)
        R, _ = cv2.Rodrigues(rvec)
        pose2 = np.eye(4)
        pose2[:3, :3] = R
        pose2[:3, 3] = tvec.squeeze()

        gray1_copy= gray1.copy()
        gray2_copy= gray2.copy()

        gray1_copy= cv2.resize(gray1, (self.w, self.h))
        gray2_copy= cv2.resize(gray2, (self.w, self.h))

        out = make_matching_plot_fast(gray1_copy, gray2_copy, keypoints1_copy, keypoints2_copy, mkpts0_copy, mkpts1_copy, color, "",
                path=None, show_keypoints=True)
        cv2.imwrite(f"./tmp/debug_matching.png", out)
        return pose2

    def get_pose_by_feature_matching(self, rgb1, depth1, mask1, pose1, rgb2, mask2):


        # Construct pose matrix for the second frame

        pose2 = pose1 @ pose2

        return pose2

    def filter_keypoints_by_mask(self, keypoints, descriptors, mask):
        filtered_keypoints = []
        filtered_descriptors = []
        
        for kp, desc in zip(keypoints, descriptors):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if mask[y, x] > 0:  # Assuming non-zero values indicate valid regions
                filtered_keypoints.append(kp)
                filtered_descriptors.append(desc)
        
        return filtered_keypoints, np.array(filtered_descriptors)

        

    def estimate_pose(self, points_3d, points_2d, K):
        rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None, flags=cv2.SOLVEPNP_EPNP
        )
        return rvec, tvec, inliers
    
    def extract_pose_parameters(self, poses):
        translations = []
        rotations = []
        for pose in poses:
            # Extract translation (x, y, z)
            translation = pose[:3, 3]
            translations.append(translation)

            # Extract rotation (convert to Euler angles if needed)
            rotation = pose[:3, :3]  # Rotation matrix
            # Convert rotation matrix to Euler angles (example for ZYX convention)
            sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
            euler_angles = np.array([
                np.arctan2(rotation[2, 1], rotation[2, 2]),  # Roll (x)
                np.arctan2(-rotation[2, 0], sy),             # Pitch (y)
                np.arctan2(rotation[1, 0], rotation[0, 0])   # Yaw (z)
            ])
            rotations.append(euler_angles)
        return np.array(translations), np.array(rotations)
    
    def calculate_mean_std(self, translations, rotations):
        # Calculate mean and std for translations
        trans_mean = np.mean(translations, axis=0)
        trans_std = np.std(translations, axis=0)

        # Calculate mean and std for rotations
        rot_mean = np.mean(rotations, axis=0)
        rot_std = np.std(rotations, axis=0)

        return trans_mean, trans_std, rot_mean, rot_std
    
    def filter_outliers(self, poses, translations, rotations, trans_mean, trans_std, rot_mean, rot_std, n_std=2):
        # Define thresholds
        trans_lower = trans_mean - n_std * trans_std
        trans_upper = trans_mean + n_std * trans_std
        rot_lower = rot_mean - n_std * rot_std
        rot_upper = rot_mean + n_std * rot_std

        # Filter poses
        filtered_poses = []
        for i, (trans, rot) in enumerate(zip(translations, rotations)):
            if (np.all(trans >= trans_lower) and np.all(trans <= trans_upper) and
                np.all(rot >= rot_lower) and np.all(rot <= rot_upper)):
                filtered_poses.append(poses[i])
        return filtered_poses
    
    def calculate_mean_pose(filtered_poses):
        # Compute mean translation
        mean_translation = np.mean([pose[:3, 3] for pose in filtered_poses], axis=0)

        # Compute mean rotation (average rotation matrices and re-normalize)
        mean_rotation = np.mean([pose[:3, :3] for pose in filtered_poses], axis=0)
        # Perform SVD to orthogonalize the mean rotation matrix
        U, _, Vt = np.linalg.svd(mean_rotation)
        mean_rotation = U @ Vt

        # Construct the mean pose matrix
        mean_pose = np.eye(4)
        mean_pose[:3, :3] = mean_rotation
        mean_pose[:3, 3] = mean_translation
        return mean_pose
    
    def robust_pose_estimation(self, poses):
        # Step 1: Extract pose parameters
        translations, rotations = self.extract_pose_parameters(poses)

        # Step 2: Calculate mean and std
        trans_mean, trans_std, rot_mean, rot_std = self.calculate_mean_std(translations, rotations)

        # Step 3: Filter outliers
        filtered_poses = self.filter_outliers(poses, translations, rotations, trans_mean, trans_std, rot_mean, rot_std, n_std=2)

        # Step 4: Recalculate the mean pose
        mean_pose = self.calculate_mean_pose(filtered_poses)
        return mean_pose

    def register(self, rgb, mask):
        self.generate_template()
        n_views=len(self.templates["rgbs"])
        poses=[]
        for i in range(n_views):
            try:
                rgbi = self.templates["rgbs"][i]
                plt.imshow(rgbi.cpu().numpy())
                plt.savefig("rgbi.png")
                plt.close()
                maski = self.templates["masks"][i]
                depthi = self.templates["depths"][i]
                posei = self.templates["poses"][i]
                # plt.imshow(depthi.cpu().numpy())
                # plt.savefig("depth1.png")
                # plt.close()
                # print(f"mask1:{maski.shape}")
                # print(f"mask2:{mask.shape}")
       
                # plt.imshow(maski.cpu().numpy())
                # plt.savefig("mask1.png")
                # plt.close()
                # plt.imshow(mask)
                # plt.savefig("mask2.png")
                # plt.close()

                # plt.imshow(rgbi.cpu().numpy())
                # plt.savefig("rgbi.png")
                # plt.close()
                pose = self.get_pose_by_feature_matching(rgbi, depthi, maski, posei, rgb, mask)
                poses.append(pose)
            except:
                # raise Exception(f"failed to register view {i}")
                print(f"failed to register view {i}")
                pass
        print(f"poses:{len(poses)}")
        pose = self.robust_pose_estimation(poses)
        self.pose_last = pose
        return pose

    def track_one(self, rgb):
        rgb, mask, depth = render_rgbd(self.cad_model, self.pose_last, self.K, self.w, self.h)
        pose = self.get_pose_by_feature_matching(rgb, depth, mask, self.pose_last, rgb, mask)
        self.pose_last = pose
        return pose
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj",
        # default=f"{code_dir}/data/CADmodels/021_bleach_cleanser/textured_simple.obj",
        # default=f"{code_dir}/data/CADmodels/003_cracker_box/textured_simple.obj",
        # default=f"{code_dir}/data/CADmodels/004_sugar_box/textured_simple.obj"
        # default=f"{code_dir}/data/CADmodels/005_tomato_soup_can/textured_simple.obj"
        
    )
    parser.add_argument(
        "--test_scene_dir", type=str, default=f"{code_dir}/data/mustard0"
        # "--test_scene_dir", type=str, default=f"{code_dir}/data/bleach0"
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
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf
    )

    i=0
    color=reader.get_color(i)
    depth=reader.get_depth(i)
    mask=reader.get_mask(i)
    
    matcher=FeatureMatchingEstimater(mesh, reader.K, to_origin, bbox)
    pose = matcher.register(color, mask)
    center_pose = pose @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(
            reader.K, img=color, ob_in_cam=center_pose, bbox=bbox
        )
    vis = draw_xyz_axis(
        color,
        ob_in_cam=center_pose,
        scale=0.1,
        K=reader.K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    plt.imshow(vis)
    plt.savefig("matching.png")