# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import torch

from tools import PoseTracker
from scipy.spatial.transform import Rotation as R
from tools import render_cad_depth, render_cad_mask, render_rgbd


class FoundationPose:
    def __init__(
        self,
        model_pts,
        model_normals,
        symmetry_tfs=None,
        mesh=None,
        scorer: ScorePredictor = None,
        refiner: PoseRefinePredictor = None,
        glctx=None,
        debug=0,
        debug_dir="/home/bowen/debug/novel_pose_debug/",
    ):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        self.tracker = PoseTracker()

        self.reset_object(
            model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh
        )
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = glctx

        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh

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

    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device="cuda")
        tf_to_center[:3, 3] = -torch.as_tensor(
            self.model_center, device="cuda", dtype=torch.float
        )
        return tf_to_center

    def to_device(self, s="cuda:0"):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            if torch.is_tensor(self.__dict__[k]) or isinstance(
                self.__dict__[k], nn.Module
            ):
                logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)
        for k in self.mesh_tensors:
            logging.info(f"Moving {k} to device {s}")
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)

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

    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
        """
        @scene_pts: torch tensor (N,3)
        """
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(
            center, device="cuda", dtype=torch.float
        ).reshape(1, 3)
        return ob_in_cams

    def generate_random_pose_hypo_rgb(self, K, rgb, mask, center):
        ob_in_cams = self.rot_grid.clone()
        ob_in_cams[:, :3, 3] = torch.tensor(
            center, device="cuda", dtype=torch.float
        ).reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            logging.info(f"mask is all zero")
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            logging.info(f"valid is empty")
            return np.zeros((3))

        zc = np.median(depth[valid])

        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        if self.debug >= 2:
            pcd = toOpen3dCloud(center.reshape(1, 3))
            o3d.io.write_point_cloud(f"{self.debug_dir}/init_center.ply", pcd)

        return center.reshape(3)

    def render_rgbd(self, cad_model, object_pose, K, W, H):
        pose_tensor = torch.from_numpy(object_pose).float().to("cuda")

        mesh_tensors = make_mesh_tensors(cad_model)

        rgb_r, depth_r, normal_r = nvdiffrast_render(
            K=K,
            H=H,
            W=W,
            ob_in_cams=pose_tensor,
            context="cuda",
            get_normal=False,
            glctx=self.glctx,
            mesh_tensors=mesh_tensors,
            output_size=[H, W],
            use_light=True,
        )
        mask_r = (depth_r > 0).float()
        return rgb_r, depth_r, mask_r

        # return best_pose.data.cpu().numpy()
        
    def generate_ref_views(self, dz=0.5, num_views=16):
        poses=self.rot_grid.clone()
        center=[0,0,dz]
        poses[:, :3, 3] = torch.tensor(center, device="cuda", dtype=torch.float).reshape(1, 3)
        # randomly choose 16 views
        indices = torch.linspace(0, poses.size(0) - 1, steps=num_views, dtype=torch.long)
        poses = poses[indices]
        rgbds=[]
        for i in range(num_views):
            pose=poses[i]
            rgb, depth, mask = render_rgbd(self.mesh, pose.data.cpu().numpy(), self.K, self.W, self.H)
            rgbds.append((rgb, depth, mask))
        
        return poses, rgbds
    
    def register_super_gule(self, rgb, mask, K,  device='cuda'):
        fme=FeatureMathcerPoseEstimator()
        posesA, rgbdsA=self.generate_ref_views(dz=0.5, num_views=16)
        for i in range(len(rgbdsA)):
            rgb_i, depth_i, mask_i=rgbdsA[i]
            rl_pose=fme.estimated_pose_from_images_superglue(rgb_i, mask_i, depth_i, K, rgb, mask, K)
            posesA[i]=rl_pose @posesA[i]
        depth_arrays=[depth_i for rgb_i, depth_i, mask_i in rgbdsA]
        normal_map = None
        xyz_maps=[depth2xyzmap(depth_i, K) for depth_i in depth_arrays]
        iteration=5
        with torch.no_grad():
            posesA, vis = self.refiner.predict_depth_batches(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depths=depth_arrays,
                K=K,
                ob_in_cams=posesA,
                normal_map=normal_map,
                xyz_map=xyz_maps,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )
        torch.cuda.empty_cache()
        if vis is not None:
            imageio.imwrite(f"{self.debug_dir}/vis_refiner.png", vis)
        with torch.no_grad():
            scores, vis = self.scorer.predict(
                mesh=self.mesh,
                rgb=rgb,
                depth=depth_arrays,
                K=K,
                ob_in_cams=posesA,
                normal_map=normal_map,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                get_vis=self.debug >= 2,
            )
        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]
    
        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.xyz = self.pose_last[:3, 3]
        self.best_id = ids[0]
        self.mask_last = mask
        self.poses = poses
        self.scores = scores
        self.track_good = True
        rgb_r, depth_r, mask_r = render_rgbd(
            cad_model=self.mesh,
            object_pose=best_pose.data.cpu().numpy(),
            K=K,
            W=rgb.shape[1],
            H=rgb.shape[0],
        )
        self.last_depth = torch.tensor(depth_r, device="cuda", dtype=torch.float)

        return best_pose.data.cpu().numpy()


    def choose_k_best(self, poses, given_pose, k=30):
        """
        Select k poses from `poses` whose rotation matrices are closest to the rotation matrix of `given_pose`.

        Args:
        - poses: (N, 4, 4) tensor containing N poses with homogeneous transformation matrices.
        - given_pose: (4, 4) tensor representing the reference pose.
        - k: Number of closest poses to select based on rotation matrix similarity.

        Returns:
        - k_best_poses: (k, 4, 4) tensor containing the k closest poses.
        """
        # Extract rotation matrices (first 3x3 block of each pose)

        rotations = poses[:, :3, :3]  # Shape: (N, 3, 3)
        given_rotation = given_pose[0, :3, :3]  # Shape: (3, 3)

        # Compute Frobenius norm distance between rotation matrices
        distances = torch.norm(rotations - given_rotation, dim=(1, 2))  # Shape: (N,)

        # Find indices of the k smallest distances
        topk_indices = torch.topk(-distances, k).indices  # Negative for closest

        # Select the corresponding poses
        k_best_poses = poses[topk_indices]

        return k_best_poses
    
    

    def register(
        self,
        K,
        rgb,
        depth,
        ob_mask,
        ob_id=None,
        glctx=None,
        iteration=5,
        return_score=False,
        rough_depth_guess=None
    ):
        """Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        """
        set_seed(0)
        logging.info("Welcome")

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()

            else:
                self.glctx = glctx

        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")
        if self.debug >= 2:
            xyz_map = depth2xyzmap(depth, K)
            valid = xyz_map[..., 2] >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(f"{self.debug_dir}/scene_raw.ply", pcd)
            cv2.imwrite(f"{self.debug_dir}/ob_mask.png", (ob_mask * 255.0).clip(0, 255))

        normal_map = None
        if  rough_depth_guess is None:
            valid = (depth >= 0.001) & (ob_mask > 0)
        else:
            print('using rough depth guess')
            valid=(ob_mask>0)
        if valid.sum() < 4:
            logging.info(f"valid too small, return")
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        if self.debug >= 2:
            imageio.imwrite(f"{self.debug_dir}/color.png", rgb)
            cv2.imwrite(f"{self.debug_dir}/depth.png", (depth * 1000).astype(np.uint16))
            valid = xyz_map[..., 2] >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(f"{self.debug_dir}/scene_complete.ply", pcd)

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask
        self.last_depth = depth
        if rough_depth_guess is not None:
            depth_guess=np.ones_like(depth)*rough_depth_guess
            center = self.guess_translation(depth=depth_guess, mask=ob_mask, K=K)
            poses = self.generate_random_pose_hypo(
                K=K, rgb=rgb, depth=depth_guess, mask=ob_mask, scene_pts=None
            )
        else:
            center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            poses=self.generate_random_pose_hypo(K=K, rgb=rgb,depth=depth, mask=ob_mask)
        

        #   num_samples = 100
        #   indices = torch.linspace(0, poses.size(0) - 1, steps=num_samples, dtype=torch.long)

        # # Select the matrices using the computed indices
        #   poses = poses[indices]


        
        # Assuming poses is a tensor of shape [252, 4, 4]

        
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")

        add_errs = self.compute_add_err_to_gt_pose(poses)

        xyz_map = depth2xyzmap(depth, K)

        t0 = time.time()
        poses, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses,
            normal_map=normal_map,
            xyz_map=xyz_map,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            iteration=iteration,
            get_vis=self.debug >= 2,
        )
        t1 = time.time()
        logging.info(f"refiner time:{t1-t0}")
        if vis is not None:
            imageio.imwrite(f"{self.debug_dir}/vis_refiner.png", vis)
        t0 = time.time()
        scores, vis = self.scorer.predict(
            mesh=self.mesh,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses,
            normal_map=normal_map,
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            get_vis=self.debug >= 2,
        )
        t1 = time.time()
        logging.info(f"scorer time:{t1-t0}")
        if vis is not None:
            imageio.imwrite(f"{self.debug_dir}/vis_score.png", vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)
        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        logging.info(f"last pose:{self.pose_last}")  
        self.xyz = self.pose_last[:3, 3]
        r = R.from_matrix(self.pose_last[:3, :3].data.cpu().numpy())
        euler_angles = r.as_euler("xyz").reshape(3, 1)
        self.tracker.initialize(
            self.xyz.detach().cpu().numpy().reshape(3, 1), euler_angles
        )
        self.best_id = ids[0]
        self.mask_last = ob_mask
        self.poses = poses
        self.scores = scores
        self.track_good = True
        if return_score:
            return best_pose.data.cpu().numpy(), scores[0]
        return best_pose.data.cpu().numpy()

    def register_without_depth(
        self,
        K,
        rgb,
        ob_mask,
        ob_id=None,
        glctx=None,
        iteration=5,
        low=0.2,
        high=2,
        return_score=False,
    ):
        """Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        """
        set_seed(0)
        logging.info("Welcome")

        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()

            else:
                self.glctx = glctx
        normal_map = None
        num_samples = self.rot_grid.size(0)
        depth_samples = np.linspace(low, high, 5)
        self.H, self.W = ob_mask.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask
        self.last_depth = None

        poses = self.rot_grid.clone()
        # poses = self.generate_random_pose_hypo_rgb(K=K, rgb=rgb,  mask=ob_mask)
        indices = torch.linspace(
            0,
            poses.size(0) - 1,
            steps=num_samples,
            dtype=torch.long,
        )

        # Select the matrices using the computed indices
        poses = poses[indices]
        poses = poses.repeat(len(depth_samples), 1, 1)
        logging.info(f"without using depth poses:{poses.shape}")
        depth_arrays = []
        j = 0
        for i in range(len(depth_samples)):
            for k in range(num_samples):
                logging.info(f"depth_samples:{depth_samples[i]}")
                depth = np.ones_like(ob_mask) * depth_samples[i]
                depth_arrays.append(depth)
                center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
                poses[j, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")
                # logging.info(f"center:{center}")
                # logging.info(f"z={poses[i,2,3]}")
                j += 1

    

        depth_arrays = np.array(depth_arrays)
        # center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
        # Assuming poses is a tensor of shape [252, 4, 4]


        add_errs = self.compute_add_err_to_gt_pose(poses)
    
        # xyz_map = depth2xyzmap(depth, K)
        xyz_maps = []
        for i in range(len(depth_arrays)):
            xyz_map = depth2xyzmap(depth_arrays[i], K)
            xyz_maps.append(xyz_map)
            
        
    
        with torch.no_grad():
            poses, vis = self.refiner.predict_depth_batches(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depths=depth_arrays,
                K=K,
                ob_in_cams=poses,
                normal_map=normal_map,
                xyz_map=xyz_map,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )
        torch.cuda.empty_cache()

        # for i,pose in enumerate(poses):
        #   rgb_r, depth_r, mask_r = render_rgbd(cad_model=self.mesh, object_pose=pose.data.cpu().numpy(), K=K, W=rgb.shape[1], H=rgb.shape[0])
        #   plt.subplot(1,2,1)
        #   plt.imshow(rgb_r)
        #   plt.subplot(1,2,2)
        #   plt.imshow(rgb)
        #   plt.savefig(f'./tmp/debug_{depth_samples[i]}.png')

        # exit()
        if vis is not None:
            imageio.imwrite(f"{self.debug_dir}/vis_refiner.png", vis)
        with torch.no_grad():
            scores, vis = self.scorer.predict(
                mesh=self.mesh,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=poses,
                normal_map=normal_map,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                get_vis=self.debug >= 2,
            )
        if vis is not None:
            imageio.imwrite(f"{self.debug_dir}/vis_score.png", vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)


        ids = torch.as_tensor(scores).argsort(descending=True)
        # logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]
        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.xyz = self.pose_last[:3, 3]
        self.best_id = ids[0]
        self.mask_last = ob_mask
        self.poses = poses
        self.scores = scores
        self.track_good = True
        rgb_r, depth_r, mask_r = render_rgbd(
            cad_model=self.mesh,
            object_pose=best_pose.data.cpu().numpy(),
            K=K,
            W=rgb.shape[1],
            H=rgb.shape[0],
        )
        self.last_depth = torch.tensor(depth_r, device="cuda", dtype=torch.float)
        if return_score:
            return best_pose.data.cpu().numpy(), scores[0]
        return best_pose.data.cpu().numpy()

    def compute_add_err_to_gt_pose(self, poses):
        """
        @poses: wrt. the centered mesh
        """
        return -torch.ones(len(poses), device="cuda", dtype=torch.float)

    def track_one(self, rgb, depth, K, iteration, extra={}):
        if self.pose_last is None:
            logging.info("Please init pose by register first")
            raise RuntimeError
        logging.info("Welcome")

        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")
    

        xyz_map = depth2xyzmap_batch(
            depth[None],
            torch.as_tensor(K, dtype=torch.float, device="cuda")[None],
            zfar=np.inf,
        )[0]

        pose, vis = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=self.pose_last.reshape(1, 4, 4),
            normal_map=None,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            glctx=self.glctx,
            iteration=iteration,
            get_vis=self.debug >= 2,
        )
        logging.info("pose done")
        if self.debug >= 2:
            extra["vis"] = vis
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)


    # trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
    def track_one_new(self, rgb, depth, K, iteration, mask, extra={}):
        if self.pose_last is None:
            logging.info("Please init pose by register first")
            raise RuntimeError
        threshold = 40

        if np.sum(mask) <= threshold:

            predict=self.tracker.update()
            xyz_predict = np.squeeze(predict["position"])
            orientation= predict["orientation"]
            predict_pose = np.eye(4)
            predict_pose[:3, 3] = xyz_predict
            matrix = R.from_euler("xyz", np.squeeze(orientation)).as_matrix()
            predict_pose[:3, :3] = matrix
            self.pose_last = torch.tensor(predict_pose, device="cuda", dtype=torch.float)
            self.track_good = False
            self.mask_last = mask
           
            return (
                (self.pose_last @ self.get_tf_to_centered_mesh())
                .data.cpu()
                .numpy()
                .reshape(4, 4)
            )
        if depth is None:
            depth = self.last_depth
        depth = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth = erode_depth(depth, radius=2, device="cuda")
        depth = bilateral_filter_depth(depth, radius=2, device="cuda")
        xyz_map = depth2xyzmap_batch(
            depth[None],
            torch.as_tensor(K, dtype=torch.float, device="cuda")[None],
            zfar=np.inf,
        )[0]

        if self.track_good == False:
            predict = self.tracker.update()
            depth = depth.detach().cpu().numpy()
            poses = self.generate_random_pose_hypo(
                K=K, rgb=rgb, depth=depth, mask=mask, scene_pts=None
            )
            num_samples = 10
            indices = torch.linspace(
                0, poses.size(0) - 1, steps=num_samples, dtype=torch.long
            )

            # Select the matrices using the computed indices
            poses = poses[indices]
            center = self.guess_translation(depth=depth, mask=mask, K=K)
            # poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")

            xyz_map = depth2xyzmap(depth, K)
        
            poses, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=poses.data,
                normal_map=None,
                xyz_map=xyz_map,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )

            scores, vis = self.scorer.predict(
                mesh=self.mesh,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=poses.data.cpu().numpy(),
                normal_map=None,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                get_vis=self.debug >= 2,
            )
            ids = torch.as_tensor(scores).argsort(descending=True)
            poses = poses[ids]
            best_pose = poses[0] @ self.get_tf_to_centered_mesh()
            pose = best_pose
            self.pose_last = pose
            xyz = self.pose_last[:3, 3].detach().cpu().numpy()
            if np.abs(xyz - center).sum() > 0.1:
                self.track_good = False
            self.track_good = True
            self.last_depth = np.ones_like(mask) * xyz[2]
        else:
            center = self.guess_translation(
                depth=depth.detach().cpu().numpy(), mask=mask, K=K
            )

            pose, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=self.pose_last.reshape(1, 4, 4),
                normal_map=None,
                xyz_map=xyz_map,
                mesh_diameter=self.diameter,
                glctx=self.glctx,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )
            xyz = pose[0, :3, 3].detach().cpu().numpy()
            if np.abs(xyz - center).sum() > 0.1:
                self.track_good = False
            self.pose_last = pose
            self.last_depth=np.ones_like(mask) * xyz[2]
            euler_angles = R.from_matrix(
                pose[0, :3, :3].detach().cpu().numpy()
            ).as_euler("xyz")

            measurement = np.concatenate((xyz, euler_angles)).reshape(6, 1)

            self.tracker.update(measurement)
        

        self.mask_last = mask
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)

    def track_one_new_without_depth(self, rgb, K, iteration, mask, extra={}):
        if self.pose_last is None:
            logging.info("Please init pose by register first")
            raise RuntimeError
        logging.info("Welcome")
        threshold = 40

        if np.sum(mask) <= threshold:
            logging.info("lost tracking as there is no mask")
            self.track_good = False
            self.mask_last = mask
            return (
                (self.pose_last @ self.get_tf_to_centered_mesh())
                .data.cpu()
                .numpy()
                .reshape(4, 4)
            )

        if self.track_good == False:
            logging.info("lost tracking using Kalman filter")
            predict = self.tracker.update()
            xyz_predict = np.squeeze(predict["position"])
            orientation = predict["orientation"]

            predict_pose = np.eye(4)
            matrix = R.from_euler("xyz", np.squeeze(orientation)).as_matrix()

            predict_pose[:3, :3] = matrix
            predict_pose[:3, 3] = xyz_predict

            depth = np.ones_like(mask) * xyz_predict[2]

            poses = self.generate_random_pose_hypo(
                K=K, rgb=rgb, depth=depth, mask=mask, scene_pts=None
            )

            num_samples = 20
            indices = torch.linspace(
                0, poses.size(0) - 1, steps=num_samples, dtype=torch.long
            )

            # Select the matrices using the computed indices
            poses = poses[indices]


            center = self.guess_translation(depth=depth, mask=mask, K=K)

            # poses = torch.as_tensor(poses, device="cuda", dtype=torch.float)
            poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device="cuda")
            depth=np.zeros_like(mask)
            xyz_map = depth2xyzmap(depth, K)
    
            t0 = time.time()
            poses, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=poses,
                normal_map=None,
                xyz_map=xyz_map,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )


            scores, vis = self.scorer.predict(
                mesh=self.mesh,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_in_cams=poses,
                normal_map=None,
                mesh_tensors=self.mesh_tensors,
                glctx=self.glctx,
                mesh_diameter=self.diameter,
                get_vis=self.debug >= 2,
            )
        
            ids = torch.as_tensor(scores).argsort(descending=True)

            poses = poses[ids]

            best_pose = poses[0] @ self.get_tf_to_centered_mesh()
            pose = best_pose
            self.pose_last = pose
            xyz = self.pose_last[:3, 3].detach().cpu().numpy()

            if np.abs(xyz - center).sum() > 0.1:
                self.track_good = False
            self.track_good = True
            self.last_depth = np.ones_like(mask) * xyz[2]
            # self.last_depth = render_cad_depth(pose.cpu().numpy(), self.mesh, K,  w=rgb.shape[0],h=rgb.shape[1],)
        else:
            center = self.guess_translation(depth=self.last_depth, mask=mask, K=K)
            zero_depth = np.zeros_like(mask)
            zero_depth = torch.as_tensor(zero_depth, device="cuda", dtype=torch.float)
            xyz_map = depth2xyzmap_batch(
                zero_depth[None],
                torch.as_tensor(K, dtype=torch.float, device="cuda")[None],
                zfar=np.inf,
            )[0]
            pose, vis = self.refiner.predict(
                mesh=self.mesh,
                mesh_tensors=self.mesh_tensors,
                rgb=rgb,
                depth=zero_depth,
                K=K,
                ob_in_cams=self.pose_last.reshape(1, 4, 4),
                normal_map=None,
                xyz_map=xyz_map,
                mesh_diameter=self.diameter,
                glctx=self.glctx,
                iteration=iteration,
                get_vis=self.debug >= 2,
            )
            xyz = pose[0, :3, 3].detach().cpu().numpy()
            euler_angles = R.from_matrix(
                pose[0, :3, :3].detach().cpu().numpy()
            ).as_euler("xyz")

            measurement = np.concatenate((xyz, euler_angles)).reshape(6, 1)

            self.tracker.update(measurement)

            self.pose_last = pose

            # self.last_depth=render_cad_depth(pose.cpu().numpy(), self.mesh, K,  w=rgb.shape[0],h=rgb.shape[1],)    
            self.last_depth = np.ones_like(mask) * xyz[2]
            gt_mask = render_cad_mask(
                pose.cpu().numpy(), self.mesh, K, w=rgb.shape[0], h=rgb.shape[1]
            )
            
            if 0.8*np.sum(mask)>np.sum(gt_mask):
                print("tracking is not good as  it is too far")
                self.track_good = False
                measurement[2]=measurement[2]*0.9
                self.tracker.update(measurement)

    
            # if np.sum(mask)<0.6*np.sum(gt_mask):
            #     self.track_good = False

            if np.abs(xyz - center).sum() > 0.1:
                print("tracking is not good as the difference is more than 0.1")
                self.track_good = False
            # if np.sum(gt_mask)>4*np.sum(mask):
                # self.track_good = False
                # measurement[2]=measurement[2]*1.2
                # self.tracker.update(measurement)

        self.mask_last = mask
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
