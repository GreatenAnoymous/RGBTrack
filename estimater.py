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
import yaml
import torch


def render_rgbd(cad_model, object_pose, K, W, H):
    pose_tensor= torch.from_numpy(object_pose).float().to("cuda")

    mesh_tensors = make_mesh_tensors(cad_model)
   
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=None, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    rgb_r = rgb_r.squeeze().cpu().numpy()
    depth_r = depth_r.squeeze().cpu().numpy()
    mask_r = (depth_r > 0)
    return rgb_r, depth_r, mask_r





class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
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

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
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
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    logging.info("ob_in_cams shape",ob_in_cams.shape)
    return ob_in_cams

  def generate_random_pose_hypo_rgb(self, K, rgb, mask, center):
    ob_in_cams=self.rot_grid.clone()
    ob_in_cams[:,:3,3]=torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams



  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    logging.info(f'uc:{uc}, vc:{vc}, zc:{zc}')
  
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)
  
  def render_rgbd(self,cad_model, object_pose, K, W, H):
    pose_tensor= torch.from_numpy(object_pose).float().to("cuda")

    mesh_tensors = make_mesh_tensors(cad_model)
  
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    mask_r = (depth_r > 0).float()
    return rgb_r, depth_r, mask_r

    # return best_pose.data.cpu().numpy()

  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, return_score=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    set_seed(0)
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()

      else:
        self.glctx = glctx
    
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
   
    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    valid = (depth>=0.001) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)
    # Assuming poses is a tensor of shape [252, 4, 4]
    
    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')
   
    
    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")
    logging.info(f'poses:{poses.shape}')
    xyz_map = depth2xyzmap(depth, K)
    logging.info(f'depth:{depth.shape}, rgb:{rgb.shape}, xyz_map:{xyz_map.shape}')
    t0 = time.time()
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    t1 = time.time()
    logging.info(f"refiner time:{t1-t0}")
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    # logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]



    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.xyz=self.pose_last[:3,3]
    self.best_id = ids[0]
    self.mask_last= ob_mask
    self.poses = poses
    self.scores = scores
    self.track_good=True
    if return_score:
      return best_pose.data.cpu().numpy(), scores[0]
    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)

  def track_one_new(self, rgb, depth, K, iteration,mask ,extra={}):
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")
    threshold = 40
    
    if np.sum(mask) <= threshold:
      logging.info("lost tracking")
      self.track_good=False
      self.mask_last = mask
      return (self.pose_last@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    logging.info("depth processing done")
    
    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]
    
    if self.track_good==False:
      
      depth=depth.detach().cpu().numpy()
      poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=mask, scene_pts=None)
      num_samples = 10
      indices = torch.linspace(0, poses.size(0) - 1, steps=num_samples, dtype=torch.long)

    # Select the matrices using the computed indices
      poses = poses[indices]
      poses = poses.data.cpu().numpy()
      
      logging.info(f'poses:{poses.shape}')
      center = self.guess_translation(depth=depth, mask=mask, K=K)

      poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
      poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

     

      xyz_map = depth2xyzmap(depth, K)
      logging.info(f'depth:{depth.shape}, rgb:{rgb.shape}, xyz_map:{xyz_map.shape}')
      t0 = time.time()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
      t1 = time.time()
      logging.info(f"refiner time:{t1-t0}")
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)
      logging.info("refiner done")
      t0 = time.time()
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=None, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
      t1 = time.time()
      logging.info(f"scorer time:{t1-t0}")
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)


      ids = torch.as_tensor(scores).argsort(descending=True)
    
    
      poses = poses[ids]

      logging.info(f'sorted scores:{scores}')

      best_pose = poses[0]@self.get_tf_to_centered_mesh()
      pose=best_pose
      self.pose_last = pose
      xyz=self.pose_last[:3,3].detach().cpu().numpy()
      if np.abs(xyz-center).sum()>0.1:
        self.track_good=False
      self.track_good=True
    else:
      center = self.guess_translation(depth=depth.detach().cpu().numpy(), mask=mask, K=K)
    
      pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
      xyz = pose[0,:3,3].detach().cpu().numpy()
      if np.abs(xyz-center).sum()>0.1:
        self.track_good=False
      self.pose_last = pose
    
    self.mask_last = mask
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
    
  def check_reprojection(self,pose, mesh, K):
    rgb_r, depth_r, mask_r = render_rgbd(cad_model=mesh, object_pose=pose, K=K, W=rgb.shape[1], H=rgb.shape[0])
    translation= self.guess_translation(depth=depth_r, mask=mask_r, K=K)
    return translation


