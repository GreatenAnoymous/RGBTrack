import numpy as np
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from Utils import *
from scipy.spatial.distance import cdist
import torch
from scipy.spatial import ConvexHull
from filterpy.kalman import KalmanFilter

from scipy.spatial.transform import Rotation
from bop_toolkit_lib import pose_error


from bop_toolkit_lib import misc
from bop_toolkit_lib import visibility

def render_rgbd(cad_model, object_pose, K, W, H):
    if type(object_pose) is np.ndarray:
        pose_tensor = torch.from_numpy(object_pose).float().to("cuda")
    else:
        pose_tensor = object_pose
    # pose_tensor= torch.from_numpy(object_pose).float().to("cuda")
    mesh_tensors = make_mesh_tensors(cad_model)
    glctx =  dr.RasterizeCudaContext()
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    rgb_r = rgb_r.squeeze()
    depth_r = depth_r.squeeze()
    mask_r = (depth_r > 0)
    return rgb_r, depth_r, mask_r





class PoseTracker:
    def __init__(self, dt=0.001):
        """
        Initialize a 6D pose tracker (position + orientation) with Kalman Filter
        dt: time step between measurements
        """
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        # 12-dimensional state: position, velocity, orientation, angular rates
        self.kf = KalmanFilter(dim_x=12, dim_z=6)  
        
        # State transition matrix
        self.kf.F = np.zeros((12, 12))
        # Position and velocity
        self.kf.F[0:3, 0:3] = np.eye(3)
        self.kf.F[0:3, 3:6] = np.eye(3) * dt
        self.kf.F[3:6, 3:6] = np.eye(3)
        # Orientation and angular rates
        self.kf.F[6:9, 6:9] = np.eye(3)
        self.kf.F[6:9, 9:12] = np.eye(3) * dt
        self.kf.F[9:12, 9:12] = np.eye(3)
        
        # Measurement matrix (we measure position and orientation)
        self.kf.H = np.zeros((6, 12))
        self.kf.H[0:3, 0:3] = np.eye(3)  # Position measurements
        self.kf.H[3:6, 6:9] = np.eye(3)  # Orientation measurements
        
        # Measurement noise
        # self.kf.R = np.eye(6)
        # self.kf.R[0:3, 0:3] *= 0.1  # Position measurement noise
        # self.kf.R[3:6, 3:6] *= 0.2  # Orientation measurement noise
        
        # Process noise
        # pos_vel_noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        # angle_rate_noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.2)
        
        # self.kf.Q = np.zeros((12, 12))
        # # Apply noise to position-velocity states
        # for i in range(3):
        #     self.kf.Q[i*2:(i+1)*2, i*2:(i+1)*2] = pos_vel_noise
        # # Apply noise to orientation-angular rate states
        # for i in range(3):
        #     self.kf.Q[(i*2+6):(i*2+8), (i*2+6):(i*2+8)] = angle_rate_noise
        
        # Initial state covariance
        self.kf.P = np.eye(12) * 100000
        
        self.is_initialized = False
        
    def normalize_angles(self, angles):
        """
        Normalize angles to [-pi, pi]
        """
        return np.mod(angles + np.pi, 2 * np.pi) - np.pi
        
    def initialize(self, position, orientation):
        """
        Initialize the tracker with first pose measurement
        position: [x, y, z]
        orientation: [roll, pitch, yaw]
        """

        self.kf.x[0:3] = position
        self.kf.x[6:9] = self.normalize_angles(orientation)
    
        self.is_initialized = True

    def get_current_pose(self):
        pose=np.eye(4)
        pose[:3, 3] = np.squeeze(self.kf.x[0:3])
        pose[:3, :3] = Rotation.from_euler("xyz", np.squeeze(self.kf.x[6:9])).as_matrix()
        return pose

    def predict_next_pose(self):
        """
        Predict the next pose without updating the Kalman filter state.
        Returns: predicted position, orientation, velocity, and angular rates
        """
        if not self.is_initialized:
            raise ValueError("Tracker not initialized!")
        
        # Compute the predicted state using the state transition matrix
        predicted_state = self.kf.F @ self.kf.x
        
        # Normalize the predicted orientation angles
        predicted_state[6:9] = self.normalize_angles(predicted_state[6:9])
        
        return {
            'position': predicted_state[0:3],
            'orientation': predicted_state[6:9],
            'velocity': predicted_state[3:6],
            'angular_rates': predicted_state[9:12]
        }
        
    def update(self, measurement=None):
        """
        Update the state estimate. If measurement is None, predict without update
        measurement: [x, y, z, roll, pitch, yaw] or None during occlusion
        Returns: position and orientation estimates
        """
        if not self.is_initialized:
            raise ValueError("Tracker not initialized!")
            
        # Predict next state
        self.kf.predict()
        # Normalize orientation states
        self.kf.x[6:9] = self.normalize_angles(self.kf.x[6:9])
    
        # Update with measurement if available
        if measurement is not None:
            # Normalize measured orientation angles
            measurement[3:6] = self.normalize_angles(measurement[3:6])

            # Handle angle wrapping in measurement update
            innovation = measurement - self.kf.H @ self.kf.x
        
            innovation[3:6] = self.normalize_angles(innovation[3:6])
            
            # Custom update to handle angle wrapping
            PHT = self.kf.P @ self.kf.H.T
            S = self.kf.H @ PHT + self.kf.R
            K = PHT @ np.linalg.inv(S)
    
            self.kf.x = self.kf.x + K @ innovation
            assert self.kf.x.shape == (12,1)
            self.kf.P = (np.eye(12) - K @ self.kf.H) @ self.kf.P
            
        # Return current pose estimate
    
        return {
            'position': self.kf.x[0:3],
            'orientation': self.normalize_angles(self.kf.x[6:9]),
            'velocity': self.kf.x[3:6],
            'angular_rates': self.kf.x[9:12]
        }
    
    def get_uncertainty(self):
        """
        Return the uncertainty in position and orientation estimates
        """
        return {
            'position_std': np.sqrt(np.diag(self.kf.P)[0:3]),
            'orientation_std': np.sqrt(np.diag(self.kf.P)[6:9])
        }
    
def evaluate_metrics(history_poses,reader, mesh):
    """
    Evaluate the tracking performance using the ground truth poses.
    history_poses: List of estimated poses at each frame
    reader: DatasetReader object
    Returns: List of errors (rotation, translation) for each frame
    """
    # errors = []
    vertices = mesh.vertices
    pairwise_distances = cdist(vertices, vertices)  # Use scipy.spatial.distance.cdist
    diameter_exact = np.max(pairwise_distances)
    data={"ADD":0, "ADD-S":0,  "rotation_error_deg":0, "translation_error":0, "mspd":0,"mssd":0, "recall":0, "AR_mspd":0, "AR_mssd":0, "AR_vsd":0}
    for i, pose in enumerate(history_poses):
        gt_pose = reader.get_gt_pose(i)
        tmp_data= evaluate_pose(gt_pose, pose, mesh, diameter_exact, reader.K)
        for key in tmp_data:
            data[key]+=tmp_data[key]
    for key in data:
        data[key]/=len(history_poses)
    return data
    
def demo_tracking():
    """
    Demonstrate tracker usage with simulated occlusion
    """
    # Create tracker
    tracker = PoseTracker()
    
    # Initialize with first position
    initial_pos = np.array([0., 0., 0.])
    tracker.initialize(initial_pos)
    
    # Simulate some measurements with occlusion
    measurements = [
        [0.1, 0.1, 0.1],    # Visible
        [0.2, 0.2, 0.2],    # Visible
        None,               # Occluded
        None,               # Occluded
        [0.5, 0.5, 0.5]     # Visible again
    ]
    
    positions = []
    uncertainties = []
    
    for measurement in measurements:
        pos = tracker.update(measurement)
        uncertainty = tracker.get_position_uncertainty()
        
        positions.append(pos)
        uncertainties.append(uncertainty)
        
        status = "OCCLUDED" if measurement is None else "VISIBLE"
        print(f"Status: {status}")
        print(f"Estimated position: {pos}")
        print(f"Position uncertainty: {uncertainty}\n")
        
    return positions, uncertainties
import numpy as np




def render_cad_depth_nvidia(pose, mesh_model, K, w=640, h=480):
    """
    Render depth image from a CAD model using the given camera pose and intrinsic matrix.

    Args:
        pose (np.ndarray): 4x4 camera pose matrix (world to camera transformation).
        mesh_model (np.ndarray): Nx3 array of 3D mesh vertices.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        w (int): Width of the output depth image.
        h (int): Height of the output depth image.

    Returns:
        np.ndarray: Depth image of size (h, w).
    """
    pose_tensor= torch.from_numpy(pose).float().to("cuda")
    mesh_tensors = make_mesh_tensors(mesh_model)
    glctx =  dr.RasterizeCudaContext()
    depth_r= nvdiffrast_render_depthonly(K=K, H=h, W=w, ob_in_cams=pose_tensor, context='cuda', glctx=glctx, mesh_tensors=mesh_tensors, output_size=[h,w])
    depth_r = depth_r.squeeze()
    return depth_r.cpu().numpy()

def render_cad_depth(pose, mesh_model, K, w=640, h=480):
    """
    Render a depth map using a CAD model and its pose.
    
    Parameters:
    pose: 4x4 numpy array - Transformation matrix
    mesh_model: Trimesh object - CAD model
    K: 3x3 numpy array - Camera intrinsic matrix
    w, h: int - Width and height of the depth image

    Returns:
    depth_map: numpy array of shape (h, w) - Rendered depth map
    """
    vertices = np.array(mesh_model.vertices)
    # Transform vertices
    transformed_vertices = (pose @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T).T[:, :3]

    # Project vertices to 2D
    projected_points = (K @ transformed_vertices.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]

    # Round projected points to nearest pixel
    pixel_coords = np.round(projected_points).astype(int)

    # Clip points that fall outside the image dimensions
    valid = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
    )
    pixel_coords = pixel_coords[valid]
    depths = transformed_vertices[valid, 2]

    # Compute the depth map using vectorized indexing
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    for i in range(len(pixel_coords)):
        x, y = pixel_coords[i]
        depth_map[y, x] = min(depth_map[y, x], depths[i])

    # Replace inf with 0 (background)
    depth_map[depth_map == np.inf] = 0
    return depth_map

    # Extract rotation and translation from the pose matrix
    
def render_cad_mask(pose, mesh_model, K, w=640, h=480):
    """
    Renders the binary mask of the object based on its pose, CAD model, and camera parameters.

    Args:
        pose (np.ndarray): 4x4 transformation matrix of the object's pose.
        mesh_model: Mesh object containing vertices of the CAD model.
        K (np.ndarray): 3x3 intrinsic matrix of the camera.
        w (int): Image width.
        h (int): Image height.

    Returns:
        np.ndarray: Binary mask of the object (1 for object pixels, 0 for background).
    """
    # Load the vertices from the mesh model
    vertices = np.array(mesh_model.vertices)
    sample_indices = np.random.choice(len(vertices), size=500, replace=False)
    vertices = vertices[sample_indices]

    # Transform vertices with the object pose
    transformed_vertices = (pose @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T).T[:, :3]

    # Project vertices to the 2D plane using the intrinsic matrix K
    projected_points = (K @ transformed_vertices.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # Normalize by z

    # Create a polygon from the projected 2D points
    polygon = np.int32(projected_points).reshape((-1, 1, 2))

    # Initialize a blank mask and draw the polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)

    return mask



def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Parameters:
    - mask1: np.ndarray, first binary mask.
    - mask2: np.ndarray, second binary mask.

    Returns:
    - iou: float, the IoU value.
    """
    # Ensure the masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    return iou


def compute_error(pose1, pose2):
    """
    Computes the rotation error (in degrees) and translation error (in meters) between two poses.
    
    Parameters:
    - pose1: (4x4 numpy array) Transformation matrix representing pose 1.
    - pose2: (4x4 numpy array) Transformation matrix representing pose 2.
    
    Returns:
    - rotation_error: The angular difference in degrees between the two poses.
    - translation_error: The Euclidean distance between the translations of the two poses.
    """
    # Extract rotation matrices (upper-left 3x3 submatrix)
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Extract translation vectors (rightmost 3 elements of the 4th column)
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    
    # Compute the relative rotation matrix R_rel = R1_inv * R2
    R_rel = np.dot(R1.T, R2)
    
    # Compute the rotation error as the angle of the relative rotation
    # trace(R_rel) = 1 + 2*cos(theta), where theta is the rotation angle
    trace_R_rel = np.trace(R_rel)
    theta = np.arccos(np.clip((trace_R_rel - 1) / 2.0, -1.0, 1.0))  # theta in radians
    
    # Convert the rotation error to degrees
    rotation_error = np.degrees(theta)
    
    # Compute the translation error as the Euclidean distance between t1 and t2
    translation_error = np.linalg.norm(t1 - t2)
    
    return rotation_error, translation_error

        
def get_3d_points(depth_image, keypoints, camera_matrix):
    points_3d = []
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    for kp in keypoints:
        try:
            u, v = int(kp.pt[0]), int(kp.pt[1])
        except:
            u,v=int(kp[0]), int(kp[1])
        z = depth_image[v, u] # assuming depth is in millimeters
        # z = depth_image[u, v] # assuming depth is in millimeters
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    
        points_3d.append([x, y, z])
    
    return np.array(points_3d)



def get_pose_icp(self, pointcloud1, pointcloud2):
    """
    Perform ICP (Iterative Closest Point) registration to compute the relative transformation 
    from pointcloud1 to pointcloud2.

    :param pointcloud1: Source point cloud as a numpy array of shape (N, 3)
    :param pointcloud2: Target point cloud as a numpy array of shape (M, 3)
    :return: 4x4 transformation matrix representing the transformation from pointcloud1 to pointcloud2
    """
    
    # Convert numpy arrays to Open3D PointCloud objects
    
    print("Pointcloud1: ", pointcloud1)
    print("Pointcloud2: ", pointcloud2)
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    source.points = o3d.utility.Vector3dVector(pointcloud1)
    target.points = o3d.utility.Vector3dVector(pointcloud2)
    
    # Estimate normals (required for point-to-plane ICP)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Perform point-to-plane ICP (this usually gives better results than point-to-point)
    threshold = 0.02  # Distance threshold for matching points
    initial_transformation = np.eye(4)  # No initial transformation
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        threshold, 
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    # Get the transformation matrix from the ICP result
    transformation = icp_result.transformation

    return transformation



#"vsd_taus": list(np.arange(0.05, 0.51, 0.05)),
def vsd(
    depth_gt,
    depth_est,
    K,
    delta,
    taus,
    diameter,
    cost_type="step",
):
    """Visible Surface Discrepancy -- by Hodan, Michel et al. (ECCV 2018).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param depth_test: hxw ndarray with the test depth image.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param taus: A list of misalignment tolerance values.
    :param normalized_by_diameter: Whether to normalize the pixel-wise distances
        by the object diameter.
    :param diameter: Object diameter.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :param cost_type: Type of the pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16
        'step' - Used for SIXD Challenge 2017 onwards.
    :return: List of calculated errors (one for each misalignment tolerance).
    """
    # Render depth images of the model in the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Convert depth images to distance images.
    dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im_fast(depth_est, K)

    # Visibility mask of the model in the ground-truth pose.
    visib_gt = visibility.estimate_visib_mask_gt(
        dist_gt,dist_gt, delta, visib_mode="bop19"
    )

    # Visibility mask of the model in the estimated pose.
    visib_est = visibility.estimate_visib_mask_est(
        dist_gt, dist_est, visib_gt, delta, visib_mode="bop19"
    )

    # Intersection and union of the visibility masks.
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)
    

    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()

    # Pixel-wise distances.
    dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    normalized_by_diameter = True
    # Normalization of pixel-wise distances by object diameter.
    if normalized_by_diameter:
        dists /= diameter

    # Calculate VSD for each provided value of the misalignment tolerance.
    if visib_union_count == 0:
        errors = [1.0] * len(taus)
    else:
        errors = []
        for tau in taus:
            # Pixel-wise matching cost.
            if cost_type == "step":
                costs = dists >= tau
            elif cost_type == "tlinear":  # Truncated linear function.
                costs = dists / tau
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError("Unknown pixel matching cost.")

            e = (np.sum(costs) + visib_comp_count) / float(visib_union_count)
            errors.append(e)

    return errors


#   "vsd": [0.3],
#         "mssd": [0.2],
#         "mspd": [10],

# thresholds=np.linspace(0.05, 0.5, 10)
def evaluate_pose(gt_pose, est_pose, mesh, diameter,K ):
    """
    Evaluate 6D pose estimation performance using ADD and ADD-S metrics.
    
    Args:
        gt_pose (np.ndarray): 4x4 ground truth pose matrix
        est_pose (np.ndarray): 4x4 estimated pose matrix
        model_points (np.ndarray): Nx3 array of 3D model points
        diameter (float): diameter of the object
        thresholds (np.ndarray): distance thresholds for AUC computation
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """

    def transform_points(points, pose):
        """Transform 3D points using pose matrix."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        return np.dot(points, R.T) + t

    def compute_add(gt_points, est_points):
        """Compute ADD metric (average distance between corresponding points)."""
        return np.mean(np.linalg.norm(gt_points - est_points, axis=1))

    def compute_adds(gt_points, est_points):
        """Compute ADD-S metric (average distance to nearest neighbor)."""
        distances = np.zeros((gt_points.shape[0],))
        for i, gt_point in enumerate(gt_points):
            distances[i] = np.min(np.linalg.norm(gt_point - est_points, axis=1))
        return np.mean(distances)

    # Transform model points using ground truth and estimated poses
    gt_transformed = transform_points(mesh.vertices , gt_pose)
    est_transformed = transform_points(mesh.vertices , est_pose)

    # Compute ADD and ADD-S
    add_value = compute_add(gt_transformed, est_transformed)
    adds_value = compute_adds(gt_transformed, est_transformed)

    # # Compute success rates and AUC for different thresholds
    # add_success_rates = []
    # adds_success_rates = []
    # # print("thresholds: ", thresholds)
    # # print("add_value: ", add_value)
    # # print("adds_value: ", adds_value)
    
    # for threshold in thresholds:
    #     # Normalize threshold by object diameter

    #     normalized_threshold = threshold * diameter
        
    #     # Compute ADD success rate
    #     add_success = (add_value < normalized_threshold)
    #     add_success_rates.append(float(add_success))
        
    #     # Compute ADD-S success rate
    #     adds_success = (adds_value < normalized_threshold)
    #     adds_success_rates.append(float(adds_success))
    # print("add_success_rates: ", add_success_rates)
    # print("diameter: ", diameter)
    # Compute AUC (normalize thresholds to 0-1 range for AUC computation)
    # normalized_thresholds = thresholds / thresholds[-1]
    # add_auc = auc(normalized_thresholds, add_success_rates)
    # adds_auc = auc(normalized_thresholds, adds_success_rates)

    # Extract rotation error
    gt_R = gt_pose[:3, :3]
    est_R = est_pose[:3, :3]
    R_diff = np.dot(est_R, gt_R.T)
    trace_value = np.trace(R_diff)
    theta_rad = np.arccos(np.clip((trace_value - 1) / 2, -1.0, 1.0))  # Avoid numerical errors
    rotation_error = np.degrees(theta_rad)  # Convert to degrees
    depth_gt=render_cad_depth_nvidia(gt_pose, mesh, K)
    depth_est=render_cad_depth_nvidia(est_pose, mesh, K)

    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    taus=list(np.arange(0.05, 0.51, 0.05))
    delta=15
    # taus=[0.5]
    theta=np.arange(0.05, 0.51, 0.05)

    vsd_errors=vsd(depth_gt, depth_est, K, delta, taus, diameter, cost_type="step")
    AR_vsd =0
    for err in vsd_errors:
        if err<=0.3:
            AR_vsd+=1
    AR_vsd/=10
    # vsd_error=np.mean(vsd_errors)
    error_mspd=pose_error.mspd(est_pose[:3, :3], est_pose[:3, 3].reshape(3,1),gt_pose[:3, :3], gt_pose[:3, 3].reshape(3,1), K, mesh.vertices, [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}])
    error_mssd=pose_error.mssd(est_pose[:3, :3], est_pose[:3, 3].reshape(3,1),gt_pose[:3, :3], gt_pose[:3, 3].reshape(3,1), mesh.vertices, [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}])
    AR_mspd =0 
    AR_mssd =0
    for th in np.arange(5,51,5):
        if error_mspd<=th:
            AR_mspd+=1
    AR_mspd/=10
    for th in theta*diameter:
        if error_mssd<=th:
            AR_mssd+=1
    AR_mssd/=10
    
    # Extract translation error
    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    res={
        'ADD': add_value,
        'ADD-S': adds_value,
        'rotation_error_deg': rotation_error,
        'translation_error': translation_error,
        "recall": (AR_mspd+AR_mssd+AR_vsd)/3,
        "mspd": error_mspd,
        "mssd": error_mssd,
        "AR_vsd": AR_vsd,
        "AR_mspd": AR_mspd,
        "AR_mssd": AR_mssd    
        # 'add_success_rates': add_success_rates,
        # 'adds_success_rates': adds_success_rates,
        # 'thresholds': thresholds
    }
    print("res: ", res)
    return res
    

# def evaluate_pose_bop(gt_pose, est_pose, K, model_points):
#     syms=[{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]
#     Rgt=gt_pose[:3, :3]
#     tgt=gt_pose[:3, 3]
#     Rest=est_pose[:3, :3]
#     test=est_pose[:3, 3]
#     err_mspd = pose_error.compute_mspd( Rest, test,Rgt, tgt,K, model_points, syms)
#     err_mssd = pose_error.compute_mssd( Rest, test,Rgt, tgt, model_points, syms)

def save_poses_to_txt(file_path, poses):
    """
    Save a list of 4x4 np.matrix to a text file.
    
    Parameters:
    - poses: A list of np.matrix (4x4 matrices).
    - file_path: The file path where the matrices will be saved.
    """
    # Verify that all poses are 4x4 matrices
    for pose in poses:
        if pose.shape != (4, 4):
            raise ValueError(f"Each pose must be a 4x4 matrix. Found shape: {pose.shape}")
    
    # Convert np.matrix objects to np.ndarray (for easier handling)
    poses_array = [np.array(pose) for pose in poses]
    
    # Stack them into a single numpy array
    stacked_poses = np.stack(poses_array)
    
    # Save to a text file (flatten the matrices into rows of 16 values)
    np.savetxt(file_path, stacked_poses.reshape(-1, 16), delimiter=' ')
    print(f"Saved {len(poses)} poses to {file_path}")

def read_poses_from_txt(file_path):
    """
    Read a list of 4x4 np.matrix from a text file.
    
    Parameters:
    - file_path: The file path from which the matrices will be read.
    
    Returns:
    - A list of np.matrix (4x4 matrices).
    """
    # Load the flattened array from the text file
    loaded_poses = np.loadtxt(file_path)
    
    # Verify that the number of elements is a multiple of 16
    if loaded_poses.size % 16 != 0:
        raise ValueError("The file does not contain a valid number of elements for 4x4 matrices.")
    
    # Reshape the array into 4x4 matrices
    poses = []
    for i in range(0, loaded_poses.shape[0], 16):
        pose = loaded_poses[i:i+16].reshape(4, 4)
        poses.append(np.matrix(pose))  # Convert back to np.matrix
    
    print(f"Loaded {len(poses)} poses from {file_path}")
    return poses
