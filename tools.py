import numpy as np
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from Utils import *
import torch
from scipy.spatial import ConvexHull

def render_rgbd(cad_model, object_pose, K, W, H):
    pose_tensor= torch.from_numpy(object_pose).float().to("cuda")
    mesh_tensors = make_mesh_tensors(cad_model)
    glctx =  dr.RasterizeCudaContext()
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    rgb_r = rgb_r.squeeze()
    depth_r = depth_r.squeeze()
    mask_r = (depth_r > 0)
    return rgb_r, depth_r, mask_r

def render_image(mesh_model, camera_pose, camera_intrinsics, image_size):
    # Render image
    mesh_model.visual.face_colors = [200, 200, 250, 100]
    image = trimesh.scene.CameraScene(camera=camera_intrinsics, bg_color=(255, 255, 255, 255), resolution=(image_size, image_size))
    image.add_geometry(mesh_model)
    image.set_camera(camera_pose)
    image = image.save_image()
    return image


import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
class PoseTracker:
    def __init__(self, dt=0.1):
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
        self.kf.R = np.eye(6)
        self.kf.R[0:3, 0:3] *= 0.1  # Position measurement noise
        self.kf.R[3:6, 3:6] *= 0.2  # Orientation measurement noise
        
        # Process noise
        pos_vel_noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        angle_rate_noise = Q_discrete_white_noise(dim=2, dt=dt, var=0.2)
        
        self.kf.Q = np.zeros((12, 12))
        # Apply noise to position-velocity states
        for i in range(3):
            self.kf.Q[i*2:(i+1)*2, i*2:(i+1)*2] = pos_vel_noise
        # Apply noise to orientation-angular rate states
        for i in range(3):
            self.kf.Q[(i*2+6):(i*2+8), (i*2+6):(i*2+8)] = angle_rate_noise
        
        # Initial state covariance
        self.kf.P = np.eye(12) * 1000
        
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
        # print("kf.x: ", self.kf.x, self.kf.x.shape)
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



def render_cad_depth(pose, mesh_model,K,w=640,h=480):

    # Load the mesh model and apply the center pose transformation
    
    vertices = np.array(mesh_model.vertices)
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]
    #random sample 1000 points
    # idx = np.random.choice(vertices.shape[0], 1000, replace=False)
    # vertices = vertices[idx]
    # Transform vertices with the center pose
    transformed_vertices = (pose @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T).T[:, :3]

    # Project vertices to the 2D plane using the intrinsic matrix K
    projected_points = (K @ transformed_vertices.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # Normalize by z

    # Initialize a depth map and project depth values into it
    image_size = (w, h)
    depth_map = np.zeros(image_size, dtype=np.float32)
    for i, point in enumerate(projected_points):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            depth_map[y, x] = transformed_vertices[i, 2]
    return depth_map


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


def create_object_mask(mesh_model, pose, intrinsics, image_shape):
    """
    Generate a binary mask of the object given its pose and camera intrinsics.

    Parameters:
    - mesh_model: Trimesh, the 3D model of the object.
    - pose: np.ndarray, 4x4 transformation matrix representing the object's pose relative to the camera.
    - intrinsics: np.ndarray, 3x3 camera intrinsic matrix.
    - image_shape: tuple, shape of the output image mask (height, width).

    Returns:
    - mask: np.ndarray, binary mask where the object is set to 1.
    """
    # Get vertices of the object in the world space
    vertices = mesh_model.vertices
    vertices = np.c_[vertices, np.ones((vertices.shape[0], 1))]  # Convert to homogeneous coordinates

    # Apply the pose transformation
    vertices_camera = (pose @ vertices.T).T  # Transform vertices to camera coordinates

    # Project the 3D points to 2D image coordinates
    vertices_2d = (intrinsics @ vertices_camera[:, :3].T).T
    vertices_2d[:, :2] /= vertices_2d[:, 2:3]  # Divide by depth to get 2D points

    # Initialize mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Remove points that fall outside the image frame
    valid_points = (vertices_2d[:, 0] >= 0) & (vertices_2d[:, 0] < image_shape[1]) & \
                (vertices_2d[:, 1] >= 0) & (vertices_2d[:, 1] < image_shape[0])
    vertices_2d = vertices_2d[valid_points]

    # Get the pixel coordinates of the projected vertices
    pixel_coords = vertices_2d[:, :2].astype(int)

    # Fill mask using the object's convex hull in 2D
    hull = trimesh.path.polygons.convex_hull(pixel_coords)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if hull.contains_point([j, i]):
                mask[i, j] = 1

    return mask

def create_mask_from_3d_bbox(img_shape, bbox, ob_in_cam, intrinsic_matrix):
    
    min_xyz=bbox.min(axis=0)
    max_xyz=bbox.max(axis=0)
    xmin, ymin, zmin = min_xyz
    xmax, ymax, zmax = max_xyz
    img=np.zeros(img_shape)
    def draw_line3d(start,end,mask,line_color=(255),linewidth=2):
        pts = np.stack((start,end),axis=0).reshape(-1,3)
        pts = (ob_in_cam@to_homo(pts).T).T[:,:3]   #(2,3)
        projected = (intrinsic_matrix@pts.T).T
        uv = np.round(projected[:,:2]/projected[:,2].reshape(-1,1)).astype(int)   #(2,2)
        mask = cv2.line(mask, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
        return mask
    for y in [ymin,ymax]:
        for z in [zmin,zmax]:
            start = np.array([xmin,y,z])
            end = start+np.array([xmax-xmin,0,0])
            img = draw_line3d(start,end,img)

    for x in [xmin,xmax]:
        for z in [zmin,zmax]:
            start = np.array([x,ymin,z])
            end = start+np.array([0,ymax-ymin,0])
            img = draw_line3d(start,end,img)

    for x in [xmin,xmax]:
        for y in [ymin,ymax]:
            start = np.array([x,y,zmin])
            end = start+np.array([0,0,zmax-zmin])
            img = draw_line3d(start,end,img)
    # convert img to binary
    img=(img/255).astype(np.uint8)
    # Find contours of the drawn boundary
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the area inside the contours on the original mask
  

    for contour in contours:
        cv2.drawContours(img, [contour], -1, color=(255), thickness=cv2.FILLED)


    return img

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



def area_contained_percentage(mask1, mask2):
    """
    Calculate the percentage of the area of mask2 that is contained within mask1.

    Parameters:
    - mask1: np.ndarray, the first binary mask (reference mask).
    - mask2: np.ndarray, the second binary mask (target mask).

    Returns:
    - percentage: float, percentage of the area of mask2 that is within mask1.
    """
    # Ensure masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate intersection between mask1 and mask2
    intersection = np.logical_and(mask1, mask2).sum()
    
    # Calculate the area of mask2
    area_mask2 = mask2.sum()
    
    # Calculate percentage
    percentage = (intersection / area_mask2) * 100 if area_mask2 > 0 else 0.0
    return percentage



import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.metrics import auc
# thresholds=np.linspace(0.05, 0.5, 10)
def evaluate_pose(gt_pose, est_pose, model_points, diameter, thresholds=np.arange(0.05, 0.51, 0.05)):
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
    # print("gt_pose: ", gt_pose)
    # print("est_pose: ", est_pose)
    

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
    gt_transformed = transform_points(model_points, gt_pose)
    est_transformed = transform_points(model_points, est_pose)

    # Compute ADD and ADD-S
    add_value = compute_add(gt_transformed, est_transformed)
    adds_value = compute_adds(gt_transformed, est_transformed)

    # Compute success rates and AUC for different thresholds
    add_success_rates = []
    adds_success_rates = []
    
    for threshold in thresholds:
        # Normalize threshold by object diameter
        normalized_threshold = threshold * diameter
        
        # Compute ADD success rate
        add_success = (add_value < normalized_threshold)
        add_success_rates.append(float(add_success))
        
        # Compute ADD-S success rate
        adds_success = (adds_value < normalized_threshold)
        adds_success_rates.append(float(adds_success))
    # print("add_success_rates: ", add_success_rates)
    # print("diameter: ", diameter)
    # Compute AUC (normalize thresholds to 0-1 range for AUC computation)
    normalized_thresholds = thresholds / thresholds[-1]
    add_auc = auc(normalized_thresholds, add_success_rates)
    adds_auc = auc(normalized_thresholds, adds_success_rates)

    # Extract rotation error
    gt_R = gt_pose[:3, :3]
    est_R = est_pose[:3, :3]
    R_diff = np.dot(est_R, gt_R.T)
    rotation_error = np.arccos((np.trace(R_diff) - 1) / 2) * 180 / np.pi

    # Extract translation error
    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    res={
        'ADD': add_value,
        'ADD-S': adds_value,
        'ADD_AUC': add_auc,
        'ADD-S_AUC': adds_auc,
        'rotation_error_deg': rotation_error,
        'translation_error': translation_error,
        'recall':np.sum(add_success_rates)/len(add_success_rates),
        # 'add_success_rates': add_success_rates,
        # 'adds_success_rates': adds_success_rates,
        # 'thresholds': thresholds
    }
    print("res: ", res)
    # exit()
    return res
    
import numpy as np

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
