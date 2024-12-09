import numpy as np
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from Utils import *
import torch


def render_rgbd(cad_model, object_pose, K, W, H):
    pose_tensor= torch.from_numpy(object_pose).float().to("cuda")
    mesh_tensors = make_mesh_tensors(cad_model)
    glctx =  dr.RasterizeCudaContext()
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=pose_tensor, context='cuda', get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=[H,W], use_light=True)
    rgb_r = rgb_r.squeeze().cpu().numpy()
    depth_r = depth_r.squeeze().cpu().numpy()
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




def render_cad_depth(pose, mesh_model,K):
    h=480
    w=640
    # Load the mesh model and apply the center pose transformation
    
    vertices = np.array(mesh_model.vertices)
    
    # Transform vertices with the center pose
    transformed_vertices = (pose @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T).T[:, :3]

    # Project vertices to the 2D plane using the intrinsic matrix K
    projected_points = (K @ transformed_vertices.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # Normalize by z

    # Initialize a depth map and project depth values into it
    image_size = (h, w)
    depth_map = np.zeros(image_size, dtype=np.float32)
    for i, point in enumerate(projected_points):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            depth_map[y, x] = transformed_vertices[i, 2]
    return depth_map



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
