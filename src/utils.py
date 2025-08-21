import cv2
import numpy as np
import open3d as o3d

def make_colored_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_colored_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

def depth_to_pcd_world(depth_path, rgb_path, K, D, R, T, depth_scale, z_min, z_max, stride=4):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_img   = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    if depth_raw is None or rgb_img is None:
        raise FileNotFoundError("Depth or RGB image not found.")

    if depth_raw.dtype != np.float32:
        depth = depth_raw.astype(np.float32) * depth_scale
    else:
        depth = depth_raw

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth.shape

    points_world, colors = [], []
    for v in range(0, h, stride):
        for u in range(0, w, stride):
            z = depth[v, u]
            if z < z_min or z > z_max or z <= 0 or np.isnan(z):
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            X_cam = np.array([x, y, z], dtype=np.float32)
            X_world = (X_cam @ R.T) + T.reshape(3)  # cam_n → orgin camera
            
            points_world.append(X_world)
            colors.append(rgb_img[v, u] / 255.0)

    if not points_world:
        return np.empty((0,3), np.float32), np.empty((0,3), np.float32)
    return np.asarray(points_world, np.float32), np.asarray(colors, np.float32)

def visualize_pcd_world(
    depth_path1, rgb_path1, K1, D1, R1, T1,
    depth_path2, rgb_path2, K2, D2, R2, T2,
    depth_scale, z_min, z_max,
    stride=4, ret=True
):
    pts1_world, col1 = depth_to_pcd_world(depth_path1, rgb_path1, K1, D1, R1, T1, depth_scale, z_min, z_max, stride)
    pts2_world, col2 = depth_to_pcd_world(depth_path2, rgb_path2, K2, D2, R2, T2, depth_scale, z_min, z_max, stride)

    pcd1 = make_colored_pcd(pts1_world, col1)
    pcd2 = make_colored_pcd(pts2_world, col2)

    axis_board = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    o3d.visualization.draw_geometries([axis_board, pcd1, pcd2])
    if ret:
        return pcd1, pcd2