import numpy as np
import cv2
import open3d as o3d
from utils import make_colored_pcd, depth_to_pcd_world

# -----------------------------
# ROI 마스크 생성 (보드 영역만)
# -----------------------------
def board_mask(rgb_shape, K, rvec, tvec, pattern_size, square_size, margin=3):
    """보드 네 모서리를 투영해 마스크 생성"""
    cols, rows = pattern_size
    W = (cols-1) * square_size
    H = (rows-1) * square_size
    corners_w = np.array([[0,0,0],[W,0,0],[W,H,0],[0,H,0]], np.float32).reshape(-1,1,3)
    img_pts, _ = cv2.projectPoints(corners_w, rvec, tvec, K, None)
    poly = img_pts.reshape(-1,2).astype(np.int32)
    mask = np.zeros(rgb_shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1))
        mask = cv2.dilate(mask, kernel)
    return mask

def preprocess_pcd(pcd, voxel=0.005, nb_neighbors=20, std_ratio=2.0, estimate_normals=True):
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3 if voxel else 0.02, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(50)
    if nb_neighbors > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

# -----------------------------
# 멀티스케일 point-to-plane ICP (강건 커널)
# -----------------------------
def icp_multiscale(source, target, voxel_scales=(0.02, 0.01, 0.005), max_iters=(50, 30, 20)):
    assert len(voxel_scales) == len(max_iters)
    T = np.eye(4)
    T_new = np.eye(4)
    current_src = source
    for scale, iters in zip(voxel_scales, max_iters):
        src = preprocess_pcd(current_src, voxel=scale)
        tgt = preprocess_pcd(target,       voxel=scale)

        loss = o3d.pipelines.registration.TukeyLoss(k=scale*3.0)
        est  = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)

        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_correspondence_distance=scale*4.0,
            init=T, estimation_method=est, criteria=crit
        )
        T = reg.transformation
        current_src = source.transform(T.copy())
        T_new = T.copy()@T_new
        print(f"[ICP] scale={scale:.3f} iters={iters}  fitness={reg.fitness:.3f}  rmse={reg.inlier_rmse:.4f}")
    return T, T_new

def multiscale_icp(
    depth_path1, rgb_path1, K1, D1, R1, T1,
    depth_path2, rgb_path2, K2, D2, R2, T2,
    depth_scale, z_min, z_max, stride=4
):
    # 1) Generate PCD(World coordinate)
    pts1, col1 = depth_to_pcd_world(depth_path1, rgb_path1, K1, D1, R1, T1, depth_scale, z_min, z_max, stride)
    pts2, col2 = depth_to_pcd_world(depth_path2, rgb_path2, K2, D2, R2, T2, depth_scale, z_min, z_max, stride)
    pcd1 = make_colored_pcd(pts1, col1)
    pcd2 = make_colored_pcd(pts2, col2)
    
    # 2) ICP (pcd2 → pcd1)
    T_icp, T_temp = icp_multiscale(pcd2, pcd1, voxel_scales=(0.02, 0.01, 0.005, 0.0001), max_iters=(120, 160, 180, 200))
    pcd2.transform(T_icp)

    # 3) Visualization
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries([axis, pcd1, pcd2])
    
    T_new = T_icp@T_temp
    return T_new
