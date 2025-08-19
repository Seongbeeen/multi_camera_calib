import numpy as np
import cv2
import open3d as o3d

DEPTH_SCALE = 0.001   # mm→m 변환
MIN_Z, MAX_Z = 0.05, 1.5  # m

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


def make_colored_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_two_colored_depthmaps_in_world(
    depth_path1, rgb_path1, K1, D1, R1, T1,
    depth_path2, rgb_path2, K2, D2, R2, T2,
    stride=4, ret=True
):
    pts1_world, col1 = depth_to_colored_pointcloud_world(depth_path1, rgb_path1, K1, D1, R1, T1, stride)
    pts2_world, col2 = depth_to_colored_pointcloud_world(depth_path2, rgb_path2, K2, D2, R2, T2, stride)

    pcd1 = make_colored_pcd(pts1_world, col1)
    pcd2 = make_colored_pcd(pts2_world, col2)

    axis_board = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    o3d.visualization.draw_geometries([axis_board, pcd1, pcd2])
    if ret:
        return pcd1, pcd2
    
def depth_to_colored_pointcloud_world(depth_path, rgb_path, K, D, R, T, stride=4):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_img   = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    if depth_raw is None or rgb_img is None:
        raise FileNotFoundError("Depth or RGB image not found.")

    if depth_raw.dtype != np.float32:
        depth = depth_raw.astype(np.float32) * DEPTH_SCALE
    else:
        depth = depth_raw

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth.shape

    points_world, colors = [], []
    for v in range(0, h, stride):
        for u in range(0, w, stride):
            z = depth[v, u]
            if z < MIN_Z or z > MAX_Z or z <= 0 or np.isnan(z):
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

def make_colored_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

# -----------------------------
# 보조: 전처리(다운샘플/노말/아웃라이어 제거)
# -----------------------------
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
        # 다음 레벨 입력을 위해 원본 source에 누적 변환 적용
        current_src = source.transform(T.copy())
        # print(scale, iters, "\n", np.asarray(current_src.points[:3]))
        T_new = T.copy()@T_new
        print(f"[ICP] scale={scale:.3f} iters={iters}  fitness={reg.fitness:.3f}  rmse={reg.inlier_rmse:.4f}")
    return T, T_new

# -----------------------------
# 포즈(rvec,tvec) 갱신(선택)
#   - pcd2는 world 좌표에서 T_icp로 보정됨: X'_world = T_icp * X_world
#   - cam2의 world 포즈(4x4)도 동일하게 왼쪽곱으로 갱신: World_T_Cam2' = T_icp * World_T_Cam2
#   - 이후 역변환해 board->cam 형식(R',t') 추출
# -----------------------------
def world_T_cam_from_rt(R, t):
    # world<-cam = [R^T | -R^T t]
    T = np.eye(4)
    T[:3,:3] = R.T
    T[:3, 3] = (-R.T @ t.reshape(3,)).reshape(3,)
    return T

def rt_from_world_T_cam(Twc):
    # Twc = [R^T | -R^T t] → (R,t)
    Rcam = Twc[:3,:3].T
    tcam = -Rcam @ Twc[:3,3]
    rvec, _ = cv2.Rodrigues(Rcam)
    return rvec.reshape(3,1), tcam.reshape(3,1)

# -----------------------------
# 메인: 두 점군 생성 → ICP 정합 → 시각화 (+옵션: cam2 포즈 업데이트)
# -----------------------------
def visualize_two_colored_depthmaps_icp(
    depth_path1, rgb_path1, K1, D1, R1, T1,
    depth_path2, rgb_path2, K2, D2, R2, T2,
    stride=4, do_update_pose=False
):
    # 1) 점군 생성(보드=월드 좌표)
    pts1, col1 = depth_to_colored_pointcloud_world(depth_path1, rgb_path1, K1, D1, R1, T1, stride)
    pts2, col2 = depth_to_colored_pointcloud_world(depth_path2, rgb_path2, K2, D2, R2, T2, stride)
    pcd1 = make_colored_pcd(pts1, col1)
    pcd2 = make_colored_pcd(pts2, col2)
    
    # 2) ICP (pcd2 → pcd1)
    T_icp, T_temp = icp_multiscale(pcd2, pcd1, voxel_scales=(0.02, 0.01, 0.005, 0.0001), max_iters=(120, 160, 180, 200))
    pcd2.transform(T_icp)

    # 3) 시각화
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries([axis, pcd1, pcd2])
    
    T_new = T_icp@T_temp
    return T_new
