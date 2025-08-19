import os
import cv2
import numpy as np
import open3d as o3d


def find_checkerboard_corners(image, root_folder, pattern_size=(9, 6), cam_id=0, idx=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    gray_blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    found, corners = cv2.findChessboardCorners(gray_blurred, pattern_size, None)
    if found:
        cv2.drawChessboardCorners(image, pattern_size, corners, found)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title(f"Checkerboard Detection for cam{cam_id}-{idx}")
    # plt.axis("off")
    # plt.savefig(f"{root_folder}/cam{cam_id}/check_{idx:03d}.png")
    return found

def find_common_checkerboard_indices(root_folder, num_cams=3, pattern_size=(10, 6)):
    """
    각 카메라 폴더에서 체커보드가 인식된 이미지 인덱스를 비교하여,
    두 대 이상의 카메라에서 동시에 인식된 인덱스를 반환.

    Returns:
        dict: { (0,1): [indices], (1,2): [indices], (0,2): [indices] }
    """
    checkerboard_found = defaultdict(set)  # cam_id -> set of indices

    for cam_id in range(num_cams):
        folder = os.path.join(root_folder, f"cam{cam_id}")
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        
        for idx, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            if find_checkerboard_corners(img, root_folder, pattern_size, cam_id, idx):
                checkerboard_found[cam_id].add(idx)

    pairwise_matches = {}
    for i in range(num_cams):
        for j in range(i + 1, num_cams):
            common = checkerboard_found[i].intersection(checkerboard_found[j])
            pairwise_matches[(i, j)] = sorted(common)

    return pairwise_matches

def generate_object_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp  # shape: (N, 3)


    
def load_corners_for_pair(root_folder, cam1, cam2, indices, pattern_size):
    objpoints = []
    imgpoints1 = []
    imgpoints2 = []
    
    for idx in indices:
        fname1 = os.path.join(root_folder, f"cam{cam1}", f"rgb_{str(idx).zfill(3)}.jpg")
        fname2 = os.path.join(root_folder, f"cam{cam2}", f"rgb_{str(idx).zfill(3)}.jpg")
        img1 = cv2.imread(fname1)
        img2 = cv2.imread(fname2)
        if img1 is None or img2 is None:
            continue
        
        gray1, found1, corners1 = process_gray(img1)
        gray2, found2, corners2 = process_gray(img2)

        if found1 and found2:
            objp = generate_object_points(pattern_size, square_size=0.025)  # ← 단위 맞게 수정
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria)

            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

    return objpoints, imgpoints1, imgpoints2, gray1.shape[::-1]  # image_size

def load_corners_for_pair_known(dict_imgpoints1,dict_imgpoints2, indices, pattern_size, img_size):
    objpoints = {}
    imgpoints1 = {}
    imgpoints2 = {}
    print("indices",indices)
    for idx in indices:
        objp = generate_object_points(pattern_size, square_size=0.025)  # ← 단위 맞게 수정
        corners1 = dict_imgpoints1[idx]
        corners2 = dict_imgpoints2[idx]
        objpoints[idx] = objp
        imgpoints1[idx] = corners1
        imgpoints2[idx] = corners2
    return objpoints, imgpoints1, imgpoints2, img_size


def rodrigues(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def camera_center_in_world(R, t):
    # C = -R^T t
    return (-R.T @ t.reshape(3,1)).reshape(3)

def cam_pose_4x4_in_world(R, t):
    # world<-cam 변환: [R^T | -R^T t]
    T = np.eye(4)
    T[:3,:3] = R.T
    T[:3, 3] = camera_center_in_world(R, t)
    return T

def make_axis(size=0.05):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

# 카메라 프러스텀(대략적) 메시 생성 (시각화용)
def make_camera_frustum(K, img_size, scale=0.1, color=(0.1,0.1,0.1)):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    w, h = img_size
    # z=1 평면의 4코너를 카메라좌표로 역투영 → z=scale로 스케일
    corners_px = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    rays = []
    for u,v in corners_px:
        x = (u - cx)/fx
        y = (v - cy)/fy
        rays.append([x*scale, y*scale, scale])
    rays = np.array(rays)
    # 원점 + 4코너로 라인세트 구성
    points = np.vstack([np.zeros((1,3)), rays])  # 0: origin
    lines  = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    colors = [color for _ in lines]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return ls

# ===============================
# (A) K, D가 이미 있는 경우: 프레임별 solvePnP
# ===============================
def pose_from_pnp_per_frame(objpoints, imgpoints, K, D):
    """각 프레임에 대해 solvePnP로 (R,t) 추정.
       objpoints: [N_frames x (N_pts,3)]
       imgpoints: [N_frames x (N_pts,1,2)]
    """
    rvecs, tvecs = [], []
    for i in range(len(objpoints)):
        # 왜곡 보정한 픽셀 좌표 사용 권장
        und = cv2.undistortPoints(imgpoints[i], K, D, P=K).reshape(-1,1,2)
        ok, rvec, tvec = cv2.solvePnP(objpoints[i], und, K, np.zeros((5,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError(f"solvePnP failed at frame {i}")
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs

# ===============================
# (B) K, D가 없는 경우: 단일카메라 보정 → 포즈
# ===============================
def calibrate_single_and_poses(objpoints, imgpoints, image_size):
    """여러 프레임으로 K,D,R,t 동시 추정 (calibrateCamera)."""
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    # rvecs,tvecs는 각 프레임의 월드(체커보드)→카메라 변환
    return K, D, rvecs, tvecs, ret

# ===============================
# 시각화: 체커보드(원점) + 두 카메라 포즈
# ===============================
def visualize_board_and_cameras(
    K1, D1, K2, D2,
    rvecs1, tvecs1, rvecs2, tvecs2,
    frame_idx, image_size
):
    # 선택 프레임의 포즈
    R1 = rodrigues(rvecs1[frame_idx]); t1 = tvecs1[frame_idx]
    R2 = rodrigues(rvecs2[frame_idx]); t2 = tvecs2[frame_idx]

    # 보드(원점) 좌표축
    axis_board = make_axis(size=0.05)

    # 각 카메라의 월드(보드) 좌표계에서의 포즈(프레임)
    T_cam1 = cam_pose_4x4_in_world(R1, t1)
    T_cam2 = cam_pose_4x4_in_world(R2, t2)

    # 카메라 좌표축 메시
    axis_cam1 = make_axis(size=0.05).transform(T_cam1)
    axis_cam2 = make_axis(size=0.05).transform(T_cam2)

    # 카메라 프러스텀(대략)
    fr1 = make_camera_frustum(K1, image_size, scale=0.15, color=(0,0,1))
    fr2 = make_camera_frustum(K2, image_size, scale=0.15, color=(1,0,0))
    fr1.transform(T_cam1)
    fr2.transform(T_cam2)

    # (선택) 체커보드 평면 메시(얕은 사각형)
    cols, rows = pattern_size
    board_w = (cols-1)*square_size
    board_h = (rows-1)*square_size
    plane = o3d.geometry.TriangleMesh.create_box(width=board_w, height=board_h, depth=1e-4)
    plane.translate([0,0, -5e-5])  # Z=0 평면에 살짝 겹치게
    plane.paint_uniform_color([0.85,0.85,0.85])

    o3d.visualization.draw_geometries([axis_board, plane, axis_cam1, axis_cam2, fr1, fr2])