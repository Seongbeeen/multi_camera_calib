import cv2
from glob import glob
import numpy as np
from natsort import natsorted as nat

class Checkerboard:
    def __init__(self, cam_idx, root_folder, save_found_cb, pattern_size, square_size, ransac):
        self.cam_idx = cam_idx
        self.save_found_cb = save_found_cb
        self.root_folder = root_folder
        self.num_cam = len(glob(f"{root_folder}/cam*"))
        self.rgb_paths = nat(glob(f"{root_folder}/cam{self.cam_idx}/rgb_*.jpg"))
        if len(self.rgb_paths)==0:
            self.rgb_paths = nat(glob(f"{root_folder}/cam{self.cam_idx}/frame_*.jpg")) # few cases
        self.depth_paths = nat(glob(f"{root_folder}/cam{self.cam_idx}/depth_*.png"))
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.ransac = ransac
        
        # NEED TO BE INITIALIZED
        self.K = None
        self.D = None
        self.R = None
        self.T = None
        self.rvecs_ = None
        self.tvecs_ = None
        self.rvecs = {}
        self.tvecs = {}
        
        self.obj_points = {}
        self.img_points = {}
        self.corner_idx = []
        self.pairs = {}
        
        self.collect_checkerboard_corners()
        print(f"[INFO] Camera {self.cam_idx} initialized")
    
    def find_checkerboard_corners(self, image, use_sb=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if use_sb and hasattr(cv2, "findChessboardCornersSB"):
            flags = (cv2.CALIB_CB_EXHAUSTIVE |
                    cv2.CALIB_CB_ACCURACY   |
                    cv2.CALIB_CB_LARGER)    # 필요시 | cv2.CALIB_CB_ACROSS_SCALE
            found, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags)
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            # 실패 프레임에서만 약한 블러/CLAHE 적용해 재시도 가능
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if not found:
            return False, None

        # 서브픽셀 정제
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
        win = (7, 7)  # 5~11 사이 튜닝
        corners = cv2.cornerSubPix(gray, corners, win, (-1,-1), criteria)
        if found:
            cv2.drawChessboardCorners(image, self.pattern_size, corners, found)
        return found, corners
    
    def generate_object_points(self):
        row, col = self.pattern_size
        objp = np.zeros((row*col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
            
    def collect_checkerboard_corners(self):
        objp = self.generate_object_points()
        for idx, rgb_path in enumerate(self.rgb_paths):
            img = cv2.imread(rgb_path)
            if img is None:
                continue
            self.img_size = img.shape[:2][::-1]
            found, corners = self.find_checkerboard_corners(img, use_sb=False)
            
            if found:
                self.obj_points[idx] = objp
                self.img_points[idx] = corners
                self.corner_idx.append(idx)
        print(f"[INFO] Camera {self.cam_idx}: Found corners from {self.corner_idx} chessboard(s)")
        
    def match_pairs(self, target_cam):
        target_idx = target_cam.cam_idx
        target_corners = target_cam.corner_idx
        
        if target_idx not in list(self.pairs.keys()):
            common = set(self.corner_idx) & set(target_corners)
            self.pairs[target_cam.cam_idx] = list(common)
        else:
            print("Already matched")
    
    def apply_RANSAC(self, obj_point, img_point):
        calib_param = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(obj_point, img_point, self.K, self.D, useExtrinsicGuess=False, iterationsCount=500, reprojectionError=2., confidence=0.99)
        
        ret, K_new, D_new, rvecs_new, tvecs_new = cv2.calibrateCamera([obj_point[inliers]], [img_point.squeeze()[inliers]], self.img_size, self.K, self.D, None, None, calib_param)
        return ret, K_new, D_new, rvecs_new, tvecs_new, inliers
        
    def calibrate_K_D_R_T(self):
        '''
        K, D: Unknown
        '''
        obj_points = []
        img_points = []
        for idx in self.corner_idx:
            obj_points.append(self.obj_points[idx])
            img_points.append(self.img_points[idx])
        
        
        ret, self.K, self.D, self.rvecs_, self.tvecs_ = cv2.calibrateCamera(obj_points, img_points, self.img_size, None, None)
    
    def calibrate_R_T(self):
        '''
        K, D: Known
        '''
        if self.K is not None or self.D is not None:
            self.calibrate_K_D_R_T() # for K, D
            self.rvecs_ = []
            self.tvecs_ = []
        for idx in self.corner_idx:
            und = cv2.undistortPoints(self.img_points[idx], self.K, self.D, P=self.K).reshape(-1,1,2)
            if self.ransac:
                ok, self.K, self.D, rvec, tvec, _ = self.apply_RANSAC(self.obj_points[idx], self.img_points[idx])
                self.rvecs_.append(rvec[0])
                self.tvecs_.append(tvec[0])
            else:
                ok, rvec, tvec = cv2.solvePnP(self.obj_points[idx], und, self.K, np.zeros((5,1)), flags=cv2.SOLVEPNP_ITERATIVE)
                self.rvecs_.append(rvec)
                self.tvecs_.append(tvec)
                
            if not ok:
                raise RuntimeError(f"solvePnP failed at frame {idx}")
    
    def vecs_to_dict(self):
        for cnt, idx in enumerate(self.corner_idx):
            self.rvecs[idx] = self.rvecs_[cnt]
            self.tvecs[idx] = self.tvecs_[cnt]
        
    def get_world_coord_mono(self, frame_idx=None, origin_cam=None):
        
        self.vecs_to_dict()
        if origin_cam is not None:
            rvecs1, tvecs1 = origin_cam.rvecs[frame_idx],   origin_cam.tvecs[frame_idx]
            rvecs2, tvecs2 = self.rvecs[frame_idx],         self.tvecs[frame_idx]
            
            R1, _ = cv2.Rodrigues(rvecs1)
            R2, _ = cv2.Rodrigues(rvecs2)
            t1 = np.asarray(tvecs1).reshape(3,1)
            t2 = np.asarray(tvecs2).reshape(3,1)
            
            self.R = R1 @ R2.T
            self.T = (t1 - self.R @ t2).reshape(3)
        else:
            self.R = np.eye(4)[:3,:3]
            self.T = np.eye(4)[:3,3]
        print(f"[INFO] R for Cam{self.cam_idx}:\n", self.R)
        print(f"[INFO] T for Cam{self.cam_idx}:\n", self.T)
            
        
    def get_world_coord_stereo(self, origin_cam=False):
        #### cv2.stereoCalibrate ####
        return 0
    
    def update_R_T(self, rvecs_new, tvecs_new):
        self.R = cv2.Rodrigues(rvecs_new)
        self.T = cv2.Rodrigues(tvecs_new)
        
def matching(cbs, num_cb):
    for i in range(num_cb):
        for j in range(i+1,num_cb):
            cbs[i].match_pairs(cbs[j])
            cbs[j].pairs[cbs[i].cam_idx] = cbs[i].pairs[cbs[j].cam_idx]
            print(f"[INFO] Cameras {(i, j)} have common detections at frames: {cbs[j].pairs[cbs[i].cam_idx]}")
            
def stereo_calibrate(objpoints, imgpoints1, imgpoints2, image_size, indices):
    # 단일 카메라 보정
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, image_size, None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, image_size, None, None)

    # 스테레오 보정
    flags = cv2.CALIB_FIX_INTRINSIC 
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2,
        image_size, criteria=criteria, flags=flags
    )
    
    dict_rvecs1, dict_tvecs1 = {}, {}
    dict_rvecs2, dict_tvecs2 = {}, {}
    
    for cnt, idx in enumerate(indices):
        dict_rvecs1[idx], dict_tvecs1[idx] = rvecs1[cnt], tvecs1[cnt]
        dict_rvecs2[idx], dict_tvecs2[idx] = rvecs2[cnt], tvecs2[cnt]
    
    return {
        "ret": ret,
        "cameraMatrix1": mtx1,
        "distCoeffs1": dist1,
        "cameraMatrix2": mtx2,
        "distCoeffs2": dist2,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rvecs1":dict_rvecs1,
        "tvecs1":dict_tvecs1,
        "rvecs2":dict_rvecs2,
        "tvecs2":dict_tvecs2,
    }