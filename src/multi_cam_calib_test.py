import cv2
import numpy as np
import os
import glob
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import open3d as o3d
import argparse

from utils import *
from icp import *
from checkerboard import Checkerboard


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

def matching(cbs, num_cb):
    for i in range(num_cb):
        for j in range(i+1,num_cb):
            cbs[i].match_pairs(cbs[j])
            cbs[j].pairs[cbs[i].cam_idx] = cbs[i].pairs[cbs[j].cam_idx]
            print(f"[INFO] Cameras {(i, j)} have common detections at frames: {cbs[j].pairs[cbs[i].cam_idx]}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-camera checkerboard data acquisition")
    parser.add_argument(
        # "--root_folder", type=str, default="../../2025_point/frames/0801_checkerboard/", required=False,
        "--root_folder", type=str, default="../test_0814_2/", required=False,
        help="Directory to save captured RGB/Depth data"
    )
    parser.add_argument(
        "--stereo", type=bool, default=False, required=False,
        help="Directory to save captured RGB/Depth data"
    )
    parser.add_argument(
        "--display_width", type=int, default=480,
        help="Width for display window (maintains aspect ratio)"
    )
    parser.add_argument(
        "--pattern_size", default=(10,6),
    )
    parser.add_argument(
        "--square_size", default=0.025,
        help="m"
    )
    args = parser.parse_args()
    print(f"[INFO] Found calibration target from {args.root_folder}")
    num_cams = len(glob.glob(f"{args.root_folder}/cam*"))
    
    cbs = []
    for i in range(num_cams):
        cbs.append(Checkerboard(i, args.root_folder, False, args.pattern_size, args.square_size, ransac=False))
    
    matching(cbs, num_cams)
    # *for single frame calibration* #
    if not args.stereo:
        for cb in cbs:
            cb.calibrate_K_D_R_T()
            cb.calibrate_R_T()
    else:
        # *Stereo Calibration* #
        0
        
    # *Visualize point cloud* #
    origin_cam = cbs[0]
    pcds = []
    for i in range(num_cams-1):
        cam0, cam1 = cbs[0], cbs[i+1]
        print(f"Calibrate camera {cam0.cam_idx} and {cam1.cam_idx}")

        matches = cam0.pairs[cam1.cam_idx]
        match_idx = 0
            # *for single frame calibration* #
        if not args.stereo:
            frame_idx = matches[match_idx]
            cam0.get_world_coord_mono(frame_idx, None)
            cam1.get_world_coord_mono(frame_idx, cam0)
        else:
            # *Stereo Calibration* #
            0
        
        rgb_paths1,  depth_paths1= cam0.rgb_paths[frame_idx], cam0.depth_paths[frame_idx]
        rgb_paths2,  depth_paths2= cam1.rgb_paths[frame_idx], cam1.depth_paths[frame_idx]
        
        K1, D1, rvecs1, tvecs1 = cam0.K, cam0.D, cam0.rvecs[frame_idx], cam0.tvecs[frame_idx]
        K2, D2, rvecs2, tvecs2 = cam1.K, cam1.D, cam1.rvecs[frame_idx], cam1.tvecs[frame_idx]
        
        visualize_two_colored_depthmaps_in_world(
            depth_paths1, rgb_paths1, K1, D1, cam0.R, cam0.T,
            depth_paths2, rgb_paths2, K2, D2, cam1.R, cam1.T,
            stride=4, ret=False
        )
        T_icp = visualize_two_colored_depthmaps_icp(
            depth_paths1, rgb_paths1, K1, D1, cam0.R, cam0.T,
            depth_paths2, rgb_paths2, K2, D2, cam1.R, cam1.T,
            stride=4, do_update_pose=True
        )
        
        T_old = np.eye(4)
        T_old[:3,:3] = cam1.R
        T_old[:3,3] = cam1.T
        
        T_new = T_icp@T_old
        cam1.R = T_new[:3,:3]
        cam1.T = T_new[:3,3]
        
        pcd0, pcd1 = visualize_two_colored_depthmaps_in_world(
            depth_paths1, rgb_paths1, K1, D1, cam0.R, cam0.T,
            depth_paths2, rgb_paths2, K2, D2, cam1.R, cam1.T,
            stride=4, ret=True
        )
        pcds.append(pcd0)
        pcds.append(pcd1)
        
    o3d.visualization.draw_geometries(pcds)
    
        
        
        