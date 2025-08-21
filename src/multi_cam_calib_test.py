import cv2
import numpy as np
import os
import time
from collections import defaultdict
from glob import glob
import matplotlib.pyplot as plt
import open3d as o3d
import argparse

from utils import *
from icp import *
from checkerboard import *
    
    
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
    parser.add_argument(
        "--z_lim", default=(0.05, 1.2),
        help="m"
    )
    parser.add_argument(
        "--solvePNP", default=False,
        help="m"
    )
    parser.add_argument(
        "--ransac", default=False,
    )
    parser.add_argument(
        "--origin_cam_idx", default=0,
    )
    args = parser.parse_args()
    print(f"[INFO] Found calibration target from {args.root_folder}")
    
    num_cams = len(glob(f"{args.root_folder}/cam*"))
    depth_scale = 0.001
    z_min, z_max = args.z_lim
    
    cbs = []
    for i in range(num_cams):
        cbs.append(Checkerboard(i, args.root_folder, False, args.pattern_size, args.square_size, ransac=args.ransac))
    
    matching(cbs, num_cams)
    # *for single frame calibration* #
    if not args.stereo:
        for cb in cbs:
            cb.calibrate_K_D_R_T()
            if args.solvePNP:
                cb.calibrate_R_T()
    else:
        # *Stereo Calibration* #
        0
        
    # *Visualize point cloud* #
    origin_cam = cbs[args.origin_cam_idx]
    pcds = []
    for i in range(num_cams-1):
        cam0, cam1 = origin_cam, cbs[i+1]
        print(f"Calibrate camera {cam0.cam_idx} and {cam1.cam_idx}")

        matches = cam0.pairs[cam1.cam_idx]
        match_idx = 0
            # *for single frame calibration* #
        if not args.stereo:
            frame_idx = matches[match_idx]
            cam0.get_world_coord_mono(frame_idx, None)
            cam1.get_world_coord_mono(frame_idx, origin_cam)
        else:
            # *Stereo Calibration* #
            0
        
        rgb_paths1,  depth_paths1= cam0.rgb_paths[frame_idx], cam0.depth_paths[frame_idx]
        rgb_paths2,  depth_paths2= cam1.rgb_paths[frame_idx], cam1.depth_paths[frame_idx]
        
        K1, D1, rvecs1, tvecs1 = cam0.K, cam0.D, cam0.rvecs[frame_idx], cam0.tvecs[frame_idx]
        K2, D2, rvecs2, tvecs2 = cam1.K, cam1.D, cam1.rvecs[frame_idx], cam1.tvecs[frame_idx]
        
        visualize_pcd_world(
            depth_paths1, rgb_paths1, K1, D1, cam0.R, cam0.T,
            depth_paths2, rgb_paths2, K2, D2, cam1.R, cam1.T,
            depth_scale, z_min, z_max, stride=4, ret=False
        )
        T_icp = multiscale_icp(
            depth_paths1, rgb_paths1, K1, D1, cam0.R, cam0.T,
            depth_paths2, rgb_paths2, K2, D2, cam1.R, cam1.T,
            depth_scale, z_min, z_max, stride=4
        )
        
        T_old = np.eye(4)
        T_old[:3,:3] = cam1.R
        T_old[:3,3] = cam1.T
        
        T_new = T_icp@T_old
        cam1.R = T_new[:3,:3]
        cam1.T = T_new[:3,3]
        
        pcd0, pcd1 = visualize_pcd_world(
            depth_paths1, rgb_paths1, K1, D1, cam0.R, cam0.T,
            depth_paths2, rgb_paths2, K2, D2, cam1.R, cam1.T,
            depth_scale, z_min, z_max, stride=4, ret=True
        )
        pcds.append(pcd0)
        pcds.append(pcd1)
        
    o3d.visualization.draw_geometries(pcds)
    