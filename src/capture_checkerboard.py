import argparse
import pyrealsense2 as rs
import time
import cv2
import os

from camera import RealSenseCamera

def list_connected_devices():
    print("[INFO] Detecting and initializing cameras...")
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        serials.append(dev.get_info(rs.camera_info.serial_number))
    cams = [RealSenseCamera(serial, use_color=True, use_depth=True) for serial in serials]
    print(f"[INFO] {len(cams)} camera(s) detected.")
    return cams

def acquire_multi_camera_data(cameras, only='both'):
    """
    cameras: RealSenseCamera 객체 리스트
    only: 'rgb', 'depth', 'both' 중 선택
    return: 각 카메라별 dict {'rgb':..., 'depth':..., 'rgb_ts':..., 'depth_ts':...}
    """
    results = []
    for cam in cameras:
        ok, [rgb, depth, rgb_ts, depth_ts] = cam.read_color_depth_aligned()
        d = {}
        if only == 'rgb':
            d['rgb'] = rgb
            d['rgb_ts'] = rgb_ts
            d['depth'] = None
            d['depth_ts'] = None
        elif only == 'depth':
            d['rgb'] = None
            d['rgb_ts'] = None
            d['depth'] = depth
            d['depth_ts'] = depth_ts
        else:
            d['rgb'] = rgb
            d['depth'] = depth
            d['rgb_ts'] = rgb_ts
            d['depth_ts'] = depth_ts
        results.append(d)
    return results

def save_multi_camera_realtime(cameras, root_folder, display_width=320):
    """
    여러 카메라의 RGB 영상 실시간 표시 및 스페이스바로 RGB/Depth 저장.

    Args:
        cameras: acquire_multi_camera_data()에 전달할 카메라 정보
        root_folder (str): 저장 루트 폴더
        display_width (int): 시각화시 리사이즈 너비
    """
    os.makedirs(root_folder, exist_ok=True)
    frame_idx = {}
    print("[INFO] Beginning data capture...")
    for i in range(len(cameras)):
        cam_dir = os.path.join(root_folder, f"cam{i}")
        os.makedirs(cam_dir, exist_ok=True)
        frame_idx[i] = 0

    saved_frame = 0    
    while True:
        data_list = acquire_multi_camera_data(cameras)

        for i, data in enumerate(data_list):
            rgb = data.get('rgb')
            if rgb is not None:
                h, w = rgb.shape[:2]
                scale = display_width / w
                resized_rgb = cv2.resize(rgb, (display_width, int(h * scale)))
                win_name = f"Camera {i+1} RGB"
                cv2.imshow(win_name, resized_rgb)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        if key == 32:  # 스페이스바로 저장
            for i, data in enumerate(data_list):
                rgb = data.get('rgb')
                depth = data.get('depth')
                idx = frame_idx[i]
                cam_dir = os.path.join(root_folder, f"cam{i}")
                if rgb is not None:
                    rgb_path = os.path.join(cam_dir, f"rgb_{idx:03d}.jpg")
                    cv2.imwrite(rgb_path, rgb)
                if depth is not None:
                    depth_path = os.path.join(cam_dir, f"depth_{idx:03d}.png")
                    if depth.dtype != 'uint16':
                        depth = depth.astype('uint16')
                    cv2.imwrite(depth_path, depth)
                frame_idx[i] += 1
            print(f"[INFO] Frame {saved_frame:2d} saved.")
            saved_frame += 1

    cv2.destroyAllWindows()
    print(f"[INFO] Data capture complete. Files saved to: {root_folder}")

def save_multi_camera_realtime_all(cameras, display_width=320, root_folder=None, save_interval=1.0):
    """
    여러 카메라의 RGB 영상 실시간 표시 및 저장.
    
    Args:
        cameras: acquire_multi_camera_data()에 전달할 카메라 정보
        display_width (int): 표시 영상의 너비 (비율 유지 리사이즈)
        root_folder (str or None): 이미지 저장 루트 폴더 (None이면 저장 안 함)
        save_interval (float): 이미지 저장 간격 (초)
    """
    last_save_time = time.time()
    num_cams = None
    frame_idx = {}

    if root_folder is not None:
        os.makedirs(root_folder, exist_ok=True)

    while True:
        data_list = acquire_multi_camera_data(cameras)

        if num_cams is None:
            num_cams = len(data_list)
            if root_folder is not None:
                for i in range(num_cams):
                    cam_folder = os.path.join(root_folder, f"cam{i}")
                    os.makedirs(cam_folder, exist_ok=True)
                    frame_idx[i] = 0
        
        for i, data in enumerate(data_list):
            rgb = data.get('rgb')
            if rgb is not None:
                h, w = rgb.shape[:2]
                scale = display_width / w
                resized_rgb = cv2.resize(rgb, (display_width, int(h * scale)))
                win_name = f"Camera {i+1} RGB"
                cv2.imshow(win_name, resized_rgb)
        
        current_time = time.time()
        if root_folder is not None and (current_time - last_save_time >= save_interval):
            for i, data in enumerate(data_list):
                rgb = data.get('rgb')
                if rgb is not None:
                    filename = f"frame_{frame_idx[i]:03d}.jpg"
                    save_path = os.path.join(root_folder, f"cam{i}", filename)
                    cv2.imwrite(save_path, rgb)
                    frame_idx[i] += 1
            last_save_time = current_time

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    print("[INFO] Starting checkerboard data acquisition...")
    
    parser = argparse.ArgumentParser(description="Multi-camera checkerboard data acquisition")
    parser.add_argument(
        "--root_folder", type=str, default="../checkerboard/0812_test", required=False,
        help="Directory to save captured RGB/Depth data"
    )
    parser.add_argument(
        "--display_width", type=int, default=480,
        help="Width for display window (maintains aspect ratio)"
    )
    args = parser.parse_args()
    
    cams = list_connected_devices()    
    
    if len(cams)!=0:
        save_multi_camera_realtime(cams, root_folder=args.root_folder, display_width=args.display_width)
    else:
        print("[WARNING] No cameras detected. Please check the connections.")