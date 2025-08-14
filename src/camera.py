import cv2
import numpy as np
import pyrealsense2 as rs

class RealSenseCamera:
    def __init__(self, serial, height = 480, width= 640, fps =30, use_color=True,use_depth = True):
        self.serial = serial;
        self.height = height;
        self.width = width;
        self.fps = fps;
        self.use_depth = use_depth;
        self.use_color = use_color;
        self.is_opened = False
        print(f"[INFO] Initializing {self.serial}")
        
        # 디바이스 변수 얻기
        self.pipeline = rs.pipeline();
        self.config = rs.config();
        self.config.enable_device(self.serial);
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline);
        self.pipeline_wrapper = pipeline_wrapper;

        # 디바이스 연결 체크
        if(self.can_connect()):        
            pipeline_profile = self.config.resolve(pipeline_wrapper);
            device = pipeline_profile.get_device();

            # 디바이스 내 color 또는 depth 모듈 체크
            found_rgb = False
            found_depth = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'Stereo Module':
                    found_depth = True;
                elif s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True

            # color 또는 depth 모듈이 없다면 비활성화
            if not found_rgb:
                use_color = False; 
            if not found_depth:
                use_depth = False;    
       
            
            # 모듈이 있다면 
            if(self.use_depth):
                self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            if(self.use_color):
                self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps);
            
            if(self.can_connect() and (self.use_depth or self.use_color)):
                self.is_opened = True;
                self.pipeline.start(self.config);
                # align 객체 생성
                self.align = rs.align(rs.stream.color)
                if(self.use_color):
                    self.intr = self.pipeline.get_active_profile().get_stream(rs.stream.color)\
                    .as_video_stream_profile().get_intrinsics();

    def get_intrinsics(self):
        return self.intr;
    
    def get_extrinsics(self):
        if not self.is_opened or not (self.use_depth and self.use_color):
            return None
        
        try:
            depth_stream = self.pipeline.get_active_profile().get_stream(rs.stream.depth)
            color_stream = self.pipeline.get_active_profile().get_stream(rs.stream.color)
            return depth_stream.get_extrinsics_to(color_stream)
        except:
            return None

    def set(self,num,value):                      
        if(num == cv2.CAP_PROP_FRAME_HEIGHT):
            self.height= value;
        elif(num == cv2.CAP_PROP_FRAME_WIDTH):
            self.width= value;
        elif(num == cv2.CAP_PROP_FPS):
            self.fps= value;
        if(self.is_opened):
            if(self.use_depth):
                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps);
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps);
            if(self.can_connect()):
                self.pipeline.stop();
                self.pipeline.start(self.config);
                self.intr = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics();
    
    def get(self,num):                      
        if(num == cv2.CAP_PROP_FRAME_HEIGHT):
            return self.height;
        elif(num == cv2.CAP_PROP_FRAME_WIDTH):
            return  self.width;
        elif(num == cv2.CAP_PROP_FPS):
            return  self.fps;
    
    def can_connect(self):
        return self.config.can_resolve(self.pipeline_wrapper);
    
    def isOpened(self):
        return self.is_opened;
    
    def release(self):
        if(not self.pipeline is None):
            if(self.isOpened()):
                if(self.can_connect()):
                    self.pipeline.stop();
    
    def read_color_depth(self):
        try:
            #파이프 라인으로부터 프레임 얻기 (100ms안에)
            frames = self.pipeline.wait_for_frames(100)
            color_frame, depth_frame = None, None;
            color_image, depth_image = None, None;

            if(self.use_color):
                color_frame = frames.get_color_frame()
            if(self.use_depth):
                depth_frame = frames.get_depth_frame()
                if(not depth_frame):
                    return False,[None, None]
            
            if(not color_frame is None):
                color_image = np.asanyarray(color_frame.get_data())
            if(not depth_frame is None):
                depth_image = np.asanyarray(depth_frame.get_data())
            
            return True if((not color_image is None) or (not depth_image is None)) else False,[color_image, depth_image];
    
        except:
            return False,[None, None]
        
    def read(self):
        try:
            #파이프 라인으로부터 프레임 얻기 (100ms안에)
            frames = self.pipeline.wait_for_frames(100)
            if(self.use_color):
                color_frame = frames.get_color_frame();
                if (not color_frame):
                    return False, None
                color_image = np.asanyarray(color_frame.get_data());
            else:
                color_image = np.zeros((self.height,self.width,3),dtype=np.uint8);
            return True, color_image;
        except:
            return False,None

    def read_color_depth_aligned(self):
        if not self.pipeline:
            return False, [None, None, None, None]
        try:
            frames = self.pipeline.wait_for_frames(1000)
            if not frames:
                return False, [None, None, None, None]
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame() if self.use_color else None
            depth_frame = aligned_frames.get_depth_frame() if self.use_depth else None
            color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            color_ts = color_frame.get_timestamp() if color_frame else None
            depth_ts = depth_frame.get_timestamp() if depth_frame else None
            return True if color_image is not None or depth_image is not None else False, \
                   [color_image, depth_image, color_ts, depth_ts]
        except Exception as e:
            print(f"[{self.serial}] Exception: {e}")
            return False, [None, None, None, None]