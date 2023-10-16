import pyrealsense2 as rs
import numpy as np
import cv2
import time
import sys
import os
from sys import platform
import torch
from torch import nn
import argparse





try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose/python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            #sys.path.append('./openpose/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # # Flags
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../openpose/models/"

    
    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    
    
    

    # hardware reset

    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
            dev.hardware_reset()
            
    time.sleep(5)
    print("reset done")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))
                
            
            # Process Image
            datum = op.Datum()
            imageToProcess = color_image
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            
            
            if datum.poseKeypoints is not None:
                keypoints = np.array(datum.poseKeypoints)
            
            
            
            images = np.hstack((datum.cvOutputData, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        pipeline.stop()
            
    
except Exception as e:
    print(e.with_traceback())
    sys.exit(-1)


# 5 lshoulder
# 6 rshoulder
# 7 lelbow
# 8 relbow
# 9 lwrist
# 10 rwrist
# 11 lhip
# 13 lknee

# 21 lheel

# robot joint pos: [lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist]


def get_keypoint_reward(keypoints, robot_joint_pos):
    
    
    AVG_SHOULDER_HEIGHT = 0.3
    AVG_SHOULDER_TO_KNEE_HEIGHT = 0.225
    AVG_SHOULDER_TO_HIP_HEIGHT = 0.15
    
    # scale the keypoints to meters based on the height of robot dog
    
    if keypoints[21] is not None:
        dots_per_m = AVG_SHOULDER_HEIGHT/(keypoints[0][1]-keypoints[21][1]) #requires the heel to be visible
    elif keypoints[13] is not None:
        dots_per_m = AVG_SHOULDER_TO_KNEE_HEIGHT/(keypoints[0][1]-keypoints[13][1])
    elif keypoints[11] is not None:
        dots_per_m = AVG_SHOULDER_TO_HIP_HEIGHT/(keypoints[0][1]-keypoints[11][1])
    
    keypoints = keypoints * dots_per_m
    
    BASE_REWARD = 100 # reward constant (can be tuned)
    
    l_elbow_pos = keypoints[7]
    r_elbow_pos = keypoints[8]
    
    l_shoulder_pos = keypoints[5]
    r_shoulder_pos = keypoints[6]
    
    l_wrist_pos = keypoints[9]
    r_wrist_pos = keypoints[10]
    
    l_elbow_height_target = l_elbow_pos[1] - l_shoulder_pos[1]
    r_elbow_height_target = r_elbow_pos[1] - r_shoulder_pos[1]
    
    l_wrist_height_target = l_wrist_pos[1] - l_shoulder_pos[1]
    r_wrist_height_target = r_wrist_pos[1] - r_shoulder_pos[1]
    
    l_elbow_reward = np.clip(BASE_REWARD-(robot_joint_pos[2] - l_elbow_height_target), 0, BASE_REWARD)
    r_elbow_reward = np.clip(BASE_REWARD-(robot_joint_pos[3] - r_elbow_height_target), 0, BASE_REWARD)
    
    l_wrist_reward = np.clip(BASE_REWARD-(robot_joint_pos[4] - l_wrist_height_target), 0, BASE_REWARD)
    r_wrist_reward = np.clip(BASE_REWARD-(robot_joint_pos[5] - r_wrist_height_target), 0, BASE_REWARD)
    
    reward = l_elbow_reward + r_elbow_reward + l_wrist_reward + r_wrist_reward
    
    return reward