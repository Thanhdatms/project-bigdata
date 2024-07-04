import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def getOpticalFlow(video):
    """Calculate dense optical flow of input video using GPU.
    Args:
        video: the input video with shape of [frames, height, width, channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames, height, width, channel]
        flows_y: the optical flow at y-axis, with the shape of [frames, height, width, channel]
    """
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video)-1):
        # calculate optical flow between each pair of frames using GPU
        flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
        d_frame0 = cp.asarray(gray_video[i])
        d_frame1 = cp.asarray(gray_video[i+1])
        flow = flow.calc(d_frame0, d_frame1, None)
        flow = cp.asnumpy(flow)

        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Add into list
        flows.append(flow)

    # Padding the last frame as empty array
    flows.append(np.zeros((224, 224, 2)))

    return np.array(flows, dtype=np.float32)

def Video2Npy(file_path, resize=(224, 224)):
    """Load video and transfer it into .npy format using GPU
    Args:
        file_path: the path of video file
        resize: the target resize dimension
    Returns:
        result: the resulting array with frames and optical flow
    """
    cap = cv2.VideoCapture(file_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame / 255.0  # normalize to [0, 1]
            frames.append(frame)
    finally:
        cap.release()
        if not frames:
            print(f"Error: No valid frames extracted from {file_path}")
            return None
        frames = np.array(frames)

    # Get the optical flow of video
    flows = getOpticalFlow(frames)
    result = np.zeros((len(flows), 224, 224, 5))
    result[..., :3] = frames
    result[..., 3:] = flows

    return result

def Save2Npy(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Get dest
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        try:
            data = Video2Npy(file_path=video_path, resize=(224, 224))
            if data is not None:
                data = np.uint8(data)
                # Save as .npy file
                np.save(save_path, data)
            else:
                print(f"Skipping {video_path} due to errors.")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    return None

source_path = "/home/islabworker3/tienvo/dataset_video"
target_path = "/home/islabworker3/tienvo/dangdat/data-video/source"

for f1 in ['train', 'val', 'test']:
    for f2 in ['normal', 'restricted']:
        path1 = os.path.join(source_path, f1, f2)
        path2 = os.path.join(target_path, f1, f2)
        Save2Npy(file_dir=path1, save_dir=path2)