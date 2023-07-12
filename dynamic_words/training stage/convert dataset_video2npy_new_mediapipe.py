import os
import cv2
import queue
import threading
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import multiprocessing


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
pose_model_path = '../assests/pose_landmarker_heavy.task'
hand_model_path = '../assests/hand_landmarker.task'

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO)

# Create a hand landmarker instance with the image mode:
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

# poselandmarker = PoseLandmarker.create_from_options(pose_options)
# handlandmarker = HandLandmarker.create_from_options(hand_options)

save_Q = queue.Queue()
# pose_Q = queue.Queue()
# hand_Q = queue.Queue()


def worker(func, tasks: queue.Queue):
    global pbr
    curr_task_idx = 0
    while True:
        task = tasks.get()
        func(*task)
        # pbr.update()
        filename = os.path.basename(task[0])
        print(f"{curr_task_idx}/{total_tasks} with percentage {100 * curr_task_idx/ total_tasks:0.1f}, {filename} is done", end='\r')
        curr_task_idx+=1


def convertNsave(videopath, dst_path):
    video_path, video_npy = video2npy(videopath)
    save_Q.put((video_path, dst_path, video_npy))


def task_done(video_path, dst_path):
    video = os.path.basename(video_path)
    label = os.path.dirname(video_path).split('\\')[-1]

    video_npy_filename = video[:-4] + ".npy"
    video_npy_dir = os.path.join(dst_path, label)
    video_npy_path = os.path.join(video_npy_dir, video_npy_filename)

    return os.path.exists(video_npy_path)


def main(dataset_path, dst_path):
    video_paths = get_video_paths(dataset_path, 20)

    print('find tasks not done yet ...')
    tasks = []
    for video_path in video_paths:
        if not task_done(video_path, dst_path):
            tasks.append(video_path)

    global pbr, total_tasks
    total_tasks = len(tasks)
    # pbr = tqdm(total=total_tasks)

    thread = threading.Thread(target=worker, args=(save_video_npy, save_Q))
    thread.daemon = True

    # pool = multiprocessing.Pool(8)
    # results = pool.imap_unordered(video2npy, video_paths)
    # pool.close()
    # thread.start()

    thread.start()
    with ThreadPoolExecutor(20) as executer:
        executer.map(lambda task: convertNsave(task, dst_path), tasks)
    
    # for task in tasks:
    #     convertNsave(task, dst_path)

    # map(lambda task: convertNsave(task, dst_path), tasks)
    # convertNsave(tasks[0], dst_path)
    # with tqdm(total=len(video_paths)) as pbr:
    #     for video_path, video_npy in results:
    #         q.put((video_path, video_npy))
    #         pbr.update(1)


def mediapipe_detection(numpy_image, frame_timestamp_ms, poselandmarker, handlandmarker): # we assume that input image is in BGR format

    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    numpy_image = cv2.flip(numpy_image, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    # with ThreadPoolExecutor(2) as executer:
    #     pose_future = executer.submit(poselandmarker.detect_for_video, mp_image, frame_timestamp_ms)
    #     hand_future = executer.submit(handlandmarker.detect_for_video, mp_image, frame_timestamp_ms)

    # pose_landmarker_result = pose_future.result()
    # hand_landmarker_result = hand_future.result()

    pose_landmarker_result = poselandmarker.detect_for_video(mp_image, frame_timestamp_ms)
    hand_landmarker_result = handlandmarker.detect_for_video(mp_image, frame_timestamp_ms)

    return pose_landmarker_result, hand_landmarker_result


def extract_keypoints(pose_landmarker_result, hand_landmarker_result):
    
    hands = hand_landmarker_result.hand_world_landmarks
    hands_npy = np.zeros(21*2*2)
    i=0
    for hand in hands:
        for landmark in hand:
            hands_npy[i:i+2] = [landmark.x, landmark.y]
            i+=2

    bodies = pose_landmarker_result.pose_world_landmarks
    body_npy = np.zeros(33*2)
    i=0
    for body in bodies:
        for landmark in body:
            body_npy[i:i+2] = [landmark.x, landmark.y]
            i+=2

    landmarks = np.concatenate([hands_npy, body_npy], axis=0)
    return landmarks


def frame2npy(image, frame_timestamp_ms, poselandmarker, handlandmarker):
    pose_results, hands_results = mediapipe_detection(image, frame_timestamp_ms, poselandmarker, handlandmarker)
    landmarks = extract_keypoints(pose_results, hands_results)
    return landmarks


def video2npy(video_path: str):
    cap = cv2.VideoCapture(video_path)
    video_npy_ls = []
    with PoseLandmarker.create_from_options(pose_options) as poselandmarker:
        with HandLandmarker.create_from_options(hand_options) as handlandmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                frame_npy = frame2npy(frame, frame_timestamp_ms, poselandmarker, handlandmarker)
                video_npy_ls.append(frame_npy)
            cap.release()

    video_npy_arr = np.array(video_npy_ls)

    return video_path, video_npy_arr


def save_video_npy(video_path, dst_path, video_npy):
    video = os.path.basename(video_path)
    label = os.path.dirname(video_path).split('\\')[-1]

    video_npy_filename = video[:-4] + ".npy"
    video_npy_dir = os.path.join(dst_path, label)
    video_npy_path = os.path.join(video_npy_dir, video_npy_filename)

    os.makedirs(video_npy_dir, exist_ok=True)
    np.save(video_npy_path, video_npy, allow_pickle=False)


def get_video_paths(dataset_path, num_signs=5):
    # video_paths = []
    for label in sorted(os.listdir(dataset_path))[:num_signs]:
        label_path = os.path.join(dataset_path, label)
        for video in os.listdir(label_path):
            video_path = os.path.join(label_path, video)
            # video_paths.append(video_path)
            yield video_path
    # return video_paths


if __name__ == '__main__':
    data_path = "write/dataset/path/here"
    dst_path = "write/destination/path/here"
    main(data_path, dst_path)