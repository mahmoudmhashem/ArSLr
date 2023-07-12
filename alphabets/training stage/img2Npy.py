import os
import gc
import cv2
import time
import queue
import threading
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from scipy.spatial.distance import euclidean
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
mp_hands = mp.solutions.hands

def worker_load(load, src_paths, images:queue.Queue, npy_paths:queue.Queue):
    for image_path, npy_path in src_paths:
        image = load(image_path)
        
        images.put(image)
        npy_paths.put(npy_path)

def worker_img2npy(img2npy, images:queue.Queue, npys:queue.Queue):
    while curr_task_idx<total_tasks:
        image = images.get()
        with mp_hands.Hands() as hands:
            _, npy = img2npy(image, hands)
        
        npys.put(npy)

def worker_save(save, npy_paths:queue.Queue, npys:queue.Queue):
    global curr_task_idx
    while curr_task_idx<total_tasks:
        curr_task_idx+=1
        
        path = npy_paths.get()
        npy = npys.get()
        if npy is not None:
            save(path, npy, allow_pickle=False)
        
        filename = os.path.basename(path)
        print(f"{curr_task_idx}/{total_tasks} with percentage {100 * curr_task_idx/ total_tasks:0.1f}, {filename} is done", end='\r')

def main(dataset_path):
    images = queue.Queue()
    npy_paths = queue.Queue()
    npys = queue.Queue()

    thread_load = threading.Thread(target=worker_load, args=(cv2.imread, get_image_paths(dataset_path), images, npy_paths))
    thread_img2npy = threading.Thread(target=worker_img2npy, args=(img2npy, images, npys))
    thread_save = threading.Thread(target=worker_save, args=(np.save, npy_paths, npys))
    

    global curr_task_idx, total_tasks

    total_tasks = 4247 # 9955
    curr_task_idx = 0

    thread_load.start()
    thread_img2npy.start()
    thread_save.start()
    

    thread_load.join()
    thread_img2npy.join()
    thread_save.join()


    

def rotate(matrix, theta, axis='z'):
    axis = axis.lower()
    if axis == 'x':
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif axis == 'z':
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    else:
        raise KeyError
    return (rotMatrix @ matrix.T).T

def img2npy(image, model):
    results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = np.zeros((21, 3))
    success = True

    try:
        mp_landmarks = results.multi_hand_world_landmarks[0]
        for i, landmark in enumerate(mp_landmarks.landmark):
            landmarks[i] = landmark.x, landmark.y, landmark.z


        # origin = landmarks[9]
        # scale = euclidean(*landmarks[9:11])
        
        # landmarks = (landmarks - origin)/scale

        
        # def rotVec():
        #     idx2pnk = landmarks[5] - landmarks[17]
        #     mdl2wrist = landmarks[0]
        #     rotVec = idx2pnk + mdl2wrist
        #     rotVec = idx2pnk
        #     return rotVec
        # tan = abs(rotVec()[1] / rotVec()[0])

        
        # landmarks = rotate(landmarks, -np.arctan(tan))
        # yflipped = landmarks[1, 1] - landmarks[0, 1] > 0
        # landmarks = rotate(landmarks, np.pi) if yflipped else landmarks

        # if results.multi_handedness[0].classification[0].label == "Right":
        #     landmarks[:, 0] = landmarks[:, 0]*-1


    except TypeError:
        success = False
    landmarks = landmarks if success else None
    return success, landmarks

def get_image_paths(dataset_path, num_signs=1000):
    for label in sorted(os.listdir(dataset_path))[:num_signs]:
        label_path = os.path.join(dataset_path, label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)

            pose_dir = os.path.join("npy_realworld", 'test', label)
            os.makedirs(pose_dir, exist_ok=True)
            pose_path = os.path.join(pose_dir, image[:-4])
            yield image_path, pose_path

if __name__ == '__main__':
    dataset_path = r"img/dataset/path"
    main(dataset_path)