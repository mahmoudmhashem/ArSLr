import os
import queue
import threading
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as LR

def worker_load(load, paths_nan:list, paths_not_nan:queue.Queue, arrs_nan:queue.Queue):
    for path in paths_nan:
        arr = load(path) # , allow_pickle=True)

        arrs_nan.put(arr)
        paths_not_nan.put(path)

def worker_fillna(fillna, arrs_nan:queue.Queue, arrs_not_nan:queue.Queue):
    while curr_task_idx<total_tasks:
        arr_nan = arrs_nan.get()
        fillna(arr_nan)
        arrs_not_nan.put(arr_nan)

def worker_save(save, paths_not_nan:queue.Queue, dst_path, arrs_not_nan:queue.Queue):
    global curr_task_idx
    while curr_task_idx<total_tasks:
        curr_task_idx+=1
        path = paths_not_nan.get()
        arr = arrs_not_nan.get()
        save(path, dst_path, arr)
        # print("type:", type(arr), "shape:", arr.shape)
        filename = os.path.basename(path)
        print(f"{curr_task_idx}/{total_tasks} with percentage {100 * curr_task_idx/ total_tasks:0.1f}, {filename} is done", end='\r')
    

def main(dataset_path, dst_path, num_signs):
    video_paths = get_video_paths(dataset_path, num_signs)
    paths_not_nan = queue.Queue()
    arrs_nan = queue.Queue()
    arrs_not_nan = queue.Queue()
    
    thread_load = threading.Thread(target=worker_load, args=(np.load, video_paths, paths_not_nan, arrs_nan))
    thread_fillna = threading.Thread(target=worker_fillna, args=(fillna, arrs_nan, arrs_not_nan))
    thread_save = threading.Thread(target=worker_save, args=(save_video_npy, paths_not_nan, dst_path, arrs_not_nan))

    global curr_task_idx, total_tasks

    total_tasks = len(video_paths)
    curr_task_idx = 0

    thread_load.start()
    thread_fillna.start()
    thread_save.start()

    thread_load.join()
    thread_save.join()
    thread_fillna.join()

def estimate_nan(x, y, x_missed):
    x = np.asarray(x).reshape(-1, 1)
    x_missed = np.asarray(x_missed).reshape(-1, 1)
    poly = PolynomialFeatures(7)
    x = poly.fit_transform(x)
    x_missed = poly.transform(x_missed)
    lr = LR()
    lr.fit(x, y)
    
    score = lr.score(x, y)
    y_missed = lr.predict(x_missed)

    y_missed = [x.mean()]*x_missed.shape[0] if score < 0 else y_missed
    return score, y_missed

def find_s_e(vec):
    s, e = 0, len(vec)
    s_not_assigned, e_not_assigned = True, True
    i = 0
    while (s_not_assigned or e_not_assigned) and i< len(vec):
        if s_not_assigned and vec[i] != 0:
            s = i
            s_not_assigned = False
        if e_not_assigned and vec[len(vec)-i-1] != 0:
            e = len(vec)-i
            e_not_assigned = False
        i+=1
    return s, e

def fillna(arr):
    s_e = np.apply_along_axis(arr=arr, func1d=find_s_e, axis=0).T
    
    for i, (vec, (s, e)) in enumerate(zip(arr.T, s_e)):
        values = vec[s:e]
        nan_mask = values==0
        if 0 < nan_mask.sum() < len(nan_mask):
            indices = np.arange(e-s)
            
            indices_not_nan, values_not_nan, indices_nan = indices[~nan_mask], values[~nan_mask], indices[nan_mask]
            score, values_nan = estimate_nan(indices_not_nan, values_not_nan, indices_nan)
            # print(f"score_{i}: {score}")
            values[nan_mask] = values_nan

def get_video_paths(dataset_path, num_signs=5):
    video_paths = []
    for label in sorted(os.listdir(dataset_path))[:num_signs]:
        label_path = os.path.join(dataset_path, label)
        for video in os.listdir(label_path):
            video_path = os.path.join(label_path, video)
            video_paths.append(video_path)
    return video_paths

def save_video_npy(video_path, dst_path, video_npy):
    video = os.path.basename(video_path)
    label = os.path.dirname(video_path).split('\\')[-1]

    video_npy_filename = video[:-4] + ".npy"
    video_npy_dir = os.path.join(dst_path, label)
    video_npy_path = os.path.join(video_npy_dir, video_npy_filename)

    os.makedirs(video_npy_dir, exist_ok=True)
    np.save(video_npy_path, video_npy)

if __name__ == '__main__':
    # dataset_path = 'dataset'
    dataset_path = "write/dataset/path/here"
    dst_path = "write/destination/path/here"
    num_signs=20

    main(dataset_path, dst_path, num_signs)