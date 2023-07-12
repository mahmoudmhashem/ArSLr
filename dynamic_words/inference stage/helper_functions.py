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

def mediapipe_detection(numpy_image, frame_timestamp_ms, poselandmarker, handlandmarker): # we assume that input image is in BGR format

    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    numpy_image = cv2.flip(numpy_image, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    pose_landmarker_result = poselandmarker.detect_for_video(mp_image, frame_timestamp_ms)
    hand_landmarker_result = handlandmarker.detect_for_video(mp_image, frame_timestamp_ms)

    return pose_landmarker_result, hand_landmarker_result

def extract_keypoints(pose_landmarker_result, hand_landmarker_result):
    
    bodies = pose_landmarker_result.pose_world_landmarks
    body_npy = np.zeros(33*2)
    i=0
    for body in bodies:
        for landmark in body:
            body_npy[i:i+2] = [landmark.x, landmark.y]
            i+=2

    body_npy = body_npy[:25*2]

    hands = hand_landmarker_result.hand_world_landmarks
    hands_npy = np.zeros(21*2*2)
    i=0
    for hand in hands:
        for landmark in hand:
            hands_npy[i:i+2] = [landmark.x, landmark.y]
            i+=2

    landmarks = np.concatenate([body_npy, hands_npy], axis=0)
    return landmarks


def frame2npy(image, frame_timestamp_ms, poselandmarker, handlandmarker):
    pose_results, hands_results = mediapipe_detection(image, frame_timestamp_ms, poselandmarker, handlandmarker)
    landmarks = extract_keypoints(pose_results, hands_results)
    return landmarks