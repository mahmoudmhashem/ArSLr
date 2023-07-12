import cv2
import pickle
import numpy as np
import mediapipe as mp
from typing import List
mp_hands = mp.solutions.hands
class MyModel:
    model_path = '../assets/model.pkl'
    le_path = '../assets/labelencoder.pkl'

    def __init__(self) -> None:
        with open(MyModel.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(MyModel.le_path, 'rb') as f:
            self.le = pickle.load(f)
    
    def predict(self, images:List[np.ndarray]):
        landmarks = np.zeros((len(images), 21, 2))
        with mp_hands.Hands() as hands:
            for i, image in enumerate(images):
                try:
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    mp_landmarks = results.multi_hand_world_landmarks[0]
                except TypeError:
                    continue
                
                for j, landmark in enumerate(mp_landmarks.landmark):
                    landmarks[i, j] = landmark.x, landmark.y
        landmarks = landmarks.reshape(len(images), 21*2)
        labelsIdx = self.model.predict(landmarks)
        labels = self.le.inverse_transform(labelsIdx)
        return labels