import cv2
import torch
import numpy as np
from helper_functions import *


def main():
    cap = cv2.VideoCapture(0)
    video_npy_ls = []

    potential_label = "None"
    current_label = "None"
    live = 0
    try:
        with PoseLandmarker.create_from_options(pose_options) as poselandmarker:
            with HandLandmarker.create_from_options(hand_options) as handlandmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    frame_npy = frame2npy(frame, frame_timestamp_ms, poselandmarker, handlandmarker)
                    video_npy_ls.append(frame_npy)

                    frame_npy = np.expand_dims(frame_npy, (0, 1))
                    frame_npy_tensor = torch.from_numpy(frame_npy)
                    # frame_npy_tensor = torch.unsqueeze(frame_npy_tensor, dim=0)
                    # frame_npy_tensor = torch.unsqueeze(frame_npy_tensor, dim=0)
                    
                    frame_npy_tensor = frame_npy_tensor.to(model.device)
                    frame_npy_tensor = frame_npy_tensor.to(torch.float32)


                    label = model.predict(frame_npy_tensor)['labels'][0]

                    if label == potential_label:
                        live+=1
                    else:
                        potential_label = label
                        live = 0
                    
                    if live > 5:
                        current_label = potential_label

                    frame = cv2.flip(frame, 1)
                    cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(frame, current_label, (3,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                cap.release()
                cv2.destroyAllWindows()
    except Exception as e:
        # print("Error", e)
        cap.release()
        cv2.destroyAllWindows()     
        raise e

if __name__ == "__main__":
    model = torch.jit.load('model.pt')
    main()