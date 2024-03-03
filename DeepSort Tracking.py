import cv2
import cvzone
import numpy as np
from sahi import AutoDetectionModel
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
from deep_sort.utils.parser import get_config
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

tracker = DeepSort(model_path='deep_sort/deep/checkpoint/ckpt.t7', max_age=70)

def run1(frame, h_w=512, weights='yolov8m.pt'):
    yolov8_model_path = f'models/{weights}'
    download_yolov8s_model(yolov8_model_path)
    
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.3,
                                                         device='cuda:0',
                                                         image_size=1080)

    results = get_sliced_prediction(frame,
                                    detection_model,
                                    slice_height=h_w,
                                    slice_width=h_w,
                                    overlap_height_ratio=0.2,
                                    overlap_width_ratio=0.2)
    object_prediction_list = results.object_prediction_list

    person_predictions = [obj for obj in object_prediction_list if obj.category.name == "person"]

    boxes_list = []
    for person_pred in person_predictions:
        boxes = person_pred.bbox.minx, person_pred.bbox.miny, person_pred.bbox.maxx, person_pred.bbox.maxy, person_pred.score.value
        boxes_list.append(boxes)

    if boxes_list:
        conf = np.array(boxes_list)[:,-1].reshape(-1, 1)
        boxes_np = np.array(boxes_list,dtype=np.int16)[:, :-1]
        trackers = tracker.update(boxes_np, conf, frame)

        for track in trackers:
            bbox = track[0:4]  # Get x1, y1, x2, y2 coordinates directly
            track_id = track[4] 
            cvzone.cornerRect(frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), l=0, rt=1, t=1, colorR=(0, 0, 255), colorC=(0, 0, 255))
            cvzone.putTextRect(frame, f'Track ID: {int(track_id)}', (max(0, int(bbox[0])), max(0, int(bbox[1]))), 1, 2, colorT=(255, 255, 255), colorR=(0, 0, 255))

    return frame

# Example usage with VideoCapture
# cap = cv2.VideoCapture(0)
# while True:
#     success, frame = cap.read()
#     frame = cv2.resize(frame, (1080, 720))  # Adjust size as needed
#     result_frame = run1(frame, 800, 'yolov8n.pt')
    
#     cv2.imshow('Live Inference - Person Class Only', result_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('1.mp4')

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (1080, 720))

while True:
    success, frame = cap.read()
    
    if not success:
        break

    frame = cv2.resize(frame, (1080, 720))  # Adjust size as needed
    result_frame = run1(frame, 300, 'yolov8m.pt')
    
    # Save the frame to the output video file
    out.write(result_frame)

    # cv2.imshow('Live Inference - Person Class Only', result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()