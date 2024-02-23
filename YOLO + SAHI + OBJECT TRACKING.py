import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model
from ultralytics.utils.files import increment_path
from sort import *
import cvzone
import numpy as np

def run(cam , h_w , weights='yolov8m.pt'):
    yolov8_model_path = f'models/{weights}'
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8',
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.3,
                                                         device='cuda:0',
                                                         image_size=1080)

    videocapture = cv2.VideoCapture(cam)  

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    while True:
        success, frame = videocapture.read()
        detections = np.empty((0, 5))

        if not success:
            break

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
            boxes = person_pred.bbox.minx, person_pred.bbox.miny, person_pred.bbox.maxx, person_pred.bbox.maxy , person_pred.score.value
            score = person_pred.score
            boxes_list.append(boxes)

        for box in boxes_list:
            x1, y1, x2, y2,conf = box
            current_arr = np.array([x1, y1, x2, y2,conf])
            detections = np.vstack((detections, current_arr))
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 1)
            cvzone.cornerRect(frame, (int(x1), int(y1), int(x2), int(y2)),l=0,rt=1,t=1,colorR=(0,0,255),colorC=(0, 0,2550))
            
        trackers_res = tracker.update(detections)

        for d in trackers_res:
            x1, y1, x2, y2, id = d
            cvzone.putTextRect(frame, f'id:{int(id)} ', (max(0,int(x1)), max(0,int(y1))), 2, 2, colorT=(255, 255,255),colorR=(0,0,255))
            # cv2.putText(frame, str(int(id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Live Inference - Person Class Only', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videocapture.release()
    cv2.destroyAllWindows()


run('9.mp4',400)
