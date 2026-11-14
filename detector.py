import cv2
import torch
import time
from collections import deque
from ultralytics import YOLO
from enums.config import GlobalValues

class FaceDetector:
    def __init__(self, model_path=GlobalValues.YOLO_MODEL_PATH.value):
        self.model = YOLO(model_path)
        self.last_update_time = 0
        self.cached_emotion_text = "Analyzing..."
        self.update_interval = 0.1
        self.prediction_history = deque(maxlen=3)
        
    def detect_faces(self, frame):
        results = self.model(frame, verbose=False, conf=0.7)
        
        faces = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2, y2))
                
        return faces
    
    def arrange_face_frames(self, faces, frame, model):
        device = GlobalValues.DEVICE.value
        emotion_labels = GlobalValues.idx_to_emotion()
        current_time = time.time()       
        for (x1, y1, x2, y2) in faces:
            h, w, _ = frame.shape       
            box_width = x2 - x1
            box_height = y2 - y1
            padding_x = int(box_width * 0.1)
            padding_y = int(box_height * 0.1)    
            startX = max(0, x1 - padding_x)
            startY = max(0, y1 - padding_y)
            endX = min(w, x2 + padding_x)
            endY = min(h, y2 + padding_y)          
            face_crop = frame[startY:endY, startX:endX]
            if face_crop.size == 0:
                continue     
            if current_time - self.last_update_time >= self.update_interval:
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, GlobalValues.IMAGE_SIZE.value)
                normalized_face = resized_face / 255.0
                tensor_face = torch.tensor(normalized_face, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)             
                tensor_face = (tensor_face - 0.5) / 0.5
                outputs = model(tensor_face)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)             
                self.prediction_history.append(predicted_idx.item())
                smoothed_idx = max(set(self.prediction_history), key=self.prediction_history.count)    
                self.cached_emotion_text = f"{emotion_labels[smoothed_idx]} ({confidence.item()*100:.1f}%)"            
                self.last_update_time = current_time 
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)     
            text_size = cv2.getTextSize(self.cached_emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (startX, startY - 30), (startX + text_size[0], startY), (0, 255, 0), -1)
            cv2.putText(frame, self.cached_emotion_text, (startX, startY - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)