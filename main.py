import argparse
import torch
import cv2
from train import Train
from test import Test
from model import HybridFERModel
from detector import FaceDetector
from enums.config import GlobalValues

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid FER Pipeline")
    parser.add_argument(
        "--mode", 
        type=str, 
        default=GlobalValues.TRAIN.value, 
        choices=[GlobalValues.TRAIN.value, GlobalValues.TEST.value, GlobalValues.CLEAN.value, GlobalValues.LIVE.value], 
        help="Operation mode: train, test, clean or live"
    )
    args = parser.parse_args()

    if args.mode == GlobalValues.TRAIN.value or args.mode == GlobalValues.VALID.value:
        print("=== Training Process Initiated ===", flush=True)
        trainer = Train()
    elif args.mode == GlobalValues.TEST.value:
        print("=== Test Process Initiated ===", flush=True)
        tester = Test()
    elif args.mode == GlobalValues.LIVE.value:
        print("=== Application Initiated ===", flush=True)
        device = GlobalValues.DEVICE.value
        model = HybridFERModel(num_classes=len(GlobalValues.emotion_to_idx()), num_blocks=3).to(device)
        try:
            model.load_state_dict(torch.load(GlobalValues.SAVE_PATH.value, map_location=device))
            model.eval()
            print(f"-> Model weights successfully loaded from {GlobalValues.SAVE_PATH.value} and eval applied.")
        except Exception as e:
            print(f"Model weights could not be loaded: {e}")
            raise RuntimeError(f"Model weights cannot found!")
        
        detector = FaceDetector()
        cv2.namedWindow(GlobalValues.WINDOW_NAME.value, cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera cannot opened", flush=True)
            raise RuntimeError(f"Camera cannot opened")
        
        print(f"Real-time emotion recognition system started. Press 'q' to exit.")
        
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                faces = detector.detect_faces(frame)
                detector.arrange_face_frames(faces, frame, model)
                cv2.imshow("Live Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty(GlobalValues.WINDOW_NAME.value, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
        cap.release()
        cv2.destroyAllWindows()
                
        
        
        