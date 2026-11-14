import torch
from enum import Enum, unique

@unique
class GlobalValues(Enum):
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 25
    IMAGE_SIZE = (48, 48)
    ANGRY = "angry"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    SUPRISE = "suprise"
    FER2013_DIR = r"dataset/Fer2013_Dataset"
    FER2013_TRAIN_DIR = r"dataset/Fer2013_Dataset/train"
    FER2013_VALID_DIR = r"dataset/Fer2013_Dataset/valid"
    FER2013_TEST_DIR = r"dataset/Fer2013_Dataset/test" 
    CK_DIR = r"dataset/CK+_Dataset"
    CK_TRAIN_DIR = r"dataset/CK+_Dataset/train"
    CK_VALID_DIR = r"dataset/CK+_Dataset/valid"
    CK_TEST_DIR = r"dataset/CK+_Dataset/test"
    CK_CSV = r"dataset/CK+_Dataset/ckextended.csv"
    AFFETCTNET_DIR = r"dataset/AffectNet_Dataset"
    AFFETCTNET_TRAIN_DIR = r"dataset/AffectNet_Dataset/train"
    AFFETCTNET_VALIDATION_DIR = r"dataset/AffectNet_Dataset/valid"
    AFFETCTNET_TEST_DIR = r"dataset/AffectNet_Dataset/test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"
    LIVE = "live"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    RGB = "RGB"
    PATH = "path"
    EMOTION = "emotion"
    TENSOR = "tensor"
    PIXELS = "pixels"
    DATASET_TRAIN_CAHCE_FILE = "dataset_train_cache.pkl"
    DATASET_VALID_CAHCE_FILE = "dataset_valid_cache.pkl"
    DATASET_TEST_CAHCE_FILE = "dataset_test_cache.pkl"
    RB = "rb"
    WB = "wb"
    MIN = "min"
    SAVE_PATH = "best_model.pth"
    L = "L"
    R = "r"
    MODE = '--mode'
    CLEAN = 'clean'
    MAIN_FILE = "main.py"
    YOLO_MODEL_PATH = "yolov8n.pt"
    ANALYZING = "Analyzing..."
    WINDOW_NAME = "Live Detection"


    @classmethod
    def emotion_to_idx(cls):
        return {
            'happy': 0, 
            'sad': 1, 
            'surprise': 2, 
            'neutral': 3 
        }

    @classmethod
    def idx_to_emotion(cls):
        return {v: k for k, v in cls.emotion_to_idx().items()}

    @classmethod
    def affectnet_emotion_to_idx(cls):
        return {
            0: 'angry',       
            1: 'None',          
            2: 'disgust',     
            3: 'fear',        
            4: 'happy',       
            5: 'neutral',     
            6: 'sad',         
            7: 'surprise'
        }

    @classmethod
    def affectnet_idx_to_emotion(cls):
        return {v: k for k, v in cls.affectnet_emotion_to_idx().items()}
    

