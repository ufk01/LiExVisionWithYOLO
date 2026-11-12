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
    CK_DIR = r"dataset/CK+_Dataset"
    CK_CSV = r"dataset/CK+_Dataset/ckextended.csv"
    AFFETCTNET_DIR = r"dataset/AffectNet_Dataset"
    AFFETCTNET_TRAIN_DIR = r"dataset/AffectNet_Dataset/train"
    AFFETCTNET_VALIDATION_DIR = r"dataset/AffectNet_Dataset/valid"
    AFFETCTNET_TEST_DIR = r"dataset/AffectNet_Dataset/test"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    RGB = "RGB"
    PATH = "path"
    EMOTION = "emotion"
    TENSOR = "tensor"
    PIXELS = "pixels"
    DATASET_CAHCE_FILE = "dataset_cache.pkl"
    RB = "rb"
    WB = "wb"
    MIN = "min"
    SAVE_PATH = "best_fer_model.pth"
    L = "L"
    R = "r"


    @classmethod
    def emotion_to_idx(cls):
        return {
            'angry': 0, 
            'disgust': 1, 
            'fear': 2, 
            'happy': 3, 
            'sad': 4, 
            'surprise': 5, 
            'neutral': 6 
        }

    @classmethod
    def idx_to_emotion(cls):
        return {v: k for k, v in cls.emotion_to_idx().items()}

    @classmethod
    def affectnet_emotion_to_idx(cls):
        return {
            0: 'angry',       
            1: 'contempt',          
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

