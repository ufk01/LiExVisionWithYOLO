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
    FER2013_DIR = "dataset/Fer2013_Dataset"
    CK_DIR = "dataset/CK+_Dataset"
    AFFETCTNET_DIR = "dataset/AffectNet_Dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    RGB = "RGB"
    PATH = "path"
    EMOTION = "emotion"
    TENSOR_OR_IMAGE = "tensor_or_image"


    @classmethod
    def emotion_to_idx(self):
        return {
            'angry': 0, 
            'disgust': 1, 
            'fear': 2, 
            'happy': 3, 
            'neutral': 4, 
            'sad': 5, 
            'surprise': 6
        }

    @classmethod
    def idx_to_emotion(self):
        return {v: k for k, v in self.emotion_to_idx.items()}