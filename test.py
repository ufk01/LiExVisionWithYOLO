import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model import HybridFERModel
from enums.config import GlobalValues

class Test:
    def __init__(self):
        print(f"--> Test process is started.", flush=True)
        self._test_model()
        print(f"--> Test process is finished.", flush=True)

    def _test_model(self):
        device = GlobalValues.DEVICE.value
        test_dataset = Dataset(split=GlobalValues.TEST.value)
        print(f"Test Dataset loaded -> Total Test Samples: {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=GlobalValues.BATCH_SIZE.value, shuffle=False, num_workers=2, pin_memory=True)
        model = HybridFERModel(num_classes=len(GlobalValues.emotion_to_idx()), num_blocks=3).to(device)
        model_path = GlobalValues.SAVE_PATH.value 
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"-> Model weights successfully loaded from {model_path}")
        except Exception as e:
            print(f"Model weights could not be loaded: {e}")
            return
        criterion = nn.CrossEntropyLoss()  
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        print("Starting evaluation on test set...", flush=True)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = correct_test / total_test
        print("========================================")
        print(f" TEST RESULTS:")
        print(f" Loss: {avg_test_loss:.4f}")
        print(f" Accuracy: {test_accuracy * 100:.2f}%")
        print("========================================")