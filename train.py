import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import HybridFERModel
from enums.config import GlobalValues

class Train:

    def __init__(self):
        print(f"--> Train process is started.", flush=True)
        self._train_modal()
        print(f"--> Train process is finished.", flush=True)

    def _train_modal(self):
        """
        Eğitim modelinin temelini oluşturur ve yeni split Dataset yapılarını kullanır.
        """
        device = GlobalValues.DEVICE.value
        train_dataset = Dataset(split=GlobalValues.TRAIN.value)
        val_dataset = Dataset(split=GlobalValues.VALID.value)  
        print(f"Dataset loaded -> Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=GlobalValues.BATCH_SIZE.value, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=GlobalValues.BATCH_SIZE.value, shuffle=False, num_workers=2, pin_memory=True)
        model = HybridFERModel(num_classes=len(GlobalValues.emotion_to_idx()), num_blocks=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=GlobalValues.LEARNING_RATE.value, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=GlobalValues.MIN.value, patience=3, factor=0.5)
        best_val_acc = 0.0
        print("Starting training loop...", flush=True)
        for epoch in range(GlobalValues.EPOCHS.value):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()
            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = correct_train / total_train
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = correct_val / total_val
            scheduler.step(epoch_val_loss)
            print(f"Epoch [{epoch+1}/{GlobalValues.EPOCHS.value}] | "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%", flush=True)
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), GlobalValues.SAVE_PATH.value)
                print(f"--> Saved new best model with Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)
        print(f"Training completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")