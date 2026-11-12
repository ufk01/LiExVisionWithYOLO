import os
import torch
import pickle
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from enums.config import GlobalValues

class Dataset(TorchDataset):
    def __init__(self,):
        self.transform_pipeline = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(GlobalValues.IMAGE_SIZE.value),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.fer_root_dir = GlobalValues.FER2013_DIR.value
        self.affect_root_dir = GlobalValues.AFFETCTNET_DIR.value
        self.ck_root_dir = GlobalValues.CK_DIR.value
        self.samples = []
        self.classes = []
        self.cache_file = GlobalValues.DATASET_CAHCE_FILE.value
        if os.path.exists(self.cache_file):
            print(f"Loading dataset samples from cache ({self.cache_file})...", flush=True)
            with open(self.cache_file, GlobalValues.RB.value) as f:
                self.samples = pickle.load(f)
            print(f"-> Sucessfully loaded {len(self.samples)} samples from cache (Skipped disk scan).")
        else:
            self._load_all_datasets()
            self._save_cache()


    def _save_cache(self):
        """
        Oluşturulan samples listesini sonraki çalıştırmalar için diske kaydeder
        """
        try:
            print(f"Saving dataset cache to {self.cache_file}...", flush=True)
            with open(self.cache_file, GlobalValues.WB.value) as f:
                pickle.dump(self.samples, f)
            print("-> Cache saved successfully.", flush=True)
        except Exception as e:
            print(f"Could not save cache: {e}")

    def _folder_FER_Affect_dataset(self, root_dir):
        """
        Klasör yapısında olan Fer2013 ve AffectNet veri seti için kullanılır
        """
        if root_dir and os.path.exists(root_dir):
            check_dirs = set()
            for split in [GlobalValues.TRAIN.value, GlobalValues.TEST.value, GlobalValues.VALID.value]:
                split_path = os.path.join(root_dir, split)
                target_dir = split_path if os.path.exists(split_path) else root_dir
                if target_dir in check_dirs:
                    continue
                check_dirs.add(target_dir)
                if os.path.exists(target_dir):
                    subdirs = sorted(os.listdir(target_dir))
                    for class_name in subdirs:
                        clean_name = class_name.lower().strip()
                        if clean_name in GlobalValues.emotion_to_idx():
                            class_idx = GlobalValues.emotion_to_idx()[clean_name]
                            class_path = os.path.join(target_dir, class_name)
                            if os.path.isdir(class_path):
                                for img_name in os.listdir(class_path):
                                    img_path = os.path.join(class_path, img_name)
                                    if img_path.lower().endswith((GlobalValues.PNG.value, GlobalValues.JPG.value, GlobalValues.JPEG.value)):
                                        self.samples.append((GlobalValues.PATH.value, img_path, class_idx))

    def _csv_ck_dataset(self, csv_path):
        """
        CK+ veri setinin CSV tabanlı pikseleri için kullanılır
        """   
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                try:
                    pixels = np.fromstring(row[GlobalValues.PIXELS.value], sep =' ')
                    image_array = pixels.reshape(48, 48).astype(np.uint8)
                    label = int(row[GlobalValues.EMOTION.value])
                    pil_img = Image.fromarray(image_array).convert(GlobalValues.RGB.value)
                    self.samples.append((GlobalValues.TENSOR_OR_IMAGE.value, pil_img, label))
                except Exception as e:
                    continue


    def _load_all_datasets(self):
        """
        Veri Setlerini yükler.
        """
        try:
            print(f"Datasets are loading...", flush=True)
            if self.fer_root_dir:
                self._folder_FER_Affect_dataset(self.fer_root_dir)
                print(f"-> Fer2013 dataset upload. Total Sample: {len(self.samples)}")

            if self.affect_root_dir:
                prev_len = len(self.samples)
                self._folder_FER_Affect_dataset(self.affect_root_dir)
                print(f"-> AffectNet dataset upload. Total Sample: {len(self.samples) - prev_len}")

            if self.ck_root_dir:
                self._csv_ck_dataset(self.ck_root_dir)
                print(f"Datasets are upload successfully.", flush=True)
        except Exception as e:
            print(f"Load all datasets critical error: {e}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_type, content, label = self.samples[idx]

        if data_type == GlobalValues.PATH.value:
            image = Image.open(content).convert(GlobalValues.RGB.value)
        else:
            image = content
        if self.transform_pipeline:
            image = self.transform_pipeline(image)
        return image, torch.tensor(label, dtype=torch.long)


