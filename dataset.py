import os
import torch
import pickle
import shutil
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
        self.ck_root_csv = GlobalValues.CK_CSV.value
        self.affect_root_dir = GlobalValues.AFFETCTNET_DIR.value
        self.affect_train_dir = GlobalValues.AFFETCTNET_TRAIN_DIR.value
        self.affect_valid_dir = GlobalValues.AFFETCTNET_VALIDATION_DIR.value
        self.affect_test_dir = GlobalValues.AFFETCTNET_TEST_DIR.value
        self.samples = []
        self.classes = []
        self.cache_file = GlobalValues.DATASET_CAHCE_FILE.value
        self._convert_ck_csv_to_images()
        self._convert_affectnet_categorize_images(self.affect_train_dir)
        self._convert_affectnet_categorize_images(self.affect_valid_dir)
        self._convert_affectnet_categorize_images(self.affect_test_dir)
        """
        if os.path.exists(self.cache_file):
            print(f"Loading dataset samples from cache ({self.cache_file})...", flush=True)
            with open(self.cache_file, GlobalValues.RB.value) as f:
                self.samples = pickle.load(f)
            print(f"-> Sucessfully loaded {len(self.samples)} samples from cache (Skipped disk scan).")
        else:
            self._load_all_datasets()
            self._save_cache()
        """


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

    def _folder_dataset(self, root_dir):
        """
        Klasör yapısında olan Fer2013 veri seti için kullanılır
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


    def _folder_AFFECTNET_dataset(self, root_dir):
        emotion_map = GlobalValues.EMOTION.value
        for split_dir_name in os.listdir(root_dir):
            split_path = os.path.join(root_dir, split_dir_name)
            if os.path.isdir(split_path):
                # Bu split altındaki klasörleri tara
                inner_subdirs = os.listdir(split_path)
                for class_name in inner_subdirs:
                    clean_name = class_name.lower().strip()
                    if clean_name in emotion_map:
                        class_idx = emotion_map[clean_name]
                        class_path = os.path.join(split_path, class_name)
                        if os.path.isdir(class_path):
                            for img_name in os.listdir(class_path):
                                img_path = os.path.join(class_path, img_name)
                                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    self.samples.append(("path", img_path, class_idx))

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
                    pil_img = Image.fromarray(image_array).convert(GlobalValues.L.value)
                    self.samples.append((GlobalValues.TENSOR.value, pil_img, label))
                except Exception as e:
                    continue


    def _load_all_datasets(self):
        """
        Veri Setlerini yükler.
        """
        try:
            print(f"Datasets are loading...", flush=True)
            if self.fer_root_dir:
                self._folder_dataset(self.fer_root_dir)
                print(f"-> Fer2013 dataset upload. Total Sample: {len(self.samples)}")
        
        except Exception as e:
            print(f"Load All Datasets Critical Error: {e}", flush=True)

      
         


    def _convert_ck_csv_to_images(self):
        idx_to_emotion_map = GlobalValues.idx_to_emotion()
        base_dir = os.path.dirname(GlobalValues.CK_CSV.value)
        if not os.path.exists(GlobalValues.CK_CSV.value):
            print(f"CSV file cannot found: {GlobalValues.CK_CSV.value}")
            return

        df = pd.read_csv(GlobalValues.CK_CSV.value)
        print("CK+ pixel data converting to the image...", flush=True)
        for idx, row in df.iterrows():
            try:
                pixels = np.fromstring(row[GlobalValues.PIXELS.value], sep=' ')
                image_array = pixels.reshape(48, 48).astype(np.uint8)
                label_idx = int(row[GlobalValues.EMOTION.value])
                if label_idx not in idx_to_emotion_map:
                    continue
                emotion_label = idx_to_emotion_map[label_idx]
                
                class_dir = os.path.join(base_dir, emotion_label)
                os.makedirs(class_dir, exist_ok=True)

                img = Image.fromarray(image_array)
                img.save(os.path.join(class_dir, f"ck_{idx}.png"))
            except Exception as e:
                continue
        print("-> CK+ convert process is done.", flush=True)


    def _convert_affectnet_categorize_images(self, root_dir):
        emotion_to_idx = GlobalValues.affectnet_emotion_to_idx()
        base_dir = os.path.dirname(root_dir)
        clean_root = root_dir.strip().rstrip("/\\")
        images_dir = f"{clean_root}/images"
        labels_dir = f"{clean_root}/labels"
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"AffectNet directory cannot be found: {root_dir}", flush=True)
            return
            
        print(f"AffectNet ({os.path.basename(root_dir)}) data are categorizing...", flush=True)

        success_count = 0
        skipped_count = 0

        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith((GlobalValues.PNG.value, GlobalValues.JPG.value, GlobalValues.JPEG.value)):
                continue
            base_name, _ = os.path.splitext(img_name)
            txt_name = base_name + ".txt"

            txt_path = f"{labels_dir}/{txt_name}"
            img_path = f"{images_dir}/{img_name}"
            
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, GlobalValues.R.value) as f:
                        line = f.readline().strip()
                        print(f"line => {line}")
                        if not line:
                            continue
                        parts = line.split()
                        kaggle_label_idx = int(parts[0])
                        print(f"kaggle_label_idx => {kaggle_label_idx}", flush=True)

                        emotion_name = emotion_to_idx.get(kaggle_label_idx, None)
                        print(f"emotion_to_idx => {emotion_to_idx}", flush=True)
                        if emotion_name:
                            target_class_dir = f"{clean_root}/{emotion_name}"
                            os.makedirs(target_class_dir, exist_ok=True)
                            target_img_path = f"{target_class_dir}/{img_name}"
                            if not os.path.exists(target_img_path):
                                shutil.copy(img_path, target_img_path)

                            success_count += 1
                        else:
                            skipped_count += 1
                
                except Exception as e:
                    continue
                    
        print(f"-> AffectNet classification is successful: {success_count}, Skipped: {skipped_count}", flush=True)
                                

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


