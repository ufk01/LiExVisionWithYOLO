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
    def __init__(self, split=GlobalValues.TRAIN.value):
        self.split = split
        self.transform_pipeline = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(GlobalValues.IMAGE_SIZE.value),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.fer_root_dir = GlobalValues.FER2013_DIR.value
        self.fer_train_dir = GlobalValues.FER2013_TRAIN_DIR.value
        self.fer_valid_dir = GlobalValues.FER2013_VALID_DIR.value
        self.fer_test_dir = GlobalValues.FER2013_TEST_DIR.value
        self.ck_root_csv = GlobalValues.CK_CSV.value
        self.ck_root_dir = GlobalValues.CK_DIR.value
        self.ck_train_dir = GlobalValues.CK_TRAIN_DIR.value
        self.ck_valid_dir = GlobalValues.CK_VALID_DIR.value
        self.ck_test_dir = GlobalValues.CK_TEST_DIR.value
        self.affect_root_dir = GlobalValues.AFFETCTNET_DIR.value
        self.affect_train_dir = GlobalValues.AFFETCTNET_TRAIN_DIR.value
        self.affect_valid_dir = GlobalValues.AFFETCTNET_VALIDATION_DIR.value
        self.affect_test_dir = GlobalValues.AFFETCTNET_TEST_DIR.value
        self.train_samples = []
        self.valid_samples = []
        self.test_samples = []
        self.samples = []
        self.train_cache_file = GlobalValues.DATASET_TRAIN_CAHCE_FILE.value
        self.valid_cache_file = GlobalValues.DATASET_VALID_CAHCE_FILE.value
        self.test_cache_file = GlobalValues.DATASET_TEST_CAHCE_FILE.value
        self.cache_file = None
        if not os.path.exists(self.train_cache_file) or not os.path.exists(self.valid_cache_file) or not os.path.exists(self.test_cache_file):
            self._load_all_datasets()
            self._save_cache()    
        if self.split == GlobalValues.TRAIN.value:
            self.cache_file = GlobalValues.DATASET_TRAIN_CAHCE_FILE.value
        elif self.split == GlobalValues.VALID.value:
            self.cache_file = GlobalValues.DATASET_VALID_CAHCE_FILE.value
        elif self.split == GlobalValues.TEST.value:
            self.cache_file = GlobalValues.DATASET_TEST_CAHCE_FILE.value
        else:
            raise ValueError(f"Unvalid Split Value: {self.split}. 'train', 'valid' and 'test' must be used.")
        if os.path.exists(self.cache_file):
            print(f"Loading {self.split} dataset samples from cache ({self.cache_file})...", flush=True)
            with open(self.cache_file, GlobalValues.RB.value) as f:
                self.samples = pickle.load(f)
            print(f"--> Successfully loaded {len(self.samples)} {self.split} samples from cache (Skipped disk scan).")
        else:
            print(f"Cache file ({self.cache_file}) not found! Starting full disk scan...", flush=True)
            self._load_all_datasets()
            self._save_cache()            
            if self.split == GlobalValues.TRAIN.value:
                self.samples = self.train_samples
            elif self.split == GlobalValues.VALID.value:
                self.samples = self.valid_samples
            elif self.split == GlobalValues.TEST.value:
                self.samples = self.test_samples

    def _save_cache(self):
        """
        Oluşturulan samples listesini sonraki çalıştırmalar için diske kaydeder
        """
        try:
            if self.split == GlobalValues.TRAIN.value:
                print(f"Saving train dataset cache to {self.train_cache_file}...", flush=True)
                with open(self.train_cache_file, GlobalValues.WB.value) as f:
                    pickle.dump(self.train_samples, f)
                print("-> Train Cache saved successfully.", flush=True)
            elif self.split == GlobalValues.VALID.value:
                print(f"Saving valid dataset cache to {self.valid_cache_file}...", flush=True)
                with open(self.valid_cache_file, GlobalValues.WB.value) as f:
                    pickle.dump(self.valid_samples, f)
                print("-> Valid Cache saved successfully.", flush=True)
            elif self.split == GlobalValues.TEST.value:
                print(f"Saving test dataset cache to {self.test_cache_file}...", flush=True)
                with open(self.test_cache_file, GlobalValues.WB.value) as f:
                    pickle.dump(self.test_samples, f)
                print("-> Test Cache saved successfully.", flush=True)
        except Exception as e:
            print(f"Could not save cache samples. Error: {e}")      

    def _folder_dataset(self, root_dir):
        """
        Klasör yapısını dinamik olarak algılayan ve FER2013, CK+ gibi 
        farklı hiyerarşilere uyum sağlayan genel veri yükleyici. 
        3 tane .pickle formatı  train-valid-test olarak oluşturulur.
        Böylece her seferinde yükleme için I/O işlemi gerekmez.
        Not: Bu işlemden önce veri setlerinin train-valid-test olarak duygu klasörlerinde toplandığına emin olunmalı.
        """
        if not root_dir or not os.path.exists(root_dir):
            return
        emotion_map = GlobalValues.emotion_to_idx()
        target_dir = root_dir
        if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
            return         
        for class_name in os.listdir(target_dir):
            clean_name = class_name.lower().strip()    
            if clean_name in emotion_map:
                class_idx = emotion_map[clean_name]
                class_path = os.path.join(target_dir, class_name)            
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        if img_path.lower().endswith((GlobalValues.PNG.value, GlobalValues.JPG.value, GlobalValues.JPEG.value)):
                            match self.split:
                                case GlobalValues.TRAIN.value:
                                    self.train_samples.append((GlobalValues.PATH.value, img_path, class_idx))
                                case GlobalValues.VALID.value:
                                    self.valid_samples.append((GlobalValues.PATH.value, img_path, class_idx))
                                case GlobalValues.TEST.value:
                                    self.test_samples.append((GlobalValues.PATH.value, img_path, class_idx))

    def _load_all_datasets(self):
        """
        Veri Setlerini yükler ve aktif split'e göre loglar.
        """
        try:
            print(f"Datasets are loading for split: {self.split}...", flush=True)
            
            # Fer2013 yükleme
            if self.fer_root_dir and self.fer_train_dir and self.fer_valid_dir and self.fer_test_dir:
                if self.split == GlobalValues.TRAIN.value:
                    self._folder_dataset(self.fer_train_dir)
                elif self.split == GlobalValues.VALID.value:
                    self._folder_dataset(self.fer_valid_dir)
                elif self.split == GlobalValues.TEST.value:
                    self._folder_dataset(self.fer_test_dir)
                print(f"-> Fer2013 dataset ({self.split}) loaded.")

            # AffectNet yükleme
            if self.affect_root_dir:
                if self.split == GlobalValues.TRAIN.value and self.affect_train_dir:
                    self._folder_dataset(self.affect_train_dir)
                elif self.split == GlobalValues.VALID.value and self.affect_valid_dir:
                    self._folder_dataset(self.affect_valid_dir)
                elif self.split == GlobalValues.TEST.value and self.affect_test_dir:
                    self._folder_dataset(self.affect_test_dir)
                print(f"-> AffectNet dataset ({self.split}) loaded.")

            # CK+ yükleme
            if self.ck_root_dir:
                if self.split == GlobalValues.TRAIN.value and self.ck_train_dir:
                    self._folder_dataset(self.ck_train_dir)
                elif self.split == GlobalValues.VALID.value and self.ck_valid_dir:
                    self._folder_dataset(self.ck_valid_dir)
                elif self.split == GlobalValues.TEST.value and self.ck_test_dir:
                    self._folder_dataset(self.ck_test_dir)
                print(f"-> CK+ dataset ({self.split}) loaded.")

            # Sonuçları aktif split'e göre yazdır
            if self.split == GlobalValues.TRAIN.value:
                print(f"-> Total Loaded Train Samples: {len(self.train_samples)}", flush=True)
            elif self.split == GlobalValues.VALID.value:
                print(f"-> Total Loaded Valid Samples: {len(self.valid_samples)}", flush=True)
            elif self.split == GlobalValues.TEST.value:
                print(f"-> Total Loaded Test Samples: {len(self.test_samples)}", flush=True)
        
        except Exception as e:
            print(f"Load All Datasets Critical Error: {e}", flush=True)
   
    def _csv_ck_dataset(self, csv_path):
        """
        CK+ veri setinin CSV tabanlı pikseleri için kullanılır. 48x48'lik görseller elde edilir.
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
 
    def _convert_ck_csv_to_images(self):
        """
        CK+ veri setinde görseller pikselden sonra karmaşık durduğundan duygu klasörlerine göre sınıflar.
        """
        
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
        """
        AffectNet veri yapısında label - images olark ayrılmıştır. Aynı şekilde label içerisindeki .txt dosyaları ile eşleştirerek
        duygu klasölerleri altında sınıflandırılmasını sağlar.
        """
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
                        if not line:
                            continue
                        parts = line.split()
                        kaggle_label_idx = int(parts[0])
                        emotion_name = emotion_to_idx.get(kaggle_label_idx, None)
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


