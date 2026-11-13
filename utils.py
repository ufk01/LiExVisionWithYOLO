import os
import random
import shutil
from enums.config import GlobalValues

def organize_datasets_in_place(ck_source_dir, fer_source_dir):
    """
    1. CK+ veri setini alır; bulunduğu dizin içinde train (%70), val (%15), test (%15) 
       olarak alt duygu klasörleriyle birlikte böler.
    2. FER2013 veri setinin mevcut train klasöründen %10 alarak bir val klasörü oluşturur,
       mevcut train ve test klasörlerini korur/yeniden düzenler.
    """
    random.seed(42)  # Kararlı ve sabit dağılım için
    
    # ==========================================
    # 1. CK+ İÇİN KENDİ DİZİNİNDE BÖLME (Train / Val / Test)
    # ==========================================
    if ck_source_dir and os.path.exists(ck_source_dir):
        print("-> CK+ veri seti kendi dizininde train, val ve test olarak bölünüyor...", flush=True)
        
        # Eğer ham veriler doğrudan duygu klasörlerindeyse onları tarıyoruz
        emotions = [d for d in os.listdir(ck_source_dir) if os.path.isdir(os.path.join(ck_source_dir, d))]
        
        # Zaten train/val/test olarak bölünmüşse tekrar bölmeyi önleyelim
        if not any(split in emotions for split in ['train', 'val', 'test']):
            for emotion in emotions:
                emotion_path = os.path.join(ck_source_dir, emotion)
                images = [img for img in os.listdir(emotion_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                random.shuffle(images)
                
                n_total = len(images)
                n_train = int(n_total * 0.70)
                n_val = int(n_total * 0.15)
                
                train_imgs = images[:n_train]
                val_imgs = images[n_train:n_train + n_val]
                test_imgs = images[n_train + n_val:]
                
                # Kendi dizini altında train, val, test klasörlerine taşıyalım/kopyalayalım
                for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
                    target_dir = os.path.join(ck_source_dir, split_name, emotion)
                    os.makedirs(target_dir, exist_ok=True)
                    for img in img_list:
                        src = os.path.join(emotion_path, img)
                        dst = os.path.join(target_dir, img)
                        if not os.path.exists(dst):
                            shutil.copy(src, dst)
            print("-> CK+ kendi dizini içinde başarıyla bölündü.")

    # ==========================================
    # 2. FER2013 İÇİN KENDİ DİZİNİNDE TRAİN'DEN VAL KOPARMA
    # ==========================================
    fer_train_source = os.path.join(fer_source_dir, GlobalValues.TRAIN.value)
    fer_val_target = os.path.join(fer_source_dir, GlobalValues.VALID.value)
    
    if fer_source_dir and os.path.exists(fer_train_source):
        print("-> FER2013 train verisinden validation (val) kendi dizini için ayrılıyor...", flush=True)
        
        # Eğer val klasörü daha önce oluşturulmadıysa oluşturalım
        if not os.path.exists(fer_val_target) or not os.listdir(fer_val_target):
            for emotion in os.listdir(fer_train_source):
                emotion_path = os.path.join(fer_train_source, emotion)
                if not os.path.isdir(emotion_path):
                    continue
                
                images = [img for img in os.listdir(emotion_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                random.shuffle(images)
                
                n_total = len(images)
                n_val = int(n_total * 0.10)  # Train verisinin %10'unu val yapıyoruz
                
                val_imgs = images[:n_val]
                
                # Val klasörüne taşıyoruz
                target_dir = os.path.join(fer_val_target, emotion)
                os.makedirs(target_dir, exist_ok=True)
                for img in val_imgs:
                    src = os.path.join(emotion_path, img)
                    dst = os.path.join(target_dir, img)
                    if not os.path.exists(dst):
                        shutil.move(src, dst) # Train'den val'e taşıyoruz ki çakışma/kopya olmasın
            print("-> FER2013 val klasörü kendi dizininde oluşturuldu.")