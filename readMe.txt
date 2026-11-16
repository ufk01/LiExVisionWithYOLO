========================================
 Hybrid FER (Facial Emotion Recognition)
========================================

Bu proje; FER2013, CK+ ve AffectNet veri setlerini birleştiren, disk-cache (pickle) optimizasyonuna sahip modüler bir derin öğrenme boru hattıdır.


---
1. KURULUM (Installation)
---

Projede kullanılan gerekli Python kütüphanelerini yüklemek için terminale şunu yazın:
pip install -r requirements.txt

Task runner (komut yönetimi) için poethepoet paketini yükleyin:
pip install poethepoet


---
2. PROJEYİ ÇALIŞTIRMA KOMUTLARI (Scripts)
---

Node.js'deki package.json konforunda, poethepoet kullanarak projeyi şu komutlarla yönetebilirsiniz:

- Eğitimi Başlatmak İçin:
  poe train

- Modeli Test Etmek İçin:
  poe test

- Pickle Cache Dosyalarını Temizlemek İçin:
  poe clean


---
3. ALTERNATİF KLASİK KOMUTLAR (CLI)
---

İsterseniz doğrudan argparse altyapısını kullanarak da çalıştırabilirsiniz:
- python main.py --mode train
- python main.py --mode test



- Nuitka DERLEME KOMUT:
python -m nuitka --standalone --mingw64 --jobs=8 --module-parameter=torch-disable-jit=yes --enable-plugin=no-qt --nofollow-import-to=torch._dynamo --include-data-file=yolov8n.pt=yolov8n.pt --include-data-file=best_model.pth=best_model.pth --output-dir=dist main.py
