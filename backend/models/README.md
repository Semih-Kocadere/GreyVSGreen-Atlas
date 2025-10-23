# Model Dosyaları

Bu klasöre eğitilmiş model dosyalarınızı koyun.

## Gerekli Dosyalar

1. **band_mean.npy** - Normalizasyon için 9-band ortalamaları
2. **band_std.npy** - Normalizasyon için 9-band standart sapmaları  
3. **unet4_best.pt** - UNET segmentation modeli (4 sınıf)
4. **temporal3d_tplus1.pt** - Conv3D temporal prediction modeli (opsiyonel)

## Dosyaları Nereden Almalıyım?

Bu dosyalar eğitim kodunuzdan (greenvsgrey.py) çıktı olarak üretilir:

```python
# Eğitim sonrası bu dosyalar oluşur:
PROJECT_ROOT / "band_mean.npy"           # Normalizasyon
PROJECT_ROOT / "band_std.npy"            # Normalizasyon
PROJECT_ROOT / "unet4_best.pt"           # UNET modeli
PROJECT_ROOT / "temporal3d_tplus1.pt"    # Conv3D modeli
```

**Google Drive'dan İndirme:**

Eğer eğitimi Google Colab'da yaptıysanız, modeller şu konumlarda:
- `/content/drive/MyDrive/greenVSgrey_stats/band_mean.npy`
- `/content/drive/MyDrive/greenVSgrey_stats/band_std.npy`
- `/content/drive/MyDrive/greenVSgrey_models/unet4_best.pt`
- `/content/drive/MyDrive/greenVSgrey_temporal_models/temporal3d_tplus1.pt`

## Kullanım

Model servisi bu dosyaları otomatik olarak yükler:

```bash
python3 backend/model_service.py  # Test için
```

Backend başlatıldığında modeller otomatik yüklenir:

```bash
python3 backend/main.py
```

## Model Spesifikasyonları

### UNET (unet4_best.pt)
- **Mimari:** ResNet34 encoder + U-Net decoder
- **Input:** 9 kanal (B02, B03, B04, B08, B11, B12, NDVI, NDWI, NDBI)
- **Output:** 4 sınıf softmax (background, green, gray, water)
- **Boyut:** 256x256 piksel
- **Doğruluk:** ~94% (Pixel Accuracy), ~91% (mIoU)

### Conv3D (temporal3d_tplus1.pt)
- **Mimari:** 3D CNN (Tiny3D)
- **Input:** 8 çeyrek softmax sequence [8, 4, H, W]
- **Output:** t+1 tahmini [4, H, W]
- **Sequence Length:** 8 quarters (2 yıl)
- **Doğruluk:** ~87% (test data)

## Troubleshooting

**"Model dosyası bulunamadı" Hatası:**
- Bu klasörde gerekli dosyaların olduğundan emin olun
- Dosya isimlerinin tam olarak eşleştiğini kontrol edin
- Dosya izinlerini kontrol edin (`chmod 644 *.npy *.pt`)

**"CUDA out of memory" Hatası:**
- GPU memory yetersiz, CPU moduna geç (otomatik)
- Veya daha az tile ile test edin

**"Modeller yüklenemedi" Hatası:**
- PyTorch ve segmentation-models-pytorch kurulu mu kontrol edin:
  ```bash
  pip install torch torchvision segmentation-models-pytorch
  ```
