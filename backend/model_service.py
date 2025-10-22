"""
Model Service - UNET + Conv3D Temporal Prediction
==================================================
Eğitilmiş modelleri yükler ve tahmin yapar.

Modeller:
- UNET: 4-class segmentation (background, green, gray, water)
  Input: 9 bands (B02-B12 + NDVI, NDWI, NDBI)
  Output: 4-class softmax [C, H, W]

- Conv3D: Temporal prediction (8 quarters → t+1)
  Input: [T, 4, H, W] softmax sequence
  Output: [4, H, W] prediction for t+1

Veri Akışı:
1. Tile'dan 6 band oku (.npy)
2. NDVI, NDWI, NDBI hesapla → 9 band
3. Normalize et (mean/std)
4. UNET → softmax
5. Sequence oluştur (8 çeyrek)
6. Conv3D → t+1 tahmin
7. İstatistikleri hesapla ve döndür
"""

import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ============================================================================
# YAPILANDIRMA
# ============================================================================

# Tile ve model dosyaları
TILES_DIR = Path(__file__).parent / "tiles" / "images"
MODEL_DIR = Path(__file__).parent / "models"  # Modelleri buraya koyacağız
MEAN_PATH = MODEL_DIR / "band_mean.npy"
STD_PATH = MODEL_DIR / "band_std.npy"
UNET_PATH = MODEL_DIR / "unet4_best.pt"
CONV3D_PATH = MODEL_DIR / "temporal3d_tplus1.pt"

# Device (GPU varsa kullan)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dosya ismi regex: 2025_Q1_00000_00256.npy
FILENAME_PATTERN = re.compile(r"(?P<year>\d{4})_Q(?P<quarter>[1-4])_(?P<row>\d{5})_(?P<col>\d{5})\.npy")

# Sınıf isimleri
CLASS_NAMES = ["background", "green", "gray", "water"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_spectral_indices(six_band_data: np.ndarray) -> np.ndarray:
    """
    6 banttan NDVI, NDWI, NDBI hesapla.
    Input: [B02, B03, B04, B08, B11, B12] - (6, H, W)
    Output: [NDVI, NDWI, NDBI] - (3, H, W)
    """
    assert six_band_data.shape[0] == 6, f"6 band bekleniyor, {six_band_data.shape[0]} geldi"
    
    B02, B03, B04, B08, B11, B12 = six_band_data
    eps = 1e-6
    
    ndvi = (B08 - B04) / (B08 + B04 + eps)
    ndwi = (B03 - B08) / (B03 + B08 + eps)
    ndbi = (B11 - B08) / (B11 + B08 + eps)
    
    return np.stack([ndvi, ndwi, ndbi], axis=0)


def calculate_9_bands_from_6(six_band_data: np.ndarray) -> np.ndarray:
    """
    6 banttan 9 bant oluştur: [B02-B12, NDVI, NDWI, NDBI]
    Input: (6, H, W)
    Output: (9, H, W)
    """
    indices = calculate_spectral_indices(six_band_data)
    return np.concatenate([six_band_data, indices], axis=0)


def parse_filename(filename: str) -> Optional[Dict[str, int]]:
    """
    Dosya isminden metadata çıkar.
    
    Args:
        filename: "2025_Q1_00000_00256.npy"
    
    Returns:
        {"year": 2025, "quarter": 1, "row": 0, "col": 256}
        veya None (eşleşmezse)
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    
    return {
        "year": int(match.group("year")),
        "quarter": int(match.group("quarter")),
        "row": int(match.group("row")),
        "col": int(match.group("col"))
    }


def get_tile_key(row: int, col: int) -> Tuple[int, int]:
    """Tile koordinatları için benzersiz anahtar."""
    return (row, col)


# ============================================================================
# CONV3D MODEL TANIMI (Eğitim kodundan)
# ============================================================================

class Tiny3D(nn.Module):
    """
    3D CNN for temporal prediction.
    Input: [B, T, 4, H, W]
    Output: [B, 4, H, W]
    """
    def __init__(self, in_channels=4, sequence_length=8):
        super().__init__()
        self.sequence_length = sequence_length
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(32), 
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1, stride=(2,1,1)),
            nn.BatchNorm3d(64), 
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64), 
            nn.ReLU(inplace=True),
        )
        
        self.head = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(sequence_length // 2, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 4, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, T, 4, H, W] → [B, 4, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        features = self.encoder(x)
        output = self.head(features)
        return output.squeeze(2)  # [B, 4, H, W]


# ============================================================================
# MODEL YÜKLEME
# ============================================================================

class ModelService:
    """
    Model yükleme ve tahmin servisi.
    Singleton pattern kullanır (uygulama başlangıcında bir kez yüklenir).
    """
    
    def __init__(self):
        self.unet_model = None
        self.conv3d_model = None
        self.mean_stats = None
        self.std_stats = None
        self.loaded = False
    
    def load_models(self):
        """Tüm modelleri ve normalizasyon istatistiklerini yükle."""
        if self.loaded:
            print("✓ Modeller zaten yüklü")
            return
        
        print(f"📦 Modeller yükleniyor... (Device: {DEVICE})")
        
        # 1. Mean/Std yükle
        if not MEAN_PATH.exists() or not STD_PATH.exists():
            raise FileNotFoundError(
                f"Mean/Std dosyaları bulunamadı:\n"
                f"  - {MEAN_PATH}\n"
                f"  - {STD_PATH}\n"
                f"Lütfen eğitim kodundan band_mean.npy ve band_std.npy dosyalarını "
                f"{MODEL_DIR} klasörüne kopyalayın."
            )
        
        self.mean_stats = np.load(MEAN_PATH).astype(np.float32)
        self.std_stats = np.load(STD_PATH).astype(np.float32)
        assert self.mean_stats.shape[0] == 9, "Mean 9 band olmalı"
        assert self.std_stats.shape[0] == 9, "Std 9 band olmalı"
        print(f"  ✓ Normalizasyon istatistikleri yüklendi (9 band)")
        
        # 2. UNET yükle
        if not UNET_PATH.exists():
            raise FileNotFoundError(
                f"UNET modeli bulunamadı: {UNET_PATH}\n"
                f"Lütfen eğitim kodundan unet4_best.pt dosyasını "
                f"{MODEL_DIR} klasörüne kopyalayın."
            )
        
        self.unet_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=9,
            classes=4
        ).to(DEVICE)
        
        checkpoint = torch.load(UNET_PATH, map_location=DEVICE)
        self.unet_model.load_state_dict(checkpoint["model_state_dict"])
        self.unet_model.eval()
        print(f"  ✓ UNET modeli yüklendi")
        
        # 3. Conv3D yükle
        if not CONV3D_PATH.exists():
            print(f"  ⚠ Conv3D modeli bulunamadı: {CONV3D_PATH}")
            print(f"    Temporal tahmin yapılamayacak (sadece UNET çalışacak)")
            self.conv3d_model = None
        else:
            self.conv3d_model = Tiny3D(in_channels=4, sequence_length=8).to(DEVICE)
            checkpoint = torch.load(CONV3D_PATH, map_location=DEVICE)
            self.conv3d_model.load_state_dict(checkpoint["model_state_dict"])
            self.conv3d_model.eval()
            print(f"  ✓ Conv3D modeli yüklendi")
        
        self.loaded = True
        torch.backends.cudnn.benchmark = True
        print(f"✅ Tüm modeller hazır!")
    
    def process_tile_to_softmax(self, tile_path: Path) -> np.ndarray:
        """
        Bir tile'ı UNET ile işleyip softmax çıktısı üret.
        
        Args:
            tile_path: 6-band .npy dosyası
        
        Returns:
            softmax: (4, H, W) float32 array
        """
        if not self.loaded:
            raise RuntimeError("Modeller yüklenmemiş! load_models() çağırın.")
        
        # 1. Tile'ı oku (6 band)
        six_band = np.load(tile_path).astype(np.float32)
        six_band = np.nan_to_num(six_band, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 9 banda dönüştür
        nine_band = calculate_9_bands_from_6(six_band)
        
        # 3. Normalize et
        mean_tensor = torch.from_numpy(self.mean_stats)[:, None, None].to(DEVICE)
        std_tensor = torch.from_numpy(self.std_stats)[:, None, None].to(DEVICE)
        
        x = torch.from_numpy(nine_band).to(DEVICE)
        x = (x - mean_tensor) / (std_tensor + 1e-6)
        x = x.unsqueeze(0)  # [1, 9, H, W]
        
        # 4. UNET inference
        with torch.no_grad():
            logits = self.unet_model(x)  # [1, 4, H, W]
            softmax = torch.softmax(logits, dim=1)[0]  # [4, H, W]
        
        return softmax.cpu().numpy().astype(np.float32)
    
    def predict_temporal(
        self,
        softmax_sequence: List[np.ndarray]
    ) -> np.ndarray:
        """
        8 çeyreklik softmax dizisinden t+1 tahmini yap.
        
        Args:
            softmax_sequence: List of (4, H, W) softmax arrays (8 adet)
        
        Returns:
            prediction: (4, H, W) softmax for t+1
        """
        if not self.loaded or self.conv3d_model is None:
            raise RuntimeError("Conv3D modeli yüklenmemiş!")
        
        if len(softmax_sequence) != 8:
            raise ValueError(f"8 çeyrek gerekli, {len(softmax_sequence)} geldi")
        
        # [T, 4, H, W] dizisi oluştur
        sequence = np.stack(softmax_sequence, axis=0)  # [8, 4, H, W]
        
        # Tensor'a çevir
        x = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)  # [1, 8, 4, H, W]
        
        # Conv3D inference
        with torch.no_grad():
            logits = self.conv3d_model(x)  # [1, 4, H, W]
            prediction = torch.softmax(logits, dim=1)[0]  # [4, H, W]
        
        return prediction.cpu().numpy().astype(np.float32)


# ============================================================================
# GLOBAL INSTANCE (Singleton)
# ============================================================================

# Uygulama başlangıcında bir kez oluşturulur
model_service = ModelService()


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def get_available_periods(tiles_dir: Path = TILES_DIR) -> List[Tuple[int, int]]:
    """
    Mevcut tile dosyalarından benzersiz (year, quarter) çiftlerini al.
    
    Returns:
        [(2018, 1), (2018, 2), ..., (2025, 4)]
    """
    periods = set()
    
    for tile_file in tiles_dir.glob("*.npy"):
        meta = parse_filename(tile_file.name)
        if meta:
            periods.add((meta["year"], meta["quarter"]))
    
    return sorted(list(periods))


def get_tiles_for_period(
    year: int,
    quarter: int,
    tiles_dir: Path = TILES_DIR
) -> List[Path]:
    """
    Belirli bir dönem için tüm tile dosyalarını bul.
    
    Args:
        year: 2018-2025
        quarter: 1-4
        tiles_dir: Tile klasörü
    
    Returns:
        Tile dosya yolları listesi
    """
    pattern = f"{year}_Q{quarter}_*.npy"
    return sorted(tiles_dir.glob(pattern))


def calculate_statistics_from_softmax(softmax: np.ndarray) -> Dict[str, float]:
    """
    Softmax çıktısından sınıf yüzdelerini hesapla.
    
    Args:
        softmax: (4, H, W) array
    
    Returns:
        {"green": 32.5, "gray": 60.2, "water": 7.3, "background": 0.0}
    """
    # Argmax ile en olası sınıfı bul
    prediction = softmax.argmax(axis=0)  # (H, W)
    
    total_pixels = prediction.size
    stats = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        pixel_count = (prediction == class_idx).sum()
        percentage = (pixel_count / total_pixels) * 100
        stats[class_name] = round(float(percentage), 2)
    
    return stats


def predict_period(
    year: int,
    quarter: int,
    use_temporal: bool = False,
    tiles_dir: Path = TILES_DIR
) -> Dict:
    """
    Bir dönem için tahmin yap ve istatistikleri döndür.
    
    Args:
        year: Yıl
        quarter: Çeyrek
        use_temporal: Conv3D ile temporal tahmin kullan mı?
        tiles_dir: Tile klasörü
    
    Returns:
        {
            "period": "2025 Q1",
            "statistics": {"green": 31.8, "gray": 61.1, ...},
            "method": "unet" or "temporal",
            "tile_count": 42
        }
    """
    # Modelleri yükle (henüz yüklenmediyse)
    if not model_service.loaded:
        model_service.load_models()
    
    # Tile'ları bul
    tiles = get_tiles_for_period(year, quarter, tiles_dir)
    
    if not tiles:
        raise FileNotFoundError(f"Hiç tile bulunamadı: {year} Q{quarter}")
    
    # Tüm tile'lar için tahmin yap ve birleştir
    all_predictions = []
    
    for tile_path in tiles:
        softmax = model_service.process_tile_to_softmax(tile_path)
        all_predictions.append(softmax)
    
    # Ortalama softmax hesapla
    avg_softmax = np.mean(all_predictions, axis=0)
    
    # Temporal tahmin isteniyorsa
    method = "unet"
    if use_temporal and model_service.conv3d_model is not None:
        # TODO: 8 çeyreklik sequence oluştur ve Conv3D kullan
        # Şimdilik sadece UNET kullanıyoruz
        method = "temporal (not implemented yet)"
    
    # İstatistikleri hesapla
    stats = calculate_statistics_from_softmax(avg_softmax)
    
    return {
        "period": f"{year} Q{quarter}",
        "statistics": stats,
        "method": method,
        "tile_count": len(tiles),
        "softmax": avg_softmax  # Harita için softmax/softmap çıktısı
    }


# ============================================================================
# MAIN (Test için)
# ============================================================================

if __name__ == "__main__":
    # Test
    print("🧪 Model Service Test\n")
    
    # Modelleri yükle
    try:
        model_service.load_models()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        exit(1)
    
    # Mevcut dönemleri listele
    periods = get_available_periods()
    print(f"\n📅 Mevcut dönemler: {len(periods)}")
    if periods:
        print(f"  İlk: {periods[0]}")
        print(f"  Son: {periods[-1]}")
    
    # Örnek tahmin
    if periods:
        year, quarter = periods[-1]
        print(f"\n🔮 Tahmin yapılıyor: {year} Q{quarter}")
        result = predict_period(year, quarter)
        print(f"\n✅ Sonuç:")
        print(f"  Period: {result['period']}")
        print(f"  Method: {result['method']}")
        print(f"  Tiles: {result['tile_count']}")
        print(f"  İstatistikler:")
        for cls, pct in result['statistics'].items():
            print(f"    {cls}: {pct}%")
