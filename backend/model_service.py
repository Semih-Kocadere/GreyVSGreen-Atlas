"""
Model Service - UNET + Conv3D Temporal Prediction
==================================================
Loads trained models and performs predictions.

Models:
- UNET: 4-class segmentation (background, green, gray, water)
    Input: 9 bands (B02-B12 + NDVI, NDWI, NDBI)
    Output: 4-class softmax [C, H, W]

- Conv3D: Temporal prediction (8 quarters → t+1)
    Input: [T, 4, H, W] softmax sequence
    Output: [4, H, W] prediction for t+1

Data Flow:
1. Read 6 bands from tile (.npy)
2. Calculate NDVI, NDWI, NDBI → 9 bands
3. Normalize (mean/std)
4. UNET → softmax
5. Build sequence (8 quarters)
6. Conv3D → t+1 prediction
7. Calculate and return statistics
"""

import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ============================================================================
# CONFIGURATION
# ============================================================================

# Tile and model file paths
TILES_DIR = Path(__file__).parent / "tiles" / "images"
MODEL_DIR = Path(__file__).parent / "models"
MEAN_PATH = MODEL_DIR / "band_mean.npy"
STD_PATH = MODEL_DIR / "band_std.npy"
UNET_PATH = MODEL_DIR / "unet4_best.pt"
CONV3D_PATH = MODEL_DIR / "temporal3d_tplus1.pt"

# Device (use GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filename regex: 2025_Q1_00000_00256.npy
FILENAME_PATTERN = re.compile(r"(?P<year>\d{4})_Q(?P<quarter>[1-4])_(?P<row>\d{5})_(?P<col>\d{5})\.npy")

# Class names
CLASS_NAMES = ["background", "green", "gray", "water"]


# ============================================================================
# DERIVING 9 BANDS AND FILENAME PARSING
# ============================================================================

def calculate_spectral_indices(six_band_data: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI, NDWI, NDBI from 6 bands.
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
    Create 9 bands from 6 bands: [B02-B12, NDVI, NDWI, NDBI]
    Input: (6, H, W)
    Output: (9, H, W)
    """
    indices = calculate_spectral_indices(six_band_data)
    return np.concatenate([six_band_data, indices], axis=0)


def parse_filename(filename: str) -> Optional[Dict[str, int]]:
    """
    Extract metadata from filename.
    Args:
        filename: "2025_Q1_00000_00256.npy"
    Returns:
        {"year": 2025, "quarter": 1, "row": 0, "col": 256}
        or None if not matched
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
    """Unique key for tile coordinates."""
    return (row, col)


# ============================================================================
# CONV3D MODEL DEFINITION (from training code)
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
# MODEL LOADING
# ============================================================================

class ModelService:
    """
    Model loading and prediction service.
    Uses singleton pattern (loaded once at app startup).
    """
    
    def __init__(self):
        self.unet_model = None
        self.conv3d_model = None
        self.conv3d_model_tplus4 = None
        self.conv3d_model_tplus8 = None
        self.mean_stats = None
        self.std_stats = None
        self.loaded = False
    
    def load_models(self):
        """Load all models and normalization statistics."""
        if self.loaded:
            print("✓ Modeller zaten yüklü")
            return
        
        print(f"📦 Modeller yükleniyor... (Device: {DEVICE})")
        
        # 1. Load Mean/Std
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
        
        # 2. Load UNET
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
        
        # 3. Load Conv3D models
        # t+1
        if not CONV3D_PATH.exists():
            print(f"  ⚠ Conv3D t+1 modeli bulunamadı: {CONV3D_PATH}")
            self.conv3d_model = None
        else:
            self.conv3d_model = Tiny3D(in_channels=4, sequence_length=8).to(DEVICE)
            checkpoint = torch.load(CONV3D_PATH, map_location=DEVICE)
            self.conv3d_model.load_state_dict(checkpoint["model_state_dict"])
            self.conv3d_model.eval()
            print(f"  ✓ Conv3D t+1 modeli yüklendi")
        # t+4
        conv3d_tplus4_path = MODEL_DIR / "temporal3d_tplus4.pt"
        if not conv3d_tplus4_path.exists():
            print(f"  ⚠ Conv3D t+4 modeli bulunamadı: {conv3d_tplus4_path}")
            self.conv3d_model_tplus4 = None
        else:
            self.conv3d_model_tplus4 = Tiny3D(in_channels=4, sequence_length=12).to(DEVICE)
            checkpoint = torch.load(conv3d_tplus4_path, map_location=DEVICE)
            self.conv3d_model_tplus4.load_state_dict(checkpoint["model_state_dict"])
            self.conv3d_model_tplus4.eval()
            print(f"  ✓ Conv3D t+4 modeli yüklendi")
        # t+8
        conv3d_tplus8_path = MODEL_DIR / "temporal3d_tplus8.pt"
        if not conv3d_tplus8_path.exists():
            print(f"  ⚠ Conv3D t+8 modeli bulunamadı: {conv3d_tplus8_path}")
            self.conv3d_model_tplus8 = None
        else:
            self.conv3d_model_tplus8 = Tiny3D(in_channels=4, sequence_length=16).to(DEVICE)
            checkpoint = torch.load(conv3d_tplus8_path, map_location=DEVICE)
            self.conv3d_model_tplus8.load_state_dict(checkpoint["model_state_dict"])
            self.conv3d_model_tplus8.eval()
            print(f"  ✓ Conv3D t+8 modeli yüklendi")
        
        self.loaded = True
        torch.backends.cudnn.benchmark = True
        print(f"✅ Tüm modeller yüklendi!")
    
    def process_tile_to_softmax(self, tile_path: Path) -> np.ndarray:
        """
        Process a tile with UNET and produce softmax output.
        Args:
            tile_path: 6-band .npy file
        Returns:
            softmax: (4, H, W) float32 array
        """
        if not self.loaded:
            raise RuntimeError("Modeller yüklenmemiş! load_models() çağırın.")
        
        # 1. Read tile (6 bands)
        six_band = np.load(tile_path).astype(np.float32)
        six_band = np.nan_to_num(six_band, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Convert to 9 bands
        nine_band = calculate_9_bands_from_6(six_band)
        
        # 3. Normalize
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
    
    def predict_temporal(self, softmax_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Predict t+1 from 8-quarter softmax sequence.
        Args:
            softmax_sequence: List of (4, H, W) softmax arrays (8 items)
        Returns:
            prediction: (4, H, W) softmax for t+1
        """
        if not self.loaded or self.conv3d_model is None:
            raise RuntimeError("Conv3D t+1 modeli yüklenmemiş!")
        if len(softmax_sequence) != 8:
            raise ValueError(f"8 çeyrek gerekli, {len(softmax_sequence)} geldi")
        sequence = np.stack(softmax_sequence, axis=0)  # [8, 4, H, W]
        x = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)  # [1, 8, 4, H, W]
        with torch.no_grad():
            logits = self.conv3d_model(x)  # [1, 4, H, W]
            prediction = torch.softmax(logits, dim=1)[0]  # [4, H, W]
        return prediction.cpu().numpy().astype(np.float32)

    def predict_temporal_tplus4(self, softmax_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Predict t+4 from 12-quarter softmax sequence.
        Args:
            softmax_sequence: List of (4, H, W) softmax arrays (12 items)
        Returns:
            prediction: (4, H, W) softmax for t+4
        """
        if not self.loaded or self.conv3d_model_tplus4 is None:
            raise RuntimeError("Conv3D t+4 modeli yüklenmemiş!")
        if len(softmax_sequence) != 12:
            raise ValueError(f"12 çeyrek gerekli, {len(softmax_sequence)} geldi")
        sequence = np.stack(softmax_sequence, axis=0)  # [12, 4, H, W]
        x = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)  # [1, 12, 4, H, W]
        with torch.no_grad():
            logits = self.conv3d_model_tplus4(x)  # [1, 4, H, W]
            prediction = torch.softmax(logits, dim=1)[0]  # [4, H, W]
        return prediction.cpu().numpy().astype(np.float32)

    def predict_temporal_tplus8(self, softmax_sequence: List[np.ndarray]) -> np.ndarray:
        """
        Predict t+8 from 16-quarter softmax sequence.
        Args:
            softmax_sequence: List of (4, H, W) softmax arrays (16 items)
        Returns:
            prediction: (4, H, W) softmax for t+8
        """
        if not self.loaded or self.conv3d_model_tplus8 is None:
            raise RuntimeError("Conv3D t+8 modeli yüklenmemiş!")
        if len(softmax_sequence) != 16:
            raise ValueError(f"16 çeyrek gerekli, {len(softmax_sequence)} geldi")
        sequence = np.stack(softmax_sequence, axis=0)  # [16, 4, H, W]
        x = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)  # [1, 16, 4, H, W]
        with torch.no_grad():
            logits = self.conv3d_model_tplus8(x)  # [1, 4, H, W]
            prediction = torch.softmax(logits, dim=1)[0]  # [4, H, W]
        return prediction.cpu().numpy().astype(np.float32)


# ============================================================================
# GLOBAL INSTANCE (Singleton)
# ============================================================================

# Created once at app startup
model_service = ModelService()


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def get_available_periods(tiles_dir: Path = TILES_DIR) -> List[Tuple[int, int]]:
    """
    Get unique (year, quarter) pairs from available tile files.
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
    Find all tile files for a specific period.
    Args:
        year: 2018-2025
        quarter: 1-4
        tiles_dir: Tile folder
    Returns:
        List of tile file paths
    """
    pattern = f"{year}_Q{quarter}_*.npy"
    return sorted(tiles_dir.glob(pattern))


def calculate_statistics_from_softmax(softmax: np.ndarray) -> Dict[str, float]:
    """
    Calculate class percentages from softmax output.
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
    Make prediction for a period and return statistics.
    Args:
        year: Year
        quarter: Quarter
        use_temporal: Use temporal prediction with Conv3D?
        tiles_dir: Tile folder
    Returns:
        {
            "period": "2025 Q1",
            "statistics": {"green": 31.8, "gray": 61.1, ...},
            "method": "unet" or "temporal",
            "tile_count": 42
        }
    """
    # Load models (if not loaded yet)
    if not model_service.loaded:
        model_service.load_models()
    
    # Find tiles
    tiles = get_tiles_for_period(year, quarter, tiles_dir)
    
    if not tiles:
        raise FileNotFoundError(f"Hiç tile bulunamadı: {year} Q{quarter}")
    
    # Make predictions for all tiles and combine
    all_predictions = []
    
    for tile_path in tiles:
        softmax = model_service.process_tile_to_softmax(tile_path)
        all_predictions.append(softmax)
    
    # Calculate average softmax
    avg_softmax = np.mean(all_predictions, axis=0)
    
    # If temporal prediction is requested
    method = "unet"
    if use_temporal and model_service.conv3d_model is not None:
        # TODO: 8 çeyreklik sequence oluştur ve Conv3D kullan
        # Şimdilik sadece UNET kullanıyoruz
        method = "temporal (not implemented yet)"
    
    # Calculate statistics
    stats = calculate_statistics_from_softmax(avg_softmax)
    
    return {
        "period": f"{year} Q{quarter}",
        "statistics": stats,
        "method": method,
        "tile_count": len(tiles),
        "softmax": avg_softmax  # Harita için softmax/softmap çıktısı
    }
