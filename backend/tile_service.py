"""
Tile Service - Local tile'ları servise et
==========================================
Backend'deki tiles/images/ klasöründen .npy patch dosyalarını okur,
renklendirir ve Leaflet'e PNG olarak serve eder.

Dosya yapısı:
backend/tiles/
  images/
    2018_Q1_00000_00000.npy  → row=0, col=0
    2018_Q1_00000_00256.npy  → row=0, col=256
    ...

Naming convention:
{year}_Q{quarter}_{row:05d}_{col:05d}.npy
Örnek: 2018_Q1_00000_00256.npy

Görselleştirmeler:
- NDVI: Yeşil alan (vegetation)
- NDWI: Su alanı (water)
- NDBI: Beton/yapı (built-up)
- RGB: Doğal görünüm
"""

import io
import math
from pathlib import Path
from typing import Optional, Tuple
from functools import lru_cache

import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.responses import StreamingResponse


# ============================================================================
# YAPILANDIRMA
# ============================================================================

TILES_DIR = Path(__file__).parent / "tiles"
IMAGES_DIR = TILES_DIR / "images"


# ============================================================================
# TILE <-> PATCH DÖNÜŞÜM
# ============================================================================

def tile_to_patch_coords(z, x, y):
    """
    Leaflet tile koordinatlarını patch row/col'a çevir.
    
    Patch grid: 18 rows × 40 cols (0-4352, 0-9984, stride 256)
    GEE AOI: [28.62, 40.75, 29.56, 41.18] (lon_min, lat_min, lon_max, lat_max)
    
    Args:
        z: Zoom level
        x, y: Leaflet tile coordinates
    
    Returns:
        (row, col): Patch koordinatları veya (None, None)
    """
    # GEE AOI boundaries
    LON_MIN, LAT_MIN = 28.62, 40.75
    LON_MAX, LAT_MAX = 29.56, 41.18
    
    # Tile merkez noktasının lat/lon değerlerini hesapla
    n = 2 ** z
    lon_deg = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat_deg = math.degrees(lat_rad)
    
    # AOI dışında mı kontrol et
    if not (LON_MIN <= lon_deg <= LON_MAX and LAT_MIN <= lat_deg <= LAT_MAX):
        return None, None
    
    # Normalize et (0-1 arası)
    lon_norm = (lon_deg - LON_MIN) / (LON_MAX - LON_MIN)
    lat_norm = (lat_deg - LAT_MIN) / (LAT_MAX - LAT_MIN)
    
    # Patch index hesapla (floor ile)
    # 18 row: index 0-17
    # 40 col: index 0-39
    row_index = int((1.0 - lat_norm) * 18)
    col_index = int(lon_norm * 40)
    
    # Sınırları zorla
    row_index = max(0, min(17, row_index))
    col_index = max(0, min(39, col_index))
    
    # Patch koordinatları
    patch_row = row_index * 256
    patch_col = col_index * 256
    
    return patch_row, patch_col


def parse_patch_filename(year: int, quarter: int, row: int, col: int) -> str:
    """
    Patch parametrelerinden dosya adı oluştur.
    
    Args:
        year: Yıl (2018-2025)
        quarter: Çeyrek (1-4)
        row: Patch row koordinatı
        col: Patch col koordinatı
    
    Returns:
        str: Dosya adı (örn: "2018_Q1_00000_00256.npy")
    """
    return f"{year}_Q{quarter}_{row:05d}_{col:05d}.npy"


# ============================================================================
# PATCH YÜKLEME
# ============================================================================

@lru_cache(maxsize=200)
def load_patch_from_disk(filepath: Path) -> Optional[np.ndarray]:
    """
    Local disk'ten patch yükle (cache'li).
    
    Args:
        filepath: Patch dosyasının tam yolu
    
    Returns:
        np.ndarray: Patch data veya None
    """
    if not filepath.exists():
        return None
    
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"❌ Patch okuma hatası ({filepath.name}): {e}")
        return None


# ============================================================================
# GÖRSELLEŞTİRME
# ============================================================================




def visualize_ndvi(patch_data: np.ndarray) -> Image.Image:
    """NDVI (Yeşil Alan): Kahverengi → Sarı → Yeşil"""
    red = np.nan_to_num(patch_data[2, :, :], nan=0.0).astype(np.float32)
    nir = np.nan_to_num(patch_data[3, :, :], nan=0.0).astype(np.float32)
    
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi_norm = (ndvi + 1) / 2  # 0-1 arası
    
    # Canlı ve net gradient: Kahverengi/Kırmızı (0) → Sarı (0.5) → Koyu Yeşil (1)
    # NDBI ile aynı mantık, ama daha canlı renkler
    r = np.where(ndvi_norm < 0.5,
                 180 + (255 - 180) * (ndvi_norm * 2),      # 180→255 (kahverengi→sarı)
                 255 - (255 - 34) * ((ndvi_norm - 0.5) * 2))  # 255→34 (sarı→koyu yeşil)
    g = np.where(ndvi_norm < 0.5,
                 82 + (255 - 82) * (ndvi_norm * 2),        # 82→255 (kahverengi→sarı)
                 255 - (255 - 139) * ((ndvi_norm - 0.5) * 2)) # 255→139 (sarı→koyu yeşil)
    b = np.where(ndvi_norm < 0.5,
                 50 + (50 - 50) * (ndvi_norm * 2),         # 50→50 (kahverengi ton)
                 50 - (50 - 34) * ((ndvi_norm - 0.5) * 2))    # 50→34 (yeşil ton)
    
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def visualize_ndwi(patch_data: np.ndarray) -> Image.Image:
    """NDWI (Su Alanı): Bej → Cyan → Mavi"""
    green = np.nan_to_num(patch_data[1, :, :], nan=0.0).astype(np.float32)
    nir = np.nan_to_num(patch_data[3, :, :], nan=0.0).astype(np.float32)
    
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi = np.clip(ndwi, -1, 1)
    ndwi_norm = (ndwi + 1) / 2  # 0-1 arası
    
    # NDBI gibi basit iki aşamalı gradient
    # 0 (bej/kuru) → 0.5 (cyan) → 1 (mavi/su)
    r = np.where(ndwi_norm < 0.5,
                 210 - (210 - 100) * (ndwi_norm * 2),      # 210→100
                 100 - (100 - 30) * ((ndwi_norm - 0.5) * 2))  # 100→30
    g = np.where(ndwi_norm < 0.5,
                 180 + (200 - 180) * (ndwi_norm * 2),      # 180→200
                 200 - (200 - 100) * ((ndwi_norm - 0.5) * 2)) # 200→100
    b = np.where(ndwi_norm < 0.5,
                 140 + (220 - 140) * (ndwi_norm * 2),      # 140→220
                 220 + (255 - 220) * ((ndwi_norm - 0.5) * 2)) # 220→255
    
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def visualize_ndbi(patch_data: np.ndarray) -> Image.Image:
    """NDBI (Beton/Yapı): Yeşil-Mavi → Gri → Turuncu-Kırmızı"""
    swir1 = np.nan_to_num(patch_data[4, :, :], nan=0.0).astype(np.float32)
    nir = np.nan_to_num(patch_data[3, :, :], nan=0.0).astype(np.float32)
    
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
    ndbi = np.clip(ndbi, -1, 1)
    ndbi_norm = (ndbi + 1) / 2  # 0-1 arası
    
    # Gradient: Yeşil-mavi (doğal, 0) → Gri (0.5) → Kırmızı (kentsel, 1)
    r = np.where(ndbi_norm < 0.5,
                 50 + (180 - 50) * (ndbi_norm * 2),        # 50→180
                 180 + (220 - 180) * ((ndbi_norm - 0.5) * 2))  # 180→220
    g = np.where(ndbi_norm < 0.5,
                 120 + (180 - 120) * (ndbi_norm * 2),      # 120→180
                 180 - (180 - 80) * ((ndbi_norm - 0.5) * 2))   # 180→80
    b = np.where(ndbi_norm < 0.5,
                 180 + (180 - 180) * (ndbi_norm * 2),      # 180→180
                 180 - (180 - 60) * ((ndbi_norm - 0.5) * 2))   # 180→60
    
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def visualize_rgb(patch_data: np.ndarray) -> Image.Image:
    """True Color RGB: Doğal renkler"""
    red = np.nan_to_num(patch_data[2, :, :], nan=0.0)
    green = np.nan_to_num(patch_data[1, :, :], nan=0.0)
    blue = np.nan_to_num(patch_data[0, :, :], nan=0.0)
    
    # Global normalizasyon sabitleri (Sentinel-2 için tipik değerler)
    # Bu değerler tüm patch'ler için aynı, böylece tutarlı renkler elde ederiz
    def normalize(band):
        # 0-3000 aralığını 0-255'e normalize et (Sentinel-2 reflectance değerleri)
        band = np.clip(band, 0, 3000)
        band = band / 3000.0
        return (band * 255).astype(np.uint8)
    
    rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
    return Image.fromarray(rgb, mode='RGB')


# ============================================================================
# API ENDPOINT FONKSİYONU
# ============================================================================

def get_tile_response(
    year: int,
    quarter: int,
    index: str,
    z: int,
    x: int,
    y: int
) -> StreamingResponse:
    """
    Tile isteği için PNG response döndür.
    
    Args:
        year: Yıl (2018-2025)
        quarter: Çeyrek (1-4)
        index: Görselleştirme tipi
            - 'ndvi': Yeşil alan
            - 'ndwi': Su alanı
            - 'ndbi': Beton/Yapı
            - 'rgb': Doğal görünüm
        z: Zoom level
        x: X tile koordinatı
        y: Y tile koordinatı
    
    Returns:
        StreamingResponse: PNG tile (256x256)
    """
    # Tile → Patch koordinat dönüşümü
    row, col = tile_to_patch_coords(z, x, y)
    
    # Kapsam dışı tile için boş döndür
    if row is None or col is None:
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')
    
    # Dosya adı oluştur
    filename = parse_patch_filename(year, quarter, row, col)
    
    # Image patch yükle
    filepath = IMAGES_DIR / filename
    patch_data = load_patch_from_disk(filepath)
    
    if patch_data is None:
        # Dosya bulunamadı - boş tile
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_bytes = io.BytesIO()
        empty_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type='image/png')
    
    # Index'e göre görselleştir
    try:
        if index == 'ndvi':
            tile_image = visualize_ndvi(patch_data)
        elif index == 'ndwi':
            tile_image = visualize_ndwi(patch_data)
        elif index == 'ndbi':
            tile_image = visualize_ndbi(patch_data)
        elif index == 'rgb':
            tile_image = visualize_rgb(patch_data)
        else:
            # Default: RGB
            tile_image = visualize_rgb(patch_data)
    
    except Exception as e:
        print(f"❌ Görselleştirme hatası ({index}, {filename}): {e}")
        # Hata durumunda boş tile
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_bytes = io.BytesIO()
        empty_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type='image/png')
    
    # PNG'ye çevir ve döndür
    img_bytes = io.BytesIO()
    tile_image.save(img_bytes, format='PNG', optimize=True)
    img_bytes.seek(0)
    
    return StreamingResponse(
        img_bytes,
        media_type='image/png',
        headers={
            'Cache-Control': 'public, max-age=86400',  # 24 saat cache
            'Access-Control-Allow-Origin': '*',
        }
    )
