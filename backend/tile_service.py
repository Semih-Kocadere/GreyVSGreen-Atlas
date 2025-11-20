"""
Tile Service - Serve local tiles
==========================================
Reads .npy patch files from backend tiles/images/ folder,
colors them and serves as PNG for Leaflet.

File structure:
backend/tiles/
    images/
        2018_Q1_00000_00000.npy  → row=0, col=0
        2018_Q1_00000_00256.npy  → row=0, col=256
        ...

Naming convention:
{year}_Q{quarter}_{row:05d}_{col:05d}.npy
Example: 2018_Q1_00000_00256.npy

Visualizations:
- NDVI: Green area (vegetation)
- NDWI: Water area
- NDBI: Built-up area
- RGB: Natural view
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
# CONFIGURATION
# ============================================================================

TILES_DIR = Path(__file__).parent / "tiles"
IMAGES_DIR = TILES_DIR / "images"


# ============================================================================
# TILE <-> PATCH CONVERSION
# ============================================================================

def tile_to_patch_coords(z, x, y):
    """
    Convert Leaflet tile coordinates to patch row/col.
    
    Patch grid: 18 rows × 40 cols (0-4352, 0-9984, stride 256)
    GEE AOI: [28.62, 40.75, 29.56, 41.18] (lon_min, lat_min, lon_max, lat_max)
    
    Args:
        z: Zoom level
        x, y: Leaflet tile coordinates
    
    Returns:
        (row, col): Patch coordinates or (None, None)
    """
    # GEE AOI boundaries
    LON_MIN, LAT_MIN = 28.62, 40.75
    LON_MAX, LAT_MAX = 29.56, 41.18
    
    # Calculate lat/lon of tile center point
    n = 2 ** z
    lon_deg = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat_deg = math.degrees(lat_rad)
    
    # Check if outside AOI
    if not (LON_MIN <= lon_deg <= LON_MAX and LAT_MIN <= lat_deg <= LAT_MAX):
        return None, None
    
    # Normalize (between 0-1)
    lon_norm = (lon_deg - LON_MIN) / (LON_MAX - LON_MIN)
    lat_norm = (lat_deg - LAT_MIN) / (LAT_MAX - LAT_MIN)
    
    # Calculate patch index (using floor)
    # 18 rows: index 0-17
    # 40 cols: index 0-39
    row_index = int((1.0 - lat_norm) * 18)
    col_index = int(lon_norm * 40)
    
    # Clamp boundaries
    row_index = max(0, min(17, row_index))
    col_index = max(0, min(39, col_index))
    
    # Patch coordinates
    patch_row = row_index * 256
    patch_col = col_index * 256
    
    return patch_row, patch_col


def parse_patch_filename(year: int, quarter: int, row: int, col: int) -> str:
    """
    Create filename from patch parameters.
    
    Args:
        year: Year (2018-2025)
        quarter: Quarter (1-4)
        row: Patch row coordinate
        col: Patch col coordinate
    
    Returns:
        str: Filename (e.g. "2018_Q1_00000_00256.npy")
    """
    return f"{year}_Q{quarter}_{row:05d}_{col:05d}.npy"


# ============================================================================
# PATCH LOADING
# ============================================================================

@lru_cache(maxsize=200)
def load_patch_from_disk(filepath: Path) -> Optional[np.ndarray]:
    """
    Load patch from local disk (cached).
    
    Args:
        filepath: Full path to patch file
    
    Returns:
        np.ndarray: Patch data or None
    """
    if not filepath.exists():
        return None
    
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"❌ Patch read error ({filepath.name}): {e}")
        return None


# ============================================================================
# VISUALIZATION
# ============================================================================




def visualize_ndvi(patch_data: np.ndarray) -> Image.Image:
    """NDVI (Green Area): Brown → Yellow → Green"""
    red = np.nan_to_num(patch_data[2, :, :], nan=0.0).astype(np.float32)
    nir = np.nan_to_num(patch_data[3, :, :], nan=0.0).astype(np.float32)
    
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi_norm = (ndvi + 1) / 2  # 0-1 arası
    
    # Vivid and clear gradient: Brown/Red (0) → Yellow (0.5) → Dark Green (1)
    # Same logic as NDBI, but more vivid colors
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
    """NDWI (Water Area): Beige → Cyan → Blue"""
    green = np.nan_to_num(patch_data[1, :, :], nan=0.0).astype(np.float32)
    nir = np.nan_to_num(patch_data[3, :, :], nan=0.0).astype(np.float32)
    
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndwi = np.clip(ndwi, -1, 1)
    ndwi_norm = (ndwi + 1) / 2  # 0-1 arası
    
    # Simple two-step gradient like NDBI
    # 0 (beige/dry) → 0.5 (cyan) → 1 (blue/water)
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
    """NDBI (Built-up): Green-Blue → Gray → Orange-Red"""
    swir1 = np.nan_to_num(patch_data[4, :, :], nan=0.0).astype(np.float32)
    nir = np.nan_to_num(patch_data[3, :, :], nan=0.0).astype(np.float32)
    
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
    ndbi = np.clip(ndbi, -1, 1)
    ndbi_norm = (ndbi + 1) / 2  # 0-1 arası
    
    # Gradient: Green-blue (natural, 0) → Gray (0.5) → Red (urban, 1)
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
    """True Color RGB: Natural colors"""
    red = np.nan_to_num(patch_data[2, :, :], nan=0.0)
    green = np.nan_to_num(patch_data[1, :, :], nan=0.0)
    blue = np.nan_to_num(patch_data[0, :, :], nan=0.0)
    
    # Global normalization constants (typical for Sentinel-2)
    # These values are the same for all patches, so we get consistent colors
    def normalize(band):
        # Normalize 0-3000 range to 0-255 (Sentinel-2 reflectance values)
        band = np.clip(band, 0, 3000)
        band = band / 3000.0
        return (band * 255).astype(np.uint8)
    
    rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=-1)
    return Image.fromarray(rgb, mode='RGB')


# ============================================================================
# API ENDPOINT FUNCTION
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
    Return PNG response for tile request.
    
    Args:
        year: Year (2018-2025)
        quarter: Quarter (1-4)
        index: Visualization type
            - 'ndvi': Green area
            - 'ndwi': Water area
            - 'ndbi': Built-up
            - 'rgb': Natural view
        z: Zoom level
        x: X tile coordinate
        y: Y tile coordinate
    
    Returns:
        StreamingResponse: PNG tile (256x256)
    """
    # Tile → Patch coordinate conversion
    row, col = tile_to_patch_coords(z, x, y)
    
    # Return empty for out-of-bounds tile
    if row is None or col is None:
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        empty_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type='image/png')
    
    # Create filename
    filename = parse_patch_filename(year, quarter, row, col)
    
    # Load image patch
    filepath = IMAGES_DIR / filename
    patch_data = load_patch_from_disk(filepath)
    
    if patch_data is None:
        # File not found - return empty tile
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_bytes = io.BytesIO()
        empty_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type='image/png')
    
    # Visualize according to index
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
        print(f"❌ Visualization error ({index}, {filename}): {e}")
        # In case of error, return empty tile
        empty_img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_bytes = io.BytesIO()
        empty_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type='image/png')
    
    # Convert to PNG and return
    img_bytes = io.BytesIO()
    tile_image.save(img_bytes, format='PNG', optimize=True)
    img_bytes.seek(0)
    
    return StreamingResponse(
        img_bytes,
        media_type='image/png',
        headers={
            'Cache-Control': 'public, max-age=86400',  # 24 hour cache
            'Access-Control-Allow-Origin': '*',
        }
    )
