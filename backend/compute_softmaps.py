"""
Batch Softmax Computation Script
=================================
Tüm tile'lar için önceden softmax hesaplaması yapar ve kaydeder.

Kullanım:
    python compute_softmaps.py --start-year 2018 --end-year 2025
    python compute_softmaps.py --year 2024 --quarter 3
    python compute_softmaps.py --workers 4 --device cuda

Çıktı Yapısı:
    backend/softmaps/
        2018_Q1_00000_00000.npy  → [4, H, W] float32 softmax
        2018_Q1_00000_00256.npy
        ...
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional
import time

import numpy as np
import torch
from tqdm import tqdm

# Model servisi modüllerini import et
from model_service import (
    ModelService,
    TILES_DIR,
    MODEL_DIR,
    DEVICE,
    parse_filename,
    get_tiles_for_period,
    get_available_periods
)


# ============================================================================
# YAPILANDIRMA
# ============================================================================

SOFTMAPS_DIR = Path(__file__).parent / "softmaps"
SOFTMAPS_DIR.mkdir(exist_ok=True)

# İşlem ayarları
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 4


# ============================================================================
# BATCH PROCESSING FONKSIYONLARI
# ============================================================================

def softmap_path_from_tile(tile_path: Path) -> Path:
    """
    Tile dosyasından softmap çıktı yolunu oluştur.
    
    Args:
        tile_path: 2025_Q1_00000_00256.npy (tiles/images/)
    
    Returns:
        softmaps/2025_Q1_00000_00256.npy
    """
    return SOFTMAPS_DIR / tile_path.name


def is_softmap_computed(tile_path: Path) -> bool:
    """Softmap daha önce hesaplanmış mı kontrol et."""
    return softmap_path_from_tile(tile_path).exists()


def compute_single_softmap(
    tile_path: Path,
    model_service: ModelService,
    force: bool = False
) -> Optional[Path]:
    """
    Tek bir tile için softmap hesapla ve kaydet.
    
    Args:
        tile_path: Input tile path
        model_service: Yüklenmiş model servisi
        force: Var olan dosyaları da yeniden hesapla
    
    Returns:
        Çıktı dosya yolu veya None (skip edilirse)
    """
    output_path = softmap_path_from_tile(tile_path)
    
    # Zaten varsa ve force değilse atla
    if output_path.exists() and not force:
        return None
    
    try:
        # Softmax hesapla
        softmax = model_service.process_tile_to_softmax(tile_path)
        
        # Kaydet
        np.save(output_path, softmax)
        
        return output_path
        
    except Exception as e:
        print(f"❌ Hata ({tile_path.name}): {e}")
        return None


def compute_batch_softmaps(
    tile_paths: List[Path],
    model_service: ModelService,
    force: bool = False,
    show_progress: bool = True
) -> Tuple[int, int]:
    """
    Bir tile listesi için softmap'leri toplu hesapla.
    
    Args:
        tile_paths: Tile dosya yolları
        model_service: Model servisi
        force: Yeniden hesaplama
        show_progress: Progress bar göster
    
    Returns:
        (computed_count, skipped_count)
    """
    computed = 0
    skipped = 0
    
    iterator = tqdm(tile_paths, desc="Computing softmaps") if show_progress else tile_paths
    
    for tile_path in iterator:
        result = compute_single_softmap(tile_path, model_service, force)
        
        if result is not None:
            computed += 1
        else:
            skipped += 1
    
    return computed, skipped


def compute_period_softmaps(
    year: int,
    quarter: int,
    model_service: ModelService,
    force: bool = False
) -> Tuple[int, int]:
    """
    Belirli bir dönem için tüm tile'ların softmap'lerini hesapla.
    
    Args:
        year: Yıl
        quarter: Çeyrek
        model_service: Model servisi
        force: Yeniden hesaplama
    
    Returns:
        (computed, skipped)
    """
    print(f"\n📅 İşleniyor: {year} Q{quarter}")
    
    # Dönem için tile'ları bul
    tiles = get_tiles_for_period(year, quarter, TILES_DIR)
    
    if not tiles:
        print(f"  ⚠️  Tile bulunamadı")
        return 0, 0
    
    print(f"  📦 Toplam tile: {len(tiles)}")
    
    # Softmap'leri hesapla
    computed, skipped = compute_batch_softmaps(tiles, model_service, force)
    
    print(f"  ✅ Hesaplanan: {computed}")
    print(f"  ⏭️  Atlanan: {skipped}")
    
    return computed, skipped


# ============================================================================
# PARALEL İŞLEME (Opsiyonel)
# ============================================================================

def worker_process(
    tile_paths: List[Path],
    worker_id: int,
    force: bool
):
    """
    Paralel işleme için worker fonksiyonu.
    
    NOT: Her worker kendi ModelService instance'ını oluşturur.
    """
    # Worker için ayrı model servisi
    model_service = ModelService()
    model_service.load_models()
    
    print(f"Worker {worker_id}: {len(tile_paths)} tile işleniyor")
    
    computed, skipped = compute_batch_softmaps(
        tile_paths,
        model_service,
        force,
        show_progress=False
    )
    
    return computed, skipped


def compute_parallel(
    tile_paths: List[Path],
    num_workers: int = 4,
    force: bool = False
) -> Tuple[int, int]:
    """
    Çoklu işlemci kullanarak paralel softmap hesaplama.
    
    Args:
        tile_paths: Tüm tile yolları
        num_workers: Paralel worker sayısı
        force: Yeniden hesaplama
    
    Returns:
        (total_computed, total_skipped)
    """
    # Tile'ları worker'lara dağıt
    chunk_size = len(tile_paths) // num_workers
    chunks = [
        tile_paths[i:i + chunk_size]
        for i in range(0, len(tile_paths), chunk_size)
    ]
    
    # Pool oluştur ve çalıştır
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            worker_process,
            [(chunk, i, force) for i, chunk in enumerate(chunks)]
        )
    
    # Sonuçları topla
    total_computed = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)
    
    return total_computed, total_skipped


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tüm tile'lar için softmax haritalarını önceden hesapla"
    )
    
    # Dönem seçimi
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Başlangıç yılı (varsayılan: 2018)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Bitiş yılı (varsayılan: 2025)"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Sadece belirli bir yıl (opsiyonel)"
    )
    parser.add_argument(
        "--quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Sadece belirli bir çeyrek (opsiyonel)"
    )
    
    # İşlem ayarları
    parser.add_argument(
        "--force",
        action="store_true",
        help="Var olan softmap'leri yeniden hesapla"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Paralel worker sayısı (varsayılan: 1, tek thread)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Hesaplama cihazı (varsayılan: auto)"
    )
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 70)
    print("🧠 BATCH SOFTMAX COMPUTATION")
    print("=" * 70)
    
    # Device ayarla
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Device: {device}")
    print(f"📁 Tiles: {TILES_DIR}")
    print(f"💾 Output: {SOFTMAPS_DIR}")
    print(f"👷 Workers: {args.workers}")
    print()
    
    # Model servisini yükle
    print("📦 Modeller yükleniyor...")
    model_service = ModelService()
    
    try:
        model_service.load_models()
    except FileNotFoundError as e:
        print(f"❌ Model dosyaları bulunamadı: {e}")
        print(f"   Lütfen {MODEL_DIR} klasörüne model dosyalarını ekleyin:")
        print(f"   - unet4_best.pt")
        print(f"   - band_mean.npy")
        print(f"   - band_std.npy")
        return 1
    
    # Dönemleri belirle
    if args.year and args.quarter:
        # Tek dönem
        periods = [(args.year, args.quarter)]
    elif args.year:
        # Bir yılın tüm çeyrekleri
        periods = [(args.year, q) for q in range(1, 5)]
    else:
        # Yıl aralığı
        periods = [
            (y, q)
            for y in range(args.start_year, args.end_year + 1)
            for q in range(1, 5)
        ]
    
    print(f"📅 Toplam dönem: {len(periods)}")
    print()
    
    # İşleme başla
    start_time = time.time()
    total_computed = 0
    total_skipped = 0
    
    for year, quarter in periods:
        computed, skipped = compute_period_softmaps(
            year, quarter, model_service, args.force
        )
        total_computed += computed
        total_skipped += skipped
    
    # Özet
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ TAMAMLANDI")
    print("=" * 70)
    print(f"⏱️  Süre: {elapsed:.1f} saniye")
    print(f"✅ Hesaplanan: {total_computed}")
    print(f"⏭️  Atlanan: {total_skipped}")
    print(f"📊 Toplam: {total_computed + total_skipped}")
    
    if total_computed > 0:
        print(f"⚡ Ortalama: {elapsed / total_computed:.2f} sn/tile")
    
    print()
    print(f"💾 Softmap'ler: {SOFTMAPS_DIR}")
    print()
    
    return 0


if __name__ == "__main__":
    main()