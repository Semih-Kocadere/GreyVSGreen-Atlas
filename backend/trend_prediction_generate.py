"""
trend_prediction_generate.py
============================
Localde softmax/istatistikten trend_prediction.json dosyası üretir.
Kullanım: python3 trend_prediction_generate.py
"""
import sys
from pathlib import Path
import numpy as np
import json
sys.path.append(str(Path(__file__).parent))
from model_service import model_service, get_available_periods, predict_period
import argparse
from PIL import Image


def main():
    
    parser = argparse.ArgumentParser(description="Trend prediction generator for t+1, t+4, t+8.")
    parser.add_argument('--horizons', nargs='+', type=int, default=[1], choices=[1,4,8], help='Tahmin horizonları: 1, 4, 8')
    args = parser.parse_args()

    # Modelleri yükle (Conv3D dahil)
    model_service.load_models()
    out_dir = Path("data/prediction_outputs_trend_tiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Softmax kaynak klasörü
    softmaps_dir = Path("softmaps")
    assert softmaps_dir.exists(), f"Softmaps klasörü yok: {softmaps_dir}"

    # Tüm softmax dosyalarını tara, tile bazında grupla
    from collections import defaultdict
    import re
    tile_softmax_seq = defaultdict(dict)  # {(row, col): {period: path}}
    softmax_re = re.compile(r"(?P<year>\d{4})_Q(?P<quarter>[1-4])_(?P<row>\d{5})_(?P<col>\d{5})\.npy")
    for fp in softmaps_dir.glob("*.npy"):
        m = softmax_re.match(fp.name)
        if not m:
            continue
        year = int(m.group("year"))
        quarter = int(m.group("quarter"))
        row = int(m.group("row"))
        col = int(m.group("col"))
        key = (row, col)
        tile_softmax_seq[key][(year, quarter)] = fp

    palette = {0: (0,0,0,0), 1: (0,200,0,160), 2: (130,130,130,160), 3: (0,180,255,160)}
    saved = {1:0, 4:0, 8:0}

    # Dönemleri hazırla (tüm çeyrekler)
    periods = get_available_periods()
    periods_all = sorted(periods)

    horizon_windows = {1:8, 4:12, 8:16}
    horizon_suffix = {1:'tplus1', 4:'tplus4', 8:'tplus8'}
    horizon_predict_fn = {
        1: getattr(model_service, 'predict_temporal', None),
        4: getattr(model_service, 'predict_temporal_tplus4', None),
        8: getattr(model_service, 'predict_temporal_tplus8', None)
    }

    for horizon in args.horizons:
        window = horizon_windows[horizon]
        periods_window = periods_all[-window:]
        predict_fn = horizon_predict_fn[horizon]
        if predict_fn is None:
            print(f"Model servisinde horizon {horizon} için fonksiyon yok!")
            continue
        for key, period2fp in tile_softmax_seq.items():
            seq_paths = [period2fp.get(p) for p in periods_window]
            if any(x is None for x in seq_paths):
                continue
            seq = [np.load(fp).astype(np.float32) for fp in seq_paths]
            trend_pred = predict_fn(seq)
            pred_mask = trend_pred.argmax(axis=0)
            h, w = pred_mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            for k, col in palette.items():
                rgba[pred_mask == k] = col
            row, col = key
            fname = f"2026_Q1_{row:05d}_{col:05d}_trend_{horizon_suffix[horizon]}"
            Image.fromarray(rgba, "RGBA").save(out_dir / f"{fname}.png")
            np.save(out_dir / f"{fname}.npy", pred_mask)
            saved[horizon] += 1

    for horizon in args.horizons:
        print(f"✓ {saved[horizon]} tile için {horizon_suffix[horizon]} tahmini ve maskesi kaydedildi: {out_dir}")
if __name__ == "__main__":
    main()
