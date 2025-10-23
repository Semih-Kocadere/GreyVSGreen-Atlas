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

def main():
    # Modelleri yükle (Conv3D dahil)
    model_service.load_models()
    # Her tile için t+1 tahmini üret (Colab mantığı, doğrudan softmaps'ten)
    from PIL import Image
    out_dir = Path("data/prediction_outputs_trend_tiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Softmax kaynak klasörü
    softmaps_dir = Path("softmaps")
    assert softmaps_dir.exists(), f"Softmaps klasörü yok: {softmaps_dir}"

    # Son 8 yılın Q1'ini bul
    periods = get_available_periods()
    periods_q1 = [p for p in periods if p[1] == 1]
    periods_q1 = sorted(periods_q1)[-8:]  # Son 8 yılın Q1'i

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

    # Sadece 8'lik sequence olan tile'lar için t+1 tahmini yap
    palette = {0: (0,0,0,0), 1: (0,200,0,160), 2: (130,130,130,160), 3: (0,180,255,160)}
    saved = 0
    for key, period2fp in tile_softmax_seq.items():
        # Her tile için 8 Q1 softmax dosyası var mı?
        seq_paths = [period2fp.get(p) for p in periods_q1]
        if any(x is None for x in seq_paths):
            continue
        seq = [np.load(fp).astype(np.float32) for fp in seq_paths]
        trend_pred = model_service.predict_temporal(seq)
        pred_mask = trend_pred.argmax(axis=0)
        # PNG kaydet
        h, w = pred_mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for k, col in palette.items():
            rgba[pred_mask == k] = col
        row, col = key
        fname = f"2026_Q1_{row:05d}_{col:05d}_trend_tplus1"
        Image.fromarray(rgba, "RGBA").save(out_dir / f"{fname}.png")
        np.save(out_dir / f"{fname}.npy", pred_mask)
        saved += 1

    print(f"✓ {saved} tile için t+1 tahmini ve maskesi kaydedildi: {out_dir}")

if __name__ == "__main__":
    main()
