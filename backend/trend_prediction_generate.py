"""
trend_prediction_generate.py
============================
Localde softmax/istatistikten trend_prediction.json dosyası üretir.
Kullanım: python3 trend_prediction_generate.py
"""
import numpy as np
import json
from pathlib import Path

def main():

    # 2018-2026 arası her yıl için Q1 tahmini üret
    predictions = []
    for year in range(2018, 2027):
        # Her yıl için rastgele softmax ve istatistik üret (örnek)
        softmax = np.random.rand(4, 256, 256)
        softmax = softmax / softmax.sum(axis=0, keepdims=True)
        pred = softmax.argmax(axis=0)
        total = pred.size
        stats = {
            "background": float((pred == 0).sum()) / total * 100,
            "green": float((pred == 1).sum()) / total * 100,
            "gray": float((pred == 2).sum()) / total * 100,
            "water": float((pred == 3).sum()) / total * 100
        }

        prediction = {
            "prediction": {
                "timeframe": f"{year} Q1",
                "green": round(stats["green"], 2),
                "grey": round(stats["gray"], 2),
                "water": round(stats["water"], 2),
                "background": round(stats["background"], 2)
            },
            "current": {
                "timeframe": f"{year-1} Q4" if year > 2018 else "-",
                "green": round(stats["green"] + 0.7, 2),
                "grey": round(stats["gray"] - 0.9, 2),
                "water": round(stats["water"] + 0.2, 2),
                "background": round(stats["background"], 2)
            },
            "input_sequence": [],
            "changes": {
                "green": -0.7,
                "grey": +0.9,
                "water": -0.2
            },
            "confidence": {
                "overall": 0.94,
                "green": 0.93,
                "grey": 0.95,
                "water": 0.92
            },
            "metadata": {
                "model": "Conv3D-Temporal",
                "sequence_length": 8,
                "timestamp": "2025-10-22T12:00:00"
            },
            "current_softmax": softmax.tolist()
        }
        predictions.append(prediction)

    out_path = Path(__file__).parent / "data" / "trend_prediction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"trend_prediction.json dosyası yazıldı: {out_path} (toplam {len(predictions)} yıl)")

if __name__ == "__main__":
    main()
