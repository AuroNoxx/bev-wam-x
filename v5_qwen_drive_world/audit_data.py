"""Read current raw annotation inventory; do not generate labels or duplicate data."""
from .runtime import ROOT
import gzip
import json
import math
from collections import Counter


def main():
    manifest = json.loads((ROOT / "data/occ_v3/manifest.json").read_text())
    results = []; total = Counter()
    for route in manifest["routes"]:
        bins = Counter(); max_speed = 0.; count = 0
        folder = ROOT / "data/public/bench2drive/clips" / route["route"] / "anno"
        for path in folder.glob("*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream: row = json.load(stream)
            speed = float(row["speed"]) * 3.6
            if not math.isfinite(speed): raise ValueError(f"Nonfinite source speed: {path}")
            bucket = min(6, max(0, int(speed // 20)))
            bins[bucket] += 1; max_speed = max(max_speed, speed); count += 1
        total.update(bins)
        results.append({"route": route["route"], "split": route["split"], "frames": count,
                        "max_speed_kmh": max_speed, "speed_bins": dict(bins)})
    output = {"scope": "all extracted annotation frames in existing 60-route split, not new v5 samples",
              "bins_kmh": ["0-20", "20-40", "40-60", "60-80", "80-100", "100-120", ">=120"],
              "histogram": {str(i): total[i] for i in range(7)}, "routes": results,
              "v5_highway_dataset_exists": (ROOT / "data/v5-qwen-drive-world/highway").exists(),
              "raw_rgb": "retained in original archives; historical clips extraction contains anno/OCC only",
              "far_occ_support": "old native OCC is +/-25.6m, cannot directly supervise 112m corridor",
              "timestamps": "annotation index*0.1s cadence assumption must be verified against collector"}
    (ROOT / "artifacts/v5-qwen-drive-world/data-audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "routes"}, indent=2))


if __name__ == "__main__": main()
