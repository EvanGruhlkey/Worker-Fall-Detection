"""Convert actual fall_pipeline signals.json events into a study alert schedule."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def convert(path, camera, offset=0):
    raw = path.read_bytes()
    signals = json.loads(raw)
    if not signals.get("classifier"):
        raise ValueError("Pipeline output has no classifier; do not fabricate detector alerts from ground truth")
    alerts = []
    for event in signals["events"]:
        time = float(event["time"]) + offset
        if not math.isfinite(time) or time < 0:
            raise ValueError("Alert times must be finite and nonnegative after offset")
        alerts.append({"camera": camera, "time": time})
    return {"alerts": sorted(alerts, key=lambda e: e["time"]), "classifier": signals["classifier"],
            "source": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "note": "Video-time recognition timestamps; live inference and delivery delay are not measured."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(convert(args.input, args.camera, args.offset), indent=2), encoding="utf-8")
