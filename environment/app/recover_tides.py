#!/usr/bin/env python3
"""Partial tide-gauge calibration entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    stations = payload.get("stations", [])
    result = {
        "status": "ok",
        "station_map": {
            record.get("id", "UNKNOWN"): stations[index % len(stations)].get("id", "UNKNOWN")
            for index, record in enumerate(records)
        },
        "calibration": {
            record.get("id", "UNKNOWN"): {"gain": 1.0, "lag_minutes": 0.0}
            for record in records
        },
    }
    target.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
