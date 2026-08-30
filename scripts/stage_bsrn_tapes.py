#!/usr/bin/env python3
"""Submit one low-rate tape-restore request for each missing BSRN archive."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


USER_AGENT = "state-space-zero-shot-forecasting/0.1 (academic dataset retrieval)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--delay", type=float, default=8.0)
    args = parser.parse_args()
    inventory = json.loads((args.dataset_root / "inventory.json").read_text(encoding="utf-8"))
    archive_dir = args.dataset_root / "archives"
    missing = [item for item in inventory if not (archive_dir / f"{item['station_code']}.zip").exists()]
    print(f"Submitting staging requests for {len(missing)} missing archives", flush=True)
    results = []
    for index, item in enumerate(missing, start=1):
        code = item["station_code"]
        request = urllib.request.Request(
            item["url"],
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                response.read(1)
                status = f"online_http_{response.status}"
        except urllib.error.HTTPError as error:
            status = "restore_requested" if error.code == 503 else f"http_{error.code}"
        except Exception as error:
            status = f"{type(error).__name__}: {error}"
        results.append({"station_code": code, "status": status})
        print(f"[{index:02d}/{len(missing)}] {code}: {status}", flush=True)
        if index < len(missing):
            time.sleep(args.delay)
    (args.dataset_root / "stage_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
