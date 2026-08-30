#!/usr/bin/env python3
"""Download and verify the fixed BSRN PANGAEA snapshot (DOI 957398)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path


DATASET_ID = "957398"
DOI_URL = f"https://doi.pangaea.de/10.1594/PANGAEA.{DATASET_ID}?format=html"
USER_AGENT = "state-space-zero-shot-forecasting/0.1 (academic dataset retrieval)"
PRINT_LOCK = threading.Lock()


class InventoryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[list[tuple[str, str]]]] = []
        self.row: list[list[tuple[str, str]]] | None = None
        self.cell: list[tuple[str, str]] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = []
        elif tag == "a" and self.cell is not None and values.get("href"):
            self.cell.append(("href", values["href"]))

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(("text", data))

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.cell is not None and self.row is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_size(value: str) -> int:
    match = re.match(r"([0-9.]+)\s*([KMG])Bytes", value)
    if not match:
        raise ValueError(f"Unrecognized PANGAEA size: {value!r}")
    return round(float(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)])


def optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def fetch_inventory() -> list[dict]:
    request = urllib.request.Request(DOI_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read().decode("utf-8", "replace")
    parser = InventoryParser()
    parser.feed(raw)
    inventory = []
    for row in parser.rows:
        links = [value for cell in row for kind, value in cell if kind == "href" and f"/dataset/{DATASET_ID}/files/" in value]
        if not links:
            continue
        texts = [" ".join(value.strip() for kind, value in cell if kind == "text").strip() for cell in row]
        inventory.append(
            {
                "station_name": texts[0],
                "station_code": Path(links[0]).stem,
                "latitude": optional_float(texts[2]),
                "longitude": optional_float(texts[3]),
                "elevation_m": optional_float(texts[4]),
                "url": links[0],
                "published_size": texts[-1],
                "published_size_bytes_approx": parse_size(texts[-1]),
            }
        )
    if len(inventory) != 76:
        raise RuntimeError(f"Expected 76 station files, found {len(inventory)}")
    return inventory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_zip(path: Path, *, deep: bool = True) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size == 0:
        return False, 0
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                return False, 0
            if deep and archive.testzip() is not None:
                return False, 0
            return True, len(names)
    except zipfile.BadZipFile:
        return False, 0


def download_one(item: dict, archive_dir: Path, attempts: int, position: int, total: int) -> dict:
    destination = archive_dir / f"{item['station_code']}.zip"
    # A completed destination was already deeply tested before its atomic
    # .part -> .zip rename. On resume, only re-read the ZIP central directory;
    # the final audit performs a fresh deep test and checksum pass over all files.
    is_valid, members = valid_zip(destination, deep=False)
    if is_valid:
        with PRINT_LOCK:
            print(f"[{position:02d}/{total}] keep  {destination.name} ({destination.stat().st_size / 1024**2:.1f} MiB, {members} files)", flush=True)
        return {**item, "status": "verified", "bytes": destination.stat().st_size, "members": members, "sha256": None}

    part = destination.with_suffix(".zip.part")
    for attempt in range(1, attempts + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT, "Referer": DOI_URL}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            request = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                append = existing > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            part.replace(destination)
            is_valid, members = valid_zip(destination, deep=True)
            if not is_valid:
                destination.replace(part)
                raise RuntimeError("downloaded file failed ZIP validation")
            result = {
                **item,
                "status": "downloaded",
                "bytes": destination.stat().st_size,
                "members": members,
                "sha256": sha256(destination),
            }
            with PRINT_LOCK:
                print(f"[{position:02d}/{total}] saved {destination.name} ({result['bytes'] / 1024**2:.1f} MiB, {members} files)", flush=True)
            return result
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504, 529}:
                raise
            retry_after = int(error.headers.get("Retry-After", "15"))
            # PANGAEA's 503 page means the file is being restored from tape and
            # explicitly says to wait about a minute. 429 needs a longer quiet
            # period; aggressive polling only delays the archive for everyone.
            if error.code == 503:
                wait = max(60, retry_after)
            elif error.code == 429:
                wait = max(180, retry_after)
            else:
                wait = max(retry_after, min(90, 5 * attempt))
            with PRINT_LOCK:
                print(f"[{position:02d}/{total}] stage {destination.name}: HTTP {error.code}; retry in {wait}s", flush=True)
        except (TimeoutError, urllib.error.URLError, ConnectionError, RuntimeError, OSError) as error:
            wait = min(90, 5 * attempt)
            with PRINT_LOCK:
                print(f"[{position:02d}/{total}] retry {destination.name}: {type(error).__name__}; retry in {wait}s", flush=True)
        if attempt == attempts:
            raise RuntimeError(f"Failed after {attempts} attempts: {item['url']}")
        time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--reverse", action="store_true", help="Process station inventory in reverse order")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = args.output / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inventory = fetch_inventory()
    if args.reverse:
        inventory.reverse()
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Inventory: {len(inventory)} station ZIPs, approximately "
        f"{sum(x['published_size_bytes_approx'] for x in inventory) / 1024**3:.2f} GiB compressed",
        flush=True,
    )

    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(download_one, item, archive_dir, args.attempts, index, len(inventory)): item
            for index, item in enumerate(inventory, start=1)
        }
        for future in concurrent.futures.as_completed(future_map):
            item = future_map[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({**item, "status": "failed", "error": repr(error)})
                with PRINT_LOCK:
                    print(f"FAILED {item['station_code']}: {error!r}", flush=True)

    results.sort(key=lambda row: row["station_code"])
    report = {
        "dataset_doi": "10.1594/PANGAEA.957398",
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified_files": len(results),
        "verified_bytes": sum(row["bytes"] for row in results),
        "files": results,
        "failures": failures,
    }
    (args.output / "download_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified {len(results)}/{len(inventory)} archives ({report['verified_bytes'] / 1024**3:.2f} GiB)", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed; rerun to resume")


if __name__ == "__main__":
    main()
