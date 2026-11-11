"""Download the parts of CAUCAFall we use (videos + per-frame labels, ~0.15 GB).

    uv run scripts/download_caucafall.py

Lists the dataset through Mendeley Data's public API, then downloads every .avi,
.txt and the details sheet (not the 20,000 PNG frames, 8 GB) into
data/external/caucafall/raw/, checking each file's SHA-256. Safe to rerun:
files already present and correct are skipped. Takes ~10-20 minutes, mostly
for the 20,000 small label files.

CAUCAFall: Eraso, Muñoz, Muñoz, Pinto (Universidad del Cauca), CC BY 4.0,
doi:10.17632/7w7fccy7ky.4. See DATA_SOURCES.md.
"""

import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "caucafall" / "raw"
API = "https://data.mendeley.com/public-api/datasets/7w7fccy7ky"
VERSION = 4
WANTED = (".avi", ".txt", ".xlsx")


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Worker-Fall-Detection dataset download"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def list_files():
    folders = {f["id"]: f for f in _get_json(f"{API}/folders/{VERSION}")}

    def folder_path(folder_id):
        parts = []
        while folder_id in folders:
            parts.append(folders[folder_id]["name"])
            folder_id = folders[folder_id].get("parent_id")
        return "/".join(reversed(parts))

    files = []
    for folder_id in ["root", *folders]:
        prefix = "" if folder_id == "root" else folder_path(folder_id) + "/"
        for item in _get_json(f"{API}/files?folder_id={folder_id}&version={VERSION}"):
            details = item["content_details"]
            files.append({"path": prefix + item["filename"], "sha256": details["sha256_hash"],
                          "url": details["download_url"]})
    return files


def fetch(item):
    target = OUT / item["path"]
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for _ in range(3):
        try:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "Worker-Fall-Detection dataset download"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("SHA-256 mismatch")
            target.write_bytes(data)
            return None
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            error = exc
    return f"{item['path']}: {error}"


def main():
    print("Listing CAUCAFall files...")
    files = [f for f in list_files() if f["path"].lower().endswith(WANTED)]
    print(f"Downloading {len(files)} files to {OUT} (already-correct files are skipped)...")
    failures, done = [], 0
    with ThreadPoolExecutor(16) as pool:
        for result in pool.map(fetch, files):
            done += 1
            if result:
                failures.append(result)
            if done % 2000 == 0:
                print(f"  {done}/{len(files)}", flush=True)
    if failures:
        print(f"{len(failures)} file(s) failed; rerun to retry:", file=sys.stderr)
        for failure in failures[:20]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"Done: {len(files)} files OK. Next: uv run scripts/import_caucafall.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
