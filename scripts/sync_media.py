"""Upload snapshots, clips and annotated videos that shared storage doesn't have yet.

    uv run scripts/sync_media.py

The backend does this by itself on every start; run it by hand after
scripts/copy_data.py, or to check the storage settings in .env work.
Uploaded source videos (data/input/) stay on this computer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.config import Settings  # noqa: E402
from backend.storage import MediaStore  # noqa: E402


def main():
    settings = Settings()
    if not settings.storage_bucket:
        sys.exit("FALL_STORAGE_BUCKET is not set in .env, so media stays on this computer.")
    media = MediaStore(settings)
    remote = media.remote_keys()  # fails here, with the reason, if the settings are wrong
    print(f"Bucket '{settings.storage_bucket}' has {len(remote)} files.")
    uploaded = media.sync()
    media.shutdown()
    print(f"Uploaded {uploaded} files from {settings.data_dir}.")


if __name__ == "__main__":
    main()
