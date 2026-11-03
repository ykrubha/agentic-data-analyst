"""Download the Chinook sample database (MIT licensed) at a pinned commit and verify it."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

COMMIT = "7f67772503d71ba90f19283c38e93923addb43fa"
URL = (
    f"https://raw.githubusercontent.com/lerocha/chinook-database/{COMMIT}"
    "/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
SHA256 = "7651ba378ac2fcd0dfc3c66fb101f7a7eed3ba39a612ec642b96e20702061f15"
TARGET = Path("data/chinook.sqlite")


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, TARGET)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if digest != SHA256:
        TARGET.unlink()
        raise SystemExit(f"Checksum mismatch: got {digest}")
    print(f"OK: {TARGET}")


if __name__ == "__main__":
    main()
