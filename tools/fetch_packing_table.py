#!/usr/bin/env python3
"""fetch_packing_table.py: download the packing table model used by the scene (not redistributed in this repository).

The table comes from the public Hugging Face dataset unitreerobotics/unitree_sim_isaaclab_usds (assets.zip, ~1.3 GB).
Only the PackingTable members are read, with HTTP range requests, so the whole archive is not downloaded. Files are
written to assets/props/PackingTable/ and checked against the md5 list in tools/packing_table.md5.

  python3 tools/fetch_packing_table.py [--out DIR] [--url URL] [--force]

Needs only the Python standard library and network access to huggingface.co.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds/resolve/main/assets.zip"
MD5_LIST = Path(__file__).resolve().parent / "packing_table.md5"
MEMBER_KEY = "objects/PackingTable/"


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (blocks cached), enough for zipfile."""

    BLOCK = 1 << 20

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url                     # follow the redirect once
        self.pos = 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            a = i * self.BLOCK
            b = min(a + self.BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.cache[i] = r.read()
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def readinto(self, buf: bytearray | memoryview) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out, p = bytearray(), self.pos
        while len(out) < n:
            blk = self._block(p // self.BLOCK)
            off = p % self.BLOCK
            chunk = blk[off:off + n - len(out)]
            out += chunk
            p += len(chunk)
        buf[:n] = out
        self.pos += n
        return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "assets/props/PackingTable")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--force", action="store_true", help="download even if every file is already present and correct")
    a = ap.parse_args()
    want = {}
    for line in MD5_LIST.read_text().splitlines():
        if line.strip():
            h, rel = line.split(None, 1)
            want[rel.strip()] = h
    have = {rel for rel, h in want.items()
            if (a.out / rel).is_file() and hashlib.md5((a.out / rel).read_bytes()).hexdigest() == h}
    if len(have) == len(want) and not a.force:
        print(f"all {len(want)} packing table files present in {a.out}")
        return 0
    print(f"reading {a.url} (range requests)", flush=True)
    zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(a.url), buffer_size=1 << 20))
    members = {n.split(MEMBER_KEY, 1)[1]: n for n in zf.namelist() if MEMBER_KEY in n and not n.endswith("/")}
    missing = sorted(set(want) - set(members))
    if missing:
        print(f"error: {len(missing)} expected files are not in the archive, e.g. {missing[:3]}", file=sys.stderr)
        return 1
    bad = []
    for rel in sorted(want):
        data = zf.read(members[rel])
        if hashlib.md5(data).hexdigest() != want[rel]:
            bad.append(rel)
            continue
        dst = a.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    if bad:
        print(f"error: md5 mismatch for {len(bad)} files, e.g. {bad[:3]}", file=sys.stderr)
        return 1
    print(f"wrote {len(want)} files to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
