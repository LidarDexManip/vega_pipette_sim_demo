#!/usr/bin/env python3
"""check_release.py [ROOT]: scan the repository for things a portable, shareable release must not contain.

  - machine-specific absolute paths (/home/<user>, /data*, /mnt, /scratch, conda/venv install paths)
  - non-ASCII CJK text (the release is English-only)
  - secrets: API tokens, private keys, password assignments, credential files
  - IPv4 addresses other than loopback / 0.0.0.0
  - sys.path edits in the package (vega_pipette/ is imported as an installed package)

Exit code 0 when clean; otherwise prints one line per finding and exits 1. Binary files are skipped.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
SKIP_DIRS = {".git", "runs", "__pycache__", "_texcache", "PackingTable"}
CJK_RANGES = ((0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF))
CHECKS = {
    "abs-path": re.compile(r"(?<![\w.])/(?:home|data\d*|mnt|scratch|Users)/[\w.-]+|miniconda\d?/envs|/opt/conda"),
    "cjk": re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in CJK_RANGES) + "]"),
    "secret": re.compile(r"(?:hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_\w{20,}|sk-[A-Za-z0-9_-]{20,}|"
                         r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?i:password|passwd|api_key)\s*[:=]\s*['\"][^'\"]+)"),
    # an address stands alone (after a space, quote, '@', '/' or ':'), unlike a version number after '==' or '-'
    "ipv4": re.compile(r"(?:^|(?<=[\s'\"@/:(]))(?!127\.0\.0\.1|0\.0\.0\.0)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                       r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"),
}
PACKAGE_ONLY = {"sys-path": re.compile(r"\bsys\.path\.(?:insert|append)\b")}
BAD_FILES = {".netrc", ".msmtprc", "token", ".env", "credentials", "id_rsa", "id_ed25519"}


def text_of(p: Path) -> str | None:
    data = p.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def files() -> list[Path]:
    """The files a release would ship: git's tracked + untracked-not-ignored list when ROOT is a git work tree."""
    if (ROOT / ".git").exists():
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, check=True).stdout.decode()
        return sorted(ROOT / f for f in out.split("\0") if f)
    return sorted(p for p in ROOT.rglob("*") if p.is_file())


def main() -> int:
    findings = []
    me = Path(__file__).resolve()
    for p in files():
        if not p.is_file() or p.resolve() == me or any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        rel = p.relative_to(ROOT)
        if p.name in BAD_FILES:
            findings.append(f"{rel}: credential-like file name")
        if p.suffix.lower() in (".obj", ".usd", ".usda", ".urdf", ".stl", ".dae", ".mtl", ".glb", ".png", ".jpg", ".npz"):
            if p.suffix.lower() not in (".usda", ".urdf", ".mtl"):
                continue                      # binary / numeric geometry: only text scene files are scanned
        text = text_of(p)
        if text is None:
            continue
        checks = {**CHECKS, **(PACKAGE_ONLY if rel.parts[0] == "vega_pipette" else {})}
        for i, line in enumerate(text.splitlines(), 1):
            for name, rx in checks.items():
                if name == "ipv4" and (p.suffix.lower() in (".usda", ".urdf", ".obj", ".lock") or p.name.startswith("requirements")):
                    continue                  # numeric geometry and dependency locks hold dotted quads that are not addresses
                m = rx.search(line)
                if m:
                    findings.append(f"{rel}:{i}: {name}: {line.strip()[:160]}")
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) in {ROOT.name}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
