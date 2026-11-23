"""Verify that an ungated Hugging Face mirror is byte-identical to the official (gated) model repo.

Compares the LFS SHA-256 (weights) and git blob ids (configs, tokenizer, chat template) that the public HF API
exposes even for gated repos, and writes the evidence to data/reference/<name>_mirror_check.json.

Usage (repo root):
    .venv/bin/python -I scripts/verify_mirror.py meta-llama/Llama-3.1-8B-Instruct RedHatAI/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP = (".gitattributes", "LICENSE", "USE_POLICY.md", "README.md")


def files(repo_id: str) -> tuple[dict, str]:
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}?blobs=true", headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    out = {}
    for s in d["siblings"]:
        f = s["rfilename"]
        if f.startswith("original/") or f in SKIP:
            continue
        out[f] = (s.get("lfs") or {}).get("sha256") or s.get("blobId")
    return out, d.get("sha", "")


def main() -> int:
    official, mirror = sys.argv[1], sys.argv[2]
    off, off_sha = files(official)
    mir, mir_sha = files(mirror)
    status = {
        f: ("identical" if mir.get(f) == h else ("missing" if f not in mir else "different")) for f, h in off.items()
    }
    ok = all(v == "identical" for v in status.values())
    rec = {
        "official": official,
        "official_commit": off_sha,
        "mirror": mirror,
        "mirror_commit": mir_sha,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "byte_identical": ok,
        "files": status,
        "extra_files_in_mirror": sorted(set(mir) - set(off)),
    }
    out = REPO / "data" / "reference" / f"{official.split('/')[-1]}_mirror_check.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    n_same = sum(v == "identical" for v in status.values())
    print(f"{mirror} vs {official}: byte_identical={ok} ({n_same}/{len(status)} files) -> {out.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
