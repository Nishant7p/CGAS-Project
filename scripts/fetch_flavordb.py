"""Download every FlavorDB entity (ingredient) as raw JSON, politely and resumably.

The CoSyLab site serves one ingredient per request at
    https://cosylab.iiitd.edu.in/<db>/entities_json?id=<entity_id>
where <db> is "flavordb" or "flavordb2". Checked on 2026-11-22: both endpoints
serve IDENTICAL current data (936 entities, IDs 0-978, 60,263 entity-molecule
links, 1,788 molecules), so the 2018 FlavorDB used by Singh & Bagler is no
longer online. CulinaryDB Entity IDs are the same IDs, so these files join
directly to CulinaryDB (24 CulinaryDB ingredients, IDs 979-1002, have no entry).

Each response is stored gzip-compressed as <out_dir>/<id>.json.gz; IDs that
return "Not Found" are recorded in <out_dir>/_status.csv so reruns skip them.

Usage:
    python -I scripts/fetch_flavordb.py --db flavordb  --out data/raw/flavordb_v1
    python -I scripts/fetch_flavordb.py --db flavordb2 --out data/raw/flavordb_v2
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://cosylab.iiitd.edu.in/{db}/entities_json?id={id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic course project; FlavorDB research client)"}


def fetch(url: str, retries: int = 4, timeout: int = 60) -> tuple[int, bytes]:
    delay = 5.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, b""
            if attempt == retries - 1:
                return e.code, b""
        except Exception:  # timeouts, connection resets
            if attempt == retries - 1:
                return -1, b""
        time.sleep(delay)
        delay *= 2
    return -1, b""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=["flavordb", "flavordb2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=1010, help="last id to try (inclusive)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    status_path = os.path.join(args.out, "_status.csv")
    done: dict[int, str] = {}
    if os.path.exists(status_path):
        with open(status_path, newline="") as f:
            for row in csv.DictReader(f):
                done[int(row["entity_id"])] = row["status"]

    new_file = not os.path.exists(status_path)
    with open(status_path, "a", newline="") as sf:
        w = csv.writer(sf)
        if new_file:
            w.writerow(["entity_id", "status", "n_molecules", "name"])
        ok = missing = failed = 0
        for eid in range(args.start, args.stop + 1):
            if done.get(eid) in ("ok", "not_found"):
                continue
            code, body = fetch(BASE.format(db=args.db, id=eid))
            if code == 200 and body.lstrip().startswith(b"{"):
                try:
                    d = json.loads(body)
                except json.JSONDecodeError:
                    w.writerow([eid, "bad_json", "", ""])
                    failed += 1
                else:
                    with gzip.open(os.path.join(args.out, f"{eid}.json.gz"), "wb") as g:
                        g.write(body)
                    n_mol = len(d.get("molecules") or [])
                    w.writerow([eid, "ok", n_mol, d.get("entity_alias_readable", "")])
                    ok += 1
            elif code in (200, 404):
                w.writerow([eid, "not_found", "", ""])
                missing += 1
            else:
                w.writerow([eid, f"error_{code}", "", ""])
                failed += 1
            sf.flush()
            time.sleep(args.sleep)
        print(f"[{args.db}] ok={ok} not_found={missing} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
