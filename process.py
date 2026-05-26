#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, Tuple


LINE_RE = re.compile(r"cifar10-nfe(?P<nfe>\d+)-ema(?P<ema>[02])\s*:\s*(?P<fid>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_fid_log(log_file: str) -> Dict[int, Dict[int, float]]:
    """Parse fid log lines and keep the best (minimum) FID for each nfe/ema pair."""
    results: Dict[int, Dict[int, float]] = {}
    with open(log_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = LINE_RE.search(line)
            if m is None:
                continue
            nfe = int(m.group("nfe"))
            ema = int(m.group("ema"))
            fid = float(m.group("fid"))
            if nfe not in results:
                results[nfe] = {}
            if ema not in results[nfe]:
                results[nfe][ema] = fid
            else:
                results[nfe][ema] = min(results[nfe][ema], fid)
    return results


def summarize(nfe2ema2fid: Dict[int, Dict[int, float]], nfe_start: int, nfe_end: int):
    rows = []
    for nfe in range(nfe_start, nfe_end + 1):
        ema_map = nfe2ema2fid.get(nfe, {})
        fid0 = ema_map.get(0)
        fid2 = ema_map.get(2)

        candidates: Dict[int, float] = {}
        if fid0 is not None:
            candidates[0] = fid0
        if fid2 is not None:
            candidates[2] = fid2

        if candidates:
            best_ema, best_fid = min(candidates.items(), key=lambda kv: kv[1])
        else:
            best_ema, best_fid = None, None

        rows.append(
            {
                "nfe": nfe,
                "fid_ema0": "" if fid0 is None else f"{fid0:.6f}",
                "fid_ema2": "" if fid2 is None else f"{fid2:.6f}",
                "best_fid": "" if best_fid is None else f"{best_fid:.6f}",
                "best_ema_kimg": "" if best_ema is None else str(best_ema),
            }
        )
    return rows


def write_csv(rows, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nfe", "fid_ema0", "fid_ema2", "best_fid", "best_ema_kimg"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Process CIFAR10 FID logs and export best EMA summary.")
    parser.add_argument("--log-file", required=True, help="Path to FID log file.")
    parser.add_argument("--out-csv", default="csv/test.csv", help="Path to output CSV.")
    parser.add_argument("--nfe-start", type=int, default=3, help="Start NFE value.")
    parser.add_argument("--nfe-end", type=int, default=8, help="End NFE value.")
    args = parser.parse_args()

    if not os.path.isfile(args.log_file):
        raise FileNotFoundError(f"FID log file not found: {args.log_file}")

    parsed = parse_fid_log(args.log_file)
    rows = summarize(parsed, args.nfe_start, args.nfe_end)
    write_csv(rows, args.out_csv)
    print(f"Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
