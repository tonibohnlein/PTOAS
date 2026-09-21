"""Checkpointed, hash-verified native corpus construction; never uses a device."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--baseline-driver", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    assert args.jobs > 0 and args.timeout > 0
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.package / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if c["target"] == "a3"]
    drivers = {"candidate": args.driver.resolve()}
    if args.baseline_driver:
        drivers["baseline"] = args.baseline_driver.resolve()
    identities = {arm: dict(path=str(p), sha256=sha(p)) for arm, p in drivers.items()}
    (args.out / "drivers.json").write_text(json.dumps(identities, indent=2) + "\n")
    for case in cases:
        assert sha(args.package / case["input"]) == case["sha256"], case["id"]

    def run(case):
        row = dict(case=case["id"], input_sha256=case["sha256"], arms={})
        for arm, driver in drivers.items():
            assert sha(driver) == identities[arm]["sha256"], "driver changed during comparison"
            out = args.out / case["id"] / arm
            out.mkdir(parents=True)
            command = [str(driver), "--construct", str((args.package / case["input"]).resolve())]
            start = time.monotonic()
            with (out / "plan.pto").open("w") as stdout, (out / "construction.log").open("w") as stderr:
                try:
                    rc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = 124
            functions = [dict(x.split("=", 1) for x in line.split() if "=" in x)
                         for line in (out / "construction.log").read_text().splitlines()
                         if line.startswith("function=")]
            row["arms"][arm] = dict(
                command=command, rc=rc, seconds=time.monotonic() - start,
                plan_sha256=sha(out / "plan.pto"), functions=functions,
                passed=rc == 0 and bool(functions) and all(
                    f.get("construction") == f.get("reconstruction") == "1" for f in functions))
        row["passed"] = all(a["passed"] for a in row["arms"].values())
        if "baseline" in drivers and row["passed"]:
            row["unchanged"] = row["arms"]["candidate"]["plan_sha256"] == row["arms"]["baseline"]["plan_sha256"]
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, case) for case in cases]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: r["case"])
            (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(row["case"], "PASS" if row["passed"] else "FAIL", row.get("unchanged", "unpaired"), flush=True)
    assert all(sha(driver) == identities[arm]["sha256"] for arm, driver in drivers.items())
    raise SystemExit(0 if len(rows) == len(cases) and all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()
