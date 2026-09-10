# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Automatic serial OAHS gate; campaign artifacts stay in the build directory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    if not __debug__:
        raise RuntimeError("OAHS acceptance requires assertions")
    parser = argparse.ArgumentParser()
    for name in ("relation-driver", "occurrence-driver", "native-driver", "candidates-driver", "slot-mapping-driver", "opt",
                 "python-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="focused-", dir=args.output_root.resolve()))
    print("OAHS focused artifacts:", output, flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(args.python_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    records = []
    deadline = time.monotonic() + 540  # Leave CTest time for failure cleanup.

    def run(name, command, expected=0):
        start = time.monotonic()
        child = subprocess.Popen([str(c) for c in command], cwd=HERE, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                 start_new_session=(os.name == "posix"))
        timed_out = False
        try:
            stdout, stderr = child.communicate(timeout=max(0.01, min(240, deadline - start)))
        except subprocess.TimeoutExpired:
            timed_out = True
            # A runner may currently own a native compiler child. Kill our
            # process group, not just the Python parent, before another test.
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            stdout, stderr = child.communicate()
        result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
        (output / (name + ".stdout")).write_text(stdout)
        (output / (name + ".stderr")).write_text(stderr)
        record = {"name": name, "command": [str(c) for c in command],
                  "seconds": time.monotonic() - start, "exit": result.returncode, "timed_out": timed_out}
        records.append(record)
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        if timed_out or (result.returncode == 0) != (expected == 0):
            raise RuntimeError(f"{name} failed: {result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        print(name, "passed", flush=True)
        return result

    # The reference is mandatory when this gate is selected; no silent skip if
    # libisl is unavailable. Python/libisl remain test-only dependencies.
    for script, driver in (("check_relations", args.relation_driver),
                           ("check_occurrences", args.occurrence_driver),
                           ("check_scalars", args.occurrence_driver),
                           ("check_slot_mapping", args.slot_mapping_driver),
                           ("check_requirements", args.native_driver),
                           ("check_qualification", args.native_driver)):
        run(script, [sys.executable, HERE / (script + ".py"), "--driver", driver,
                     "--output", output / script])
    run("candidates", [args.candidates_driver])
    run("physical-addresses", [sys.executable, HERE / "check_physical_addresses.py", "--opt", args.opt,
                                 "--driver", args.native_driver, "--python-root", args.python_root,
                                 "--output", output / "physical-addresses"])
    run("slots", [sys.executable, HERE / "check_slots.py", "--focused", "--driver", args.native_driver,
                  "--python-root", args.python_root, "--output", output / "slots"])
    run("compact_slots", [sys.executable, HERE / "check_slots.py", "--compact-scaling", "--driver", args.native_driver,
                          "--python-root", args.python_root, "--output", output / "compact_slots"])
    run("slot_components", [sys.executable, HERE / "check_slot_components.py", "--driver", args.native_driver,
                            "--python-root", args.python_root, "--output", output / "slot_components"])
    run("scalability", [sys.executable, HERE / "check_scalability.py", "--opt", args.opt,
                        "--python-root", args.python_root, "--output", output / "scalability", "--repetitions", "1"])
    run("reconstruction-cuts", [sys.executable, HERE / "check_reconstruction_cuts.py", "--driver", args.native_driver,
                                 "--python-root", args.python_root, "--output", output / "reconstruction-cuts"])
    run("observers", [sys.executable, HERE / "test_observations.py"])
    run("constructor", [sys.executable, HERE / "check_constructor.py", "--focused",
                         "--python-root", args.python_root, "--native-driver", args.native_driver,
                         "--output", output / "constructor"])
    run("allocation", [sys.executable, HERE / "check_allocation.py",
                        "--python-root", args.python_root, "--output", output / "allocation"])
    run("direct-guards", [sys.executable, HERE / "check_direct_guards.py",
                           "--python-root", args.python_root, "--output", output / "direct-guards"])
    run("retirement", [sys.executable, HERE / "check_retirement.py", "--driver", args.native_driver,
                        "--python-root", args.python_root, "--output", output / "retirement"])
    run("compact-updates", [sys.executable, HERE / "check_compact_updates.py",
                             "--driver", args.relation_driver, "--output", output / "compact-updates"])
    run("completion-thresholds", [sys.executable, HERE / "check_completion_thresholds.py",
                                   "--driver", args.relation_driver,
                                   "--output", output / "completion-thresholds"])
    run("endpoints", [sys.executable, HERE / "check_endpoints.py", "--driver", args.native_driver,
                       "--python-root", args.python_root, "--output", output / "endpoints"])
    source = ROOT / "test/lit/pto/insert_sync_authored_dynamic.pto"
    base = [args.opt, "--mlir-disable-threading"]
    parsed = run("authored.input", [*base, source]).stdout
    existing = run("authored.existing", [*base, "--pto-insert-sync=planner=existing", source]).stdout
    assert parsed == existing, "Existing mode changed an authored dynamic protocol"
    hybrid = run("authored.hybrid", [*base, "--pto-insert-sync=planner=logical-or-existing", source]).stdout
    assert 'pto.insert_sync.producer = "authored"' in hybrid
    assert parsed[parsed.index("    %"): ] == hybrid[hybrid.index("    %"): ], "Hybrid changed authored body"
    strict = run("authored.strict", [*base, "--pto-insert-sync=planner=logical", source], expected=1)
    assert "authored event protocol has no qualified summary" in strict.stderr
    print("OAHS focused gate passed; full milestone and device acceptance are separate", flush=True)


if __name__ == "__main__":
    main()
