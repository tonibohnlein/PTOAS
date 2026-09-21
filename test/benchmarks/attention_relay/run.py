#!/usr/bin/env python3
"""Check native slot import and split-relay construction against pinned words.

Candidate models and emission come from the production native wrapper.
Input and original-graph pins bind the captured baseline to these fixtures.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


PINS = {
    "2127cef1e8080fa8972ba4b952167d7c4e164265da592b66cdaaac144485faac": "row48",
    "7ca19d0b9d6d506c9bbe32c177390fe9afd00ec7ec3c50039354af4aa83cd7c2": "row49",
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"driver anchor changed: {old!r}")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    build, out = args.build.resolve(), args.out.resolve()
    inputs = []
    for path in args.inputs:
        path = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in PINS:
            raise RuntimeError(f"unpinned input: {path}: {digest}")
        inputs.append((path, digest, PINS[digest]))
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((here / "baseline.json").read_text())
    generated = ["mlir::pto::oahs::Commands baselineWords(const mlir::pto::oahs::Program& p) {",
                 "namespace o=mlir::pto::oahs;",
                 f"if(!p.observed || p.observed->sites.size()!={baseline['site_count']} || p.operations.size()!={len(baseline['pipes'])}) std::abort();",
                 f"o::Commands words({baseline['site_count']});"]
    for i, pipe in enumerate(baseline["pipes"]):
        generated.append(f"if(p.operations[{i}].pipe!=o::Pipe::{pipe}) std::abort();")
    for cut in baseline["cuts"]:
        op = "o::NoControlId" if cut["op"] < 0 else str(cut["op"])
        successors = ",".join(map(str, cut["next"]))
        observation = "o::NoControlId" if cut["observation"] < 0 else str(cut["observation"])
        generated.append(f"if(p.observed->sites[{cut['id']}].operation!={op} || "
                         f"p.observed->sites[{cut['id']}].observation!={observation} || "
                         f"p.observed->sites[{cut['id']}].successors!=std::vector<std::size_t>{{{successors}}}) std::abort();")
    for cut, commands in baseline["commands"].items():
        for c in commands:
            kind = ("Publish", "Acquire", "Barrier", "BarrierAll")[c["kind"]]
            generated.append(f"words[{cut}].push_back({{o::Command::{kind},o::Pipe::{c['source']},o::Pipe::{c['observer']},{c['key']}}});")
    generated += ["return words;", "}"]
    (out / "baseline.inc").write_text("\n".join(generated) + "\n")
    source = (repo / "tools/pto-test-opt/pto-oahs-selected-test.cpp").read_text()
    source = replace_once(
        source,
        '"../../lib/PTO/Transforms/OAHS/SelectedInternal.h"',
        '"' + str(repo / "lib/PTO/Transforms/OAHS/SelectedInternal.h") + '"',
    )
    source = replace_once(
        source, "#include <limits>",
        '#include <limits>\n#include "llvm/Support/JSON.h"\n'
        '#include "diagnostic.inc"\n#include "baseline.inc"\n#include "finite.inc"\n#include "static.inc"',
    )
    source = replace_once(
        source,
        "    oahs::SelectedPlan report;\n    const auto status =",
        "    oahs::NativeAnalysis imported;\n"
        "    if (failed(oahs::testing::analyzeSelectedHandoffSync(function, imported))) "
        "{ accepted = false; return; }\n"
        '    if (function.getSymName().ends_with("_aic")) diagnoseSlots(imported);\n'
        "    auto metadata = diagnosticModel(imported);\n"
        "    oahs::SelectedPlan report;\n    const auto status =",
    )
    source = replace_once(
        source, "    accepted &= succeeded(status);",
        "    accepted &= succeeded(status);\n"
        "    diagnosticPlan(function.getSymName(), std::move(metadata), report);",
    )
    probe = out / "probe.cpp"
    probe.write_text(source)
    cmake = build / "tools/pto-test-opt/CMakeFiles/pto-oahs-selected-test.dir"
    flags = dict(
        line.split(" = ", 1)
        for line in (cmake / "flags.make").read_text().splitlines()
        if " = " in line
    )
    link = shlex.split((cmake / "link.txt").read_text())
    compile_cmd = [link[0]]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        compile_cmd.extend(shlex.split(flags[key]))
    compile_cmd += [
        "-O2", "-fexceptions", "-I" + str(out), "-I" + str(here), "-I" + str(repo / "test/oahs"),
        "-c", str(probe), "-o", str(out / "probe.o"),
    ]
    object_name = "CMakeFiles/pto-oahs-selected-test.dir/pto-oahs-selected-test.cpp.o"
    if link.count(object_name) != 1:
        raise RuntimeError("native link recipe changed")
    link = [str(out / "probe.o") if x == object_name else x for x in link
            if not x.startswith("-Wl,--dependency-file=")]
    link[link.index("-o") + 1] = str(out / "probe")
    (out / "build.json").write_text(json.dumps({"compile": compile_cmd, "link": link}, indent=2))
    # Serial compilation, linking and cases; no hidden parallel workers.
    subprocess.run(compile_cmd, check=True)
    subprocess.run(link, cwd=build / "tools/pto-test-opt", check=True)
    for path, digest, label in inputs:
        with (out / f"{label}.log").open("w") as log, \
                (out / f"{label}-native.pto").open("w") as pto:
            subprocess.run([str(out / "probe"), "--construct", str(path)],
                           stdout=pto, stderr=log, check=True)
        lines = (out / f"{label}.log").read_text().splitlines()
        models = [json.loads(line.removeprefix("DIAGNOSTIC_JSON "))
                  for line in lines if line.startswith("DIAGNOSTIC_JSON ")]
        names = {model["function"] for model in models}
        required = {"native_relay"}
        if not required <= names or not all(model["success"] for model in models):
            raise RuntimeError(f"missing or failed experiment: {label}")
        if "FINITE cases=6" not in lines:
            raise RuntimeError(f"incomplete finite checks: {label}")
        (out / f"{label}-models.json").write_text(json.dumps(models, indent=2) + "\n")
        summary = [line for line in lines if line.startswith(
            ("NATIVE_RESULT", "FINITE", "MUTATION"))]
        record = {"input": str(path), "sha256": digest, "checks": summary}
        (out / f"{label}-summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(label + ":\n" + "\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
