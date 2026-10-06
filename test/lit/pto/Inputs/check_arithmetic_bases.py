# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check physical-base identity and relative byte relations under both alias policies."""
import json
import shutil
import subprocess
import sys
from arithmetic_membership import membership_index


def access(document, kind, site, base, byte, coordinates=(), parameters=(), space=1):
    # The identity components are tags, not dimensions of the integer system.
    selected = dict(document)
    selected["relations"] = [r for r in document["relations"]
                             if r["space"] == space and r["base_argument"] == base]
    contains = membership_index(selected)
    return contains((kind, site, -1, 0, 0), (*coordinates, byte, *parameters))


def check_same_base(document):
    assert len(document["sites"]) == 2 and document["parameters"] == [1]
    for n in range(-1, 5):
        for i in range(-1, 5):
            for byte in range(-1, 21):
                expected = 0 <= i < n and 4 * i <= byte < 4 * i + 4
                assert access(document, 4, 0, 0, byte, (i,), (n,)) == expected
                assert access(document, 5, 1, 0, byte, (i,), (n,)) == expected
                assert not access(document, 4, 0, 1, byte, (i,), (n,))


def check_views(document):
    assert len(document["sites"]) == 2 and not document["parameters"]
    for byte in (-1, 0, 1, 510, 511, 512, 513, 514, 1023, 1024):
        assert access(document, 4, 0, 0, byte) == (512 <= byte < 1024)
        assert access(document, 5, 0, -1, byte, space=2) == (0 <= byte < 512)
        assert access(document, 4, 1, 0, byte) == (512 <= byte < 514)


def check_local(document):
    assert len(document["sites"]) == 2
    for byte in range(-1, 9):
        assert access(document, 5, 0, -1, byte, space=6) == (0 <= byte < 4)
        assert access(document, 4, 1, -1, byte, space=6) == (0 <= byte < 4)
    assert all(r["base_argument"] == -1 for r in document["relations"])


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    for policy in ("may-not-alias", "may-alias"):
        run = subprocess.run([tool, "--gm-alias=" + policy, "--arithmetic", sys.argv[2]],
                             check=True, capture_output=True, text=True, timeout=60)
        documents = {d["function"]: d for line in run.stdout.splitlines()
                     if line.startswith("arithmetic-json ")
                     for d in [json.loads(line.removeprefix("arithmetic-json "))]}
        assert len(documents) == 7
        check_same_base(documents["same_base"])
        check_views(documents["same_base_views"])
        check_local(documents["local_reuse"])
        rejected = ["carried_base"]
        for name in ["selected_base", "absolute_mixture", "distinct_bases"]:
            assert documents[name]["sites"], documents[name]
        document = documents["distinct_bases"]
        assert len(document["sites"]) == 2
        for byte in range(-1, 9):
            assert access(document, 4, 0, 0, byte) == (0 <= byte < 4)
            assert access(document, 5, 1, 1, byte) == (0 <= byte < 4)
            assert not access(document, 4, 0, 1, byte)
            assert not access(document, 5, 1, 0, byte)
        for name in rejected:
            assert not documents[name]["sites"], (policy, name)
            assert not documents[name]["relations"], (policy, name)
            assert not documents[name]["parameters"], (policy, name)
    print("arithmetic physical-base checks passed under both GM policies")


if __name__ == "__main__":
    main()
