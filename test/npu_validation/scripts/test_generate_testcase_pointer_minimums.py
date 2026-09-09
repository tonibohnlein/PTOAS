#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "generate_testcase",
    SCRIPT_DIR / "generate_testcase.py",
)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


A5_BOARD_FAILURE_MINIMUMS = {
    ("Qwen3DecodeA5", "sv_matmul"): (262_144, 524_288, 67_108_864),
    ("Qwen3_32BDecode4DA5", "down_proj"): (409_600, 209_715_200, 131_072),
    ("Qwen3_32BDecode4DA5", "k_proj"): (131_072, 8_388_608, 16_384),
    ("Qwen3_32BDecode4DA5", "out_proj"): (131_072, 67_108_864, 131_072),
    ("Qwen3_32BDecode4DA5", "v_proj"): (131_072, 8_388_608, 16_384),
    ("DeepseekV4ProPrefillA5", "rope"): (8_388_608, 1_048_576, 8_192, 8_192),
    ("DeepseekV4ProPrefillA5", "prefill_idx_qr_proj"): (8_192, 1_048_576, 196_608, 12_582_912, 128),
    ("DeepseekV4ProDecodeA5", "hca_cache_topk"): (8, 4, 1_024),
    ("DeepseekV4FlashMtpPrefillA5", "merge_rope_pack"): (
        1_024, 1_024, 8_192, 8_192, 4_194_304, 4_194_304, 1, 1, 512, 64,
    ),
    ("DeepseekV4FlashMtpPrefillA5", "prefill_idx_c4_rmsnorm_rope"): (
        32, 32, 1_048_576, 1_048_576, 4_096, 128, 4_096,
    ),
    ("DeepseekV4FlashMtpPrefillA5", "prefill_idx_qr_proj"): (
        131_072, 8_388_608, 8_192, 1_048_576, 128,
    ),
    ("DeepseekV4FlashDsparkA5", "gather_ori_kv"): (25_165_824, 16_384, 512),
    ("DeepseekV4FlashDsparkA5", "lm_head_combine_gather"): (16_547_840, 16_547_840),
    ("DeepseekV4FlashDsparkA5", "merge_rope_pack"): (
        1_024, 1_024, 8_192, 8_192, 4_194_304, 4_194_304, 1, 1, 512, 64,
    ),
    ("DeepseekV4FlashDsparkA5", "prefill_idx_qr_proj"): (
        131_072, 8_388_608, 8_192, 1_048_576, 128,
    ),
    ("DeepseekV4DecodeA5", "hca_rope"): (128, 128, 512, 512, 8, 1_048_576, 1_048_576),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_1"): (
        1_048_576, 117_440_512, 458_752,
    ),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_5"): (
        458_752, 132_120_576, 8_192,
    ),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_6"): (
        458_752, 132_120_576, 8_192,
    ),
    ("DeepseekV3_2PrefillBackA5", "deepseek_v3_2_prefill_back_layer_incore_8"): (
        1_179_648, 132_120_576, 8_192,
    ),
    ("DeepseekV3_2DecodeFrontA5", "decode_cache_write"): (
        16, 262_144, 262_144, 8_192, 9_216, 33_554_432, 4_194_304,
    ),
    ("DeepseekV3_2DecodeFrontA5", "kv_a_proj"): (114_688, 4_128_768, 1_024),
    ("DeepseekV3_2DecodeFrontA5", "q_head_proj"): (24_576, 37_748_736, 1_024),
    ("DeepseekV3_2DecodeFrontA5", "q_lora_proj"): (114_688, 11_010_048, 24_576),
    ("DeepseekV3_2DecodeFrontA5", "s2_k_idx_proj"): (114_688, 917_504, 1_024),
    ("DeepseekV3_2DecodeFrontA5", "s2_q_idx_proj"): (24_576, 12_582_912, 2_048),
    ("DeepseekV3_2DecodeFrontA5", "s4_dispatch"): (1, 33_554_432, 262_144),
    ("DeepseekV3_2DecodeFrontA5", "s4_softmax"): (
        32_768, 33_554_432, 4_194_304, 8_192, 1_024, 8_192,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_0"): (
        262_144, 117_440_512, 1_024,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_3"): (
        114_688, 132_120_576, 4_096,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_4"): (
        114_688, 132_120_576, 4_096,
    ),
    ("DeepseekV3_2DecodeBackA5", "deepseek_v3_2_decode_back_layer_incore_6"): (
        294_912, 132_120_576, 2_048,
    ),
}

A3_BOARD_FAILURE_MINIMUMS = {
    ("DeepseekV4DecodeA3", "gate"): (64, 65_536, 262_144, 16, 4_096, 2_048),
    ("DeepseekV4DecodeA3", "qk_pv"): (
        1_024, 524_288, 512, 512, 262_144, 262_144, 32_768,
    ),
    ("DeepseekV4DecodeA3", "score"): (
        4, 8, 65_536, 512, 1_024, 32_768, 256, 16_384, 128, 16_384,
    ),
    ("Qwen3_14BPrefillA3", "lm_head"): (152_064, 81_920, 778_567_680),
    ("Qwen3_14BPrefillA3", "qk_pv_online_phase"): (
        8_192, 8_192, 1_048_576, 1, 131_072, 131_072, 1_048_576, 16_384,
    ),
    ("Qwen3_14BPrefillA3", "qk_pv_skew_probe"): (
        8_192, 1_048_576, 1, 131_072, 131_072, 1_048_576, 327_680, 32_768,
    ),
}

A3_MIXED_SCALAR_DEFAULTS = {
    ("Qwen3_14BPrefillA3", "out_proj_aiv"): {
        "v4": 0,
        "v5": 64,
        "v6": 0,
        "v7": 1,
    },
    ("DeepseekV4DecodeA3", "gate"): {"v7": 0, "v8": 1},
    ("DeepseekV4DecodeA3", "qk_pv"): {"v8": 0, "v9": 1},
    ("DeepseekV4DecodeA3", "score"): {"v11": 1, "v12": 0, "v13": 1},
    ("Qwen3_14BPrefillA3", "qk_pv_online_phase"): {
        "v9": 1,
        "v10": 0,
        "v11": 0,
        "v12": 1,
        "v13": 0,
        "v14": 0,
        "v15": 0,
        "v16": 0,
        "v17": 1,
        "v18": 128,
        "v19": 0,
        "v20": 1,
    },
    ("Qwen3_14BPrefillA3", "qk_pv_skew_probe"): {
        "v9": 0,
        "v10": 0,
        "v11": 1,
        "v12": 0,
        "v13": 0,
        "v14": 0,
        "v15": 1,
        "v16": 128,
        "v17": 0,
        "v18": 1,
    },
}


def _generate_case(root: Path, sample: str, testcase: str, expected_count: int,
                   layout: str = "", body: str = "") -> Path:
    sample_dir = root / sample
    kernel_dir = sample_dir / layout
    kernel_dir.mkdir(parents=True, exist_ok=True)
    params = ", ".join(f"__gm__ float* v{index}" for index in range(1, expected_count + 1))
    kernel = kernel_dir / f"{testcase}-pto.cpp"
    kernel.write_text(
        f'extern "C" __global__ AICORE void {testcase}({params}) {{{body}}}\n',
        encoding="utf-8",
    )
    output_root = root / "generated"
    GENERATOR.generate_testcase(kernel, output_root, testcase, "npu", "Ascend950")
    return output_root / sample / testcase


def _counts(output: Path, expected_count: int) -> tuple[int, ...]:
    main_cpp = (output / "main.cpp").read_text(encoding="utf-8")
    return tuple(
        int(re.search(rf"elemCount_v{index} = (\d+);", main_cpp).group(1))
        for index in range(1, expected_count + 1)
    )


def _generate_counts(root: Path, sample: str, testcase: str, expected_count: int) -> tuple[int, ...]:
    return _counts(_generate_case(root, sample, testcase, expected_count), expected_count)


def _check_layouts(root: Path) -> None:
    sample = root / "Qwen3DecodeA3"
    layouts = ("", "npu_validation", "npu_validation/case", "kernels/aic", "kernels/aiv")
    runner = (SCRIPT_DIR / "run_remote_npu_validation.sh").read_text()
    start = runner.index('  case_dir=')
    naming = runner[start:runner.index('  sample_name_lc=', start)]
    assert "--print-sample-root" in naming
    for index, layout in enumerate(layouts):
        path = sample / layout / "q_proj-pto.cpp"
        assert GENERATOR._resolve_sample_root(path) == sample, path
        resolved = subprocess.check_output(
            [sys.executable, str(SCRIPT_DIR / "generate_testcase.py"), "--input", str(path), "--print-sample-root"],
            text=True,
        ).strip()
        assert resolved == str(sample), (path, resolved)
        output = _generate_case(root / str(index), "Qwen3DecodeA3", "q_proj", 3, layout)
        assert _counts(output, 3) == (131_072, 67_108_864, 131_072)
        kernel = root / str(index) / sample.name / layout / "q_proj-pto.cpp"
        # Exercise the actual runner naming fragment, without its build/device
        # setup. Spaces in the fixture root also check shell argument quoting.
        name = subprocess.check_output(
            ["bash", "-c", 'set -eu\n' + naming + '\nprintf "%s\\n" "$sample_name"'],
            env=dict(os.environ, cpp=str(kernel), ROOT_DIR=str(SCRIPT_DIR.parents[2])), text=True).strip()
        assert name == sample.name, (kernel, name)
        assert not (output.parent.parent / "aic" / "q_proj").exists()
        assert not (output.parent.parent / "aiv" / "q_proj").exists()
    for layout in ("other/aic", "other/aiv", "kernels/other", "aic"):
        path = sample / layout / "case.cpp"
        assert GENERATOR._resolve_sample_root(path) == path.parent, path

    # Correct sample identity also selects model assets and scalar contracts.
    sample.mkdir(parents=True, exist_ok=True)
    golden = "# custom golden from the sample root\n"
    compare = "# custom comparison from the sample root\n"
    helper = "# shared golden helper\n"
    (sample / "q_proj_golden.py").write_text(golden)
    (sample / "q_proj_compare.py").write_text(compare)
    (sample / "qwen3_decode_golden_lib.py").write_text(helper)
    output = _generate_case(root, sample.name, "q_proj", 3, "kernels/aic")
    assert (output / "golden.py").read_text() == golden
    assert (output / "compare.py").read_text() == compare
    assert (output / "qwen3_decode_golden_lib.py").read_text() == helper
    assert (output / "validation_meta.env").read_text() == "CUSTOM_GOLDEN=1\nCUSTOM_COMPARE=1\n"
    scoped = root / "Qwen3_14BPrefillA3" / "kernels/aiv/out_proj_aiv.cpp"
    assert GENERATOR._integer_scalar_default_value(
        "out_proj_aiv", "v5", "int32_t", GENERATOR._resolve_sample_root(scoped).name) == 64


def _check_allocation_floors(root: Path) -> None:
    # No footprint was inferred: a small configured minimum must not replace
    # the existing 32x32 fallback. Golden input lengths use the same counts.
    floors = {"floor_defaults": {"v1": 16, "v2": 1024, "v3": 2048}}
    with patch.dict(GENERATOR.CASE_POINTER_COUNT_MINIMUMS, floors):
        output = _generate_case(root, "UnrelatedSample", "floor_defaults", 3)
    assert _counts(output, 3) == (1024, 1024, 2048)
    golden = (output / "golden.py").read_text()
    for name, count in (("v1", 1024), ("v2", 1024), ("v3", 2048)):
        assert re.search(rf"\b{name} = np\.(?:random\.random\(size=|zeros\()\({count},\)", golden), golden

    # The same known 64-element footprint is above one floor and below another.
    # Backing storage can grow to 128 without claiming 128 defined output elements.
    body = '''
  using GTShape = pto::Shape<1, 1, 1, 8, 8>;
  using GTStride = pto::Stride<64, 64, 64, 8, 1>;
  using GT = GlobalTensor<float, GTShape, GTStride>;
  GTShape shape;
  GTStride stride;
  GT input = GT(v1, shape, stride);
  GT output = GT(v2, shape, stride);
'''
    assert GENERATOR._infer_gm_pointer_elem_counts(body, ["v1", "v2"]) == {"v1": 64, "v2": 64}
    with patch.dict(GENERATOR.CASE_POINTER_COUNT_MINIMUMS, {"floor_inferred": {"v1": 32, "v2": 128}}):
        output = _generate_case(root, "UnrelatedSample", "floor_inferred", 2, body=body)
    assert _counts(output, 2) == (64, 128)
    comparison = (output / "compare.py").read_text()
    assert re.search(r'compare_bin_prefix\("golden_v2.bin", "v2.bin", np.float32, [^,]+, 64\)', comparison), comparison


def _check_emitted_qwen(root: Path, emitted_root: Path) -> None:
    """Optional actual emitted-source replay; generates scripts, no device/data allocation."""
    sample = root / "Qwen3DecodeA3"
    for testcase, core, count in (("q_proj", "aic", 3), ("qk_matmul", "aic", 3),
                                   ("online_softmax", "aiv", 4)):
        source = emitted_root / f"{testcase}.logical.cpp"
        assert source.is_file(), source
        kernel = sample / "kernels" / core / f"{testcase}-pto.cpp"
        kernel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, kernel)
        GENERATOR.generate_testcase(kernel, root / "generated", testcase, "npu", "Ascend910B")
        output = root / "generated" / sample.name / testcase
        actual = _counts(output, count)
        floors = GENERATOR._pointer_count_minimums_for_case(sample.name, testcase)
        for index, value in enumerate(actual, 1):
            assert value >= floors[f"v{index}"], (testcase, index, value, floors)
        print(f"{sample.name}/kernels/{core}/{testcase}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emitted-root", type=Path,
                        help="Also stage actual *.logical.cpp Qwen outputs through kernels/aic|aiv")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Pointer-minimum checks require assertions")
    assert len(A5_BOARD_FAILURE_MINIMUMS) == 32
    with tempfile.TemporaryDirectory(prefix="ptoas-npu-validation-test-") as temp_dir:
        root = Path(temp_dir)
        for (sample, testcase), expected in A5_BOARD_FAILURE_MINIMUMS.items():
            actual = _generate_counts(root, sample, testcase, len(expected))
            expected = tuple(max(1024, floor) for floor in expected)
            assert actual == expected, f"{sample}/{testcase}: expected {expected}, got {actual}"

        for (sample, testcase), expected in A3_BOARD_FAILURE_MINIMUMS.items():
            actual = _generate_counts(root, sample, testcase, len(expected))
            expected = tuple(max(1024, floor) for floor in expected)
            assert actual == expected, f"{sample}/{testcase}: expected {expected}, got {actual}"

        for (sample, testcase), expected in A3_MIXED_SCALAR_DEFAULTS.items():
            actual = {
                name: GENERATOR._integer_scalar_default_value(
                    testcase,
                    name,
                    "int64_t" if name in {"v11", "v13", "v17", "v18"} else "int32_t",
                    sample,
                )
                for name in expected
            }
            assert actual == expected, f"{sample}/{testcase}: expected {expected}, got {actual}"

        isolated = _generate_counts(root, "UnrelatedSample", "sv_matmul", 3)
        assert isolated == (1_024, 1_024, 1_024), isolated
        _check_layouts(root / "layouts with spaces")
        _check_allocation_floors(root / "floors")
        if args.emitted_root:
            _check_emitted_qwen(root / "emitted", args.emitted_root.resolve())


if __name__ == "__main__":
    main()
