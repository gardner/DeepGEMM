from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def workflow_header(path: str) -> str:
    return "\n".join(read(path).splitlines()[:24])


def test_gb10_release_workflow_builds_native_sm121a_wheel():
    workflow = read(".github/workflows/gb10-release.yml")

    assert "gb10-deepgemm-v*" in workflow
    assert 'default: "13.0.2"' in workflow
    assert "Jimver/cuda-toolkit@v0.2.29" in workflow
    assert 'TORCH_CUDA_ARCH_LIST: "12.1a"' in workflow
    assert 'CMAKE_CUDA_ARCHITECTURES: "121a"' in workflow
    assert "torch-wheel-url" in workflow
    assert "cuobjdump" in workflow
    assert "gh release upload" in workflow


def test_upstream_publish_matrix_is_manual_only_for_the_fork():
    header = workflow_header(".github/workflows/publish.yml")

    assert "workflow_dispatch:" in header
    assert "create:" not in header


def test_sm121_runtime_reuses_sm120_family_cubin_when_supported():
    runtime = read("csrc/jit/device_runtime.hpp")
    compiler = read("csrc/jit/compiler.hpp")

    assert "set_support_arch_family" in runtime
    assert "major == 12" in runtime
    assert 'return "120";' in runtime
    assert 'return support_arch_family ? "120f" : "120a";' in runtime
    assert "device_runtime->set_support_arch_family" in compiler
    assert "-gencode=arch=compute_{}" in compiler


def test_architecture_12_dispatches_to_sm120_kernels():
    dispatch_files = {
        "csrc/apis/gemm.hpp": [
            "sm120_fp8_fp4_gemm_1d1d",
            "sm120_bf16_gemm",
            "arch_major == 12",
        ],
        "csrc/apis/hyperconnection.hpp": [
            "sm120_tf32_hc_prenorm_gemm",
            "arch_major == 12",
        ],
        "csrc/apis/attention.hpp": [
            "sm120_fp8_fp4_gemm_1d1d",
            "sm120_fp4_mqa_logits",
            "arch_major == 12",
        ],
        "csrc/apis/einsum.hpp": [
            "sm120_bmn_bnk_mn_gemm",
            "sm120_fp8_fp4_bmm",
            "arch_major == 12",
        ],
    }

    for path, needles in dispatch_files.items():
        source = read(path)
        for needle in needles:
            assert needle in source, f"{needle!r} missing from {path}"


def test_sm120_runtime_headers_are_packaged():
    required_paths = [
        "csrc/jit_kernels/heuristics/sm120.hpp",
        "csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp",
        "csrc/jit_kernels/impls/sm120_bf16_gemm.hpp",
        "csrc/jit_kernels/impls/sm120_bmk_bnk_mn.hpp",
        "csrc/jit_kernels/impls/sm120_tf32_hc_prenorm_gemm.hpp",
        "deep_gemm/include/deep_gemm/common/sm120_utils.cuh",
        "deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh",
        "deep_gemm/include/deep_gemm/impls/sm120_bf16_gemm.cuh",
        "deep_gemm/include/deep_gemm/impls/sm120_bmk_bnk_mn.cuh",
        "deep_gemm/include/deep_gemm/impls/sm120_tf32_hc_prenorm_gemm.cuh",
        "deep_gemm/include/deep_gemm/mma/sm120.cuh",
        "tests/test_split_k_swap.py",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert not missing

    setup_py = read("setup.py")
    assert "'include/deep_gemm/**/*'" in setup_py


def test_sm121_megamoe_uses_composed_sm120_path():
    mega = read("deep_gemm/mega/__init__.py")

    assert "_fp8_fp4_mega_moe_sm120" in mega
    assert "_untranspose_sf_from_utccp" in mega
    assert "gemm_m = max(expected_m, 64)" in mega
    assert "torch.cuda.get_device_capability(y.device)[0] == 12" in mega
    assert "m_grouped_fp8_fp4_gemm_nt_masked" in mega
