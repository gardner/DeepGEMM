# DeepGEMM GB10 Finish Plan

## Goal

Finish our DeepGEMM fork as a durable GB10/DGX Spark dependency: build and publish an aarch64 wheel that supports SM121 at runtime, supplies the DeepGEMM kernels vLLM needs, and makes any remaining NVFP4 scope explicit instead of relying on local patches.

## Upstream Findings

- There are no direct upstream `sm121` / `sm_121` issues or PRs on `main`, but [issue #236](https://github.com/deepseek-ai/DeepGEMM/issues/236) contains multiple GB10/SM121 reports. The failure is host dispatch, especially `hyperconnection.hpp:56 Unsupported architecture`, not just a build flag.
- Maintainers said they do not have SM120 hardware and prefer a community-maintained fork for workstation Blackwell support.
- [PR #318](https://github.com/deepseek-ai/DeepGEMM/pull/318) is a correctness-first SM120/SM121 proof of concept validated by community users on RTX PRO 6000 and DGX Spark.
- [PR #324](https://github.com/deepseek-ai/DeepGEMM/pull/324) is the serious NVIDIA DevTech SM120 implementation against `nv_dev`. It adds dense FP8/FP4, grouped FP8/FP4, BF16, einsum, HyperConnection, FP8/FP4 MQA logits, tests, and an SM121 compatibility commit that reuses the SM120 family cubin with NVCC/NVRTC >= 12.9.
- [issue #329](https://github.com/deepseek-ai/DeepGEMM/issues/329) says upstream currently supports MXFP4 only and has no NVFP4 plan. Treat native NVFP4 in DeepGEMM as separate work unless vLLM can consume DeepGEMM's existing packed E2M1 + UE8M0 FP4 format.
- [issue #223](https://github.com/deepseek-ai/DeepGEMM/issues/223) confirms ARM wheels are not published upstream today. [issue #330](https://github.com/deepseek-ai/DeepGEMM/issues/330) shows wheels must be rebuilt against the exact PyTorch build used by the target image.
- CUDA 13 support exists upstream via [PR #174](https://github.com/deepseek-ai/DeepGEMM/pull/174) and [PR #157](https://github.com/deepseek-ai/DeepGEMM/pull/157). Our path should stay on CUDA 13.x and NVCC/NVRTC >= 12.9.

## Local State

- Work branch `gb10/deepgemm-sm120-import` is based on `upstream/nv_dev`.
- The branch preserves our GB10 release workflow discipline and imports PR #324 from `leavelet/sm120`.
- Architecture 12 now dispatches to SM120 kernels for GEMM, BF16, einsum, HyperConnection, and MQA logits paths covered by PR #324.
- `.github/workflows/gb10-release.yml` builds an aarch64 SM121A-labeled wheel, but DeepGEMM kernels are JIT-compiled at runtime. The critical runtime behavior is in `csrc/jit/device_runtime.hpp` and `csrc/jit/compiler.hpp`.
- CUDA 13.0 is the validated local toolkit. `/usr/local/cuda` currently points at a partial CUDA 13.2 layout, so local validation should set `CUDA_HOME=/usr/local/cuda-13.0`.
- GitHub Actions release builds use CUDA Toolkit `13.0.2`, which is the CUDA 13.0 update available through `Jimver/cuda-toolkit@v0.2.29`.

## Local Validation

- Static GB10 discipline tests pass:
  - `uv run --with pytest python -m pytest -q tests/test_gb10_release_discipline.py`
- The first release tag run `gb10-deepgemm-v13efe6d` failed before build because `Jimver/cuda-toolkit@v0.2.26` did not provide CUDA `13.0.0`; the workflow now defaults to CUDA `13.0.2` and action `v0.2.29`.
- Local aarch64 wheel builds against PyTorch CUDA 13 nightly using PyPI, PyTorch nightly, and NVIDIA PyPI as indexes:
  - `deep_gemm-2.5.0-cp313-cp313-linux_aarch64.whl`
- Installed-wheel smoke from outside the source tree passes on GB10:
  - PyTorch `2.13.0.dev20260530+cu130`
  - CUDA `13.0`
  - capability `(12, 1)`
  - `deep_gemm.get_num_sms() == 48`
- HyperConnection JIT correctness smoke passes, and NVCC logs show native SM12-family codegen:
  - `-gencode=arch=compute_120f,code=sm_120f`
  - `ptxas ... for 'sm_120f'`
  - observed `hc_prenorm_diff tensor(1.4403e-10, device='cuda:0', dtype=torch.float64)`
- Full upstream HyperConnection script passes from the built wheel outside the source tree:
  - `PYTHONPATH=/mnt/dgx-ssd/src/GB10/DeepGEMM/tests python /mnt/dgx-ssd/src/GB10/DeepGEMM/tests/test_hyperconnection.py`
  - Note: `pytest` collection against this file also collects imported helper `deep_gemm.testing.test_filter`, so use the script entrypoint or adjust upstream test naming before treating that pytest collection behavior as a kernel failure.
- The SM100 MegaMoE JIT path cannot be reused directly on GB10. Dispatching architecture 12 into `sm100_fp8_fp4_mega_moe` makes NVCC target `sm_120f`, but `ptxas` rejects the SM100-only `tcgen05.*`, `.cta_group::2`, and `.block32` instructions.
- A correctness-first SM120 MegaMoE path now runs on GB10 by composing the imported SM120 grouped FP8/FP4 GEMMs with PyTorch staging, SwiGLU, FP8 recast, and local combine logic:
  - Static release discipline tests pass.
  - The composed path JIT logs show both L1 and L2 grouped GEMMs compiling for `sm_120f`.
  - A single-rank deterministic smoke passes on GB10 with finite nonzero output and `stats_sum == num_tokens * num_topk`.
  - A small dequantized-reference accuracy smoke passes with `diff ~= 2.6e-05`.
  - The composed path pads per-expert GEMM work buffers to the SM120 grouped GEMM tile floor (`64`) while preserving the real expert token counts as masks.

## Definition Of Done

- A GB10 integration branch is based on upstream `nv_dev`, includes our GB10 release workflow/tests, and includes the SM120/SM121 implementation from PR #324 or an equivalent reviewed import.
- On GB10 SM121, JIT compile logs show the intended SM12 target, preferably `sm_120f` with CUDA >= 12.9, not accidental SM90/SM100 fallback.
- Correctness tests pass on GB10 for:
  - `tests/test_fp8_fp4.py`
  - `tests/test_bf16.py`
  - `tests/test_einsum.py`
  - `tests/test_hyperconnection.py`
  - `tests/test_attention.py`
  - `tests/test_split_k_swap.py` from PR #324
- The fork publishes a GitHub Release wheel built on aarch64 against our target PyTorch/CUDA combination.
- vLLM can install that wheel from a release URL or git ref and pass a DeepGEMM smoke test inside the GB10 container.
- NVFP4 status is explicit: either DeepGEMM supports the exact NVFP4 layout vLLM sends, or vLLM does not route NVFP4 through DeepGEMM and the README/release notes say so.

## Implementation Order

1. Create an integration branch from `upstream/nv_dev`.
2. Reapply our GB10 release workflow and `tests/test_gb10_release_discipline.py`.
3. Import PR #324 from `leavelet/sm120`. Prefer a merge or ordered cherry-pick preserving commit history because the PR has many bug-fix commits after the first kernel drop.
4. Add static tests before resolving runtime details:
   - assert `sm120` runtime/header files are packaged;
   - assert architecture 12 dispatch exists for HyperConnection and GEMM;
   - assert compiler code can map SM121 to SM120 family mode when supported.
5. Resolve compile conflicts and run no-GPU tests:
   - `python -m pytest tests/test_gb10_release_discipline.py`
   - `python setup.py bdist_wheel --dist-dir=dist`
6. Run GB10 hardware validation with compiler logging:
   - `DG_JIT_PRINT_COMPILER_COMMAND=1 DG_JIT_PTXAS_VERBOSE=1 pytest -q tests/test_hyperconnection.py`
   - then the full correctness list above.
7. Check NVFP4/MXFP4 compatibility against vLLM tensor layouts. If incompatible, decide whether to add a DeepGEMM NVFP4 adapter/kernel or keep NVFP4 routed through another backend.
8. Dispatch the release workflow:
   - `gh workflow run gb10-release.yml -R gardner/DeepGEMM -f torch-wheel-url=<url> -f cuda-version=13.0.0`
9. Download the wheel, install it into the vLLM fork/container, and run a vLLM DeepGEMM probe before tagging the dependency in the root stack.

## Known Gaps And Risks

- PR #324 does not add SM120 MegaMoE API dispatch. The current fork has an accurate correctness-first SM120 composed path, but the performant endpoint remains a real fused-kernel project.
- PR #324 supports FP4/MXFP4-style paths, while upstream explicitly says no NVFP4 plan. This is the biggest strategy check for native NVFP4.
- Multi-rank MegaMoE all-gather/combine behavior still needs EP validation; current GB10 validation is single-rank.
- The release workflow must build against the exact PyTorch ABI used in the final vLLM image.
- GitHub-hosted arm64 runners plus CUDA toolkit installation may be fragile; keep release workflow manual and artifact-focused until proven.
- JIT cache keys include compiler signature, flags, and code. After architecture mapping changes, clear `~/.deep_gemm/cache` during validation to avoid stale cubins.

## Acceptance Checklist

- [x] Branch based on `upstream/nv_dev`.
- [x] GB10 release workflow preserved and passing static tests.
- [x] SM120/SM121 kernels imported.
- [x] Local wheel builds.
- [x] GB10 JIT logs show expected SM12 target.
- [x] Single-rank SM120 MegaMoE composed path runs on GB10.
- [ ] GB10 correctness suite passes.
- [ ] Fused SM120 MegaMoE kernel replaces the composed staging path.
- [ ] Multi-rank MegaMoE validation passes.
- [ ] NVFP4 routing decision documented.
- [ ] GitHub Release wheel published.
- [ ] vLLM container consumes the published wheel successfully.
