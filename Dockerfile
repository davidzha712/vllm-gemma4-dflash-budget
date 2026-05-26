# vLLM v0.21.0 + PR #41703 (Gemma 4 DFlash) + CUDA-graph alignment patch
# ----------------------------------------------------------------------------
# Goal: real `thinking_token_budget` enforcement under DFlash speculative
# decoding, on Blackwell-class GPUs (sm_120 / sm_121a — DGX Spark GB10,
# GB200, etc.).
#
# Layers:
#   1. v0.21.0 base  — PR #34668 (engine-level spec-decode + thinking_budget
#                      compatibility), aarch64 wheels, SM_120 CUTLASS NVFP4.
#   2. PR #41703     — Gemma 4 DFlash batched verification (still open upstream
#                      at time of writing).
#   3. Patch         — patches/patch_cudagraph_align.py: DFlash k=15 PIECEWISE
#                      CUDA-graph capture-size alignment fix; prevents
#                      cudaErrorIllegalAddress on partial-acceptance decode.
#                      Adapted from the public AEON-7/Qwen3.6-NVFP4-DFlash
#                      patch set (Apache-2.0). See ATTRIBUTIONS.md.
#
# Build (native arm64 on a Blackwell host):
#
#     docker buildx build --platform=linux/arm64 \
#         -t ghcr.io/<you>/vllm-gemma4-dflash-budget:v0.21 \
#         --load .
#
# Multi-arch (cross-build with QEMU; slower):
#
#     docker buildx build --platform=linux/arm64,linux/amd64 \
#         -t <registry>/vllm-gemma4-dflash-budget:v0.21 \
#         --push .
# ----------------------------------------------------------------------------

FROM vllm/vllm-openai:v0.21.0

LABEL org.opencontainers.image.title="vLLM v0.21 + Gemma 4 DFlash + thinking_token_budget"
LABEL org.opencontainers.image.description="vLLM v0.21.0 mainline + PR #41703 (Gemma 4 DFlash) + PIECEWISE CUDA-graph alignment fix. Real thinking_token_budget enforcement under speculative decoding on sm_120 / sm_121a."
LABEL org.opencontainers.image.source="https://github.com/davidzha712/vllm-gemma4-dflash-budget"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# ---------------------------------------------------------------------------
# Layer 2 — cherry-pick PR #41703 (Gemma 4 DFlash batched verification)
# ---------------------------------------------------------------------------
# PR #41703 is still open upstream. We pull the PR head and replace the
# specific files it adds/modifies. Since v0.21.0 was branched AFTER PR #34668
# but BEFORE PR #41703, the PR head adds new Gemma 4 DFlash files; conflicts
# should be minimal.
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; \
    SITE_PKG="$(python3 -c 'import site;print(site.getsitepackages()[0])')"; \
    git clone --depth=50 https://github.com/vllm-project/vllm.git /tmp/vllm-pr41703; \
    cd /tmp/vllm-pr41703; \
    git fetch origin pull/41703/head:pr-41703 --depth=20; \
    git checkout pr-41703; \
    # Files PR #41703 adds or modifies for Gemma 4 DFlash:
    for f in \
        vllm/model_executor/models/gemma4.py \
        vllm/v1/spec_decode/gemma4.py \
        vllm/model_executor/models/config.py \
    ; do \
        if [ -f "$f" ]; then \
            install -D "$f" "${SITE_PKG}/$f"; \
            echo "applied: $f"; \
        else \
            echo "PR #41703 didn't ship $f (skipping)"; \
        fi; \
    done; \
    rm -rf /tmp/vllm-pr41703

# ---------------------------------------------------------------------------
# Layer 3 — PIECEWISE CUDA-graph alignment patch (DFlash k=15 stability)
# ---------------------------------------------------------------------------
# DFlash num_speculative_tokens=15 needs CUDA-graph capture-sizes aligned to
# multiples of (1+K). Without this, partial-acceptance decode steps dispatch
# to a misaligned cached graph -> cudaErrorIllegalAddress.
# See patches/patch_cudagraph_align.py for the full upstream-bug trace.
# ---------------------------------------------------------------------------
COPY patches/patch_cudagraph_align.py /tmp/patch_cudagraph_align.py
RUN python3 /tmp/patch_cudagraph_align.py && rm /tmp/patch_cudagraph_align.py

# ---------------------------------------------------------------------------
# Sanity check — imports only, no GPU access during build
# ---------------------------------------------------------------------------
RUN python3 -c "import vllm; \
    from vllm.config.reasoning import ReasoningConfig; \
    from vllm.sampling_params import SamplingParams; \
    sp = SamplingParams(); \
    assert hasattr(sp, 'thinking_token_budget'), 'thinking_token_budget missing'; \
    print(f'vllm {vllm.__version__} OK — thinking_token_budget exposed; ReasoningConfig + alignment patch loaded')"

# Blackwell (sm_120 / sm_121a) target — adjust if building for other arches
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV VLLM_TEST_FORCE_FP8_MARLIN="1"
ENV VLLM_NVFP4_GEMM_BACKEND="marlin"

# Default runtime env (overridable in your deployment manifest)
ENV VLLM_LOGGING_LEVEL="INFO"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
