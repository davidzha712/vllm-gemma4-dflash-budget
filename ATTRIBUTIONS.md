# Attributions

This repository is an integration of three pieces of open-source work. None of
the core algorithms are original to this project — credit belongs to:

## vLLM upstream

- Project: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- License: Apache-2.0
- Used: base image `vllm/vllm-openai:v0.21.0`, which includes
  [**PR #34668**](https://github.com/vllm-project/vllm/pull/34668) — the
  engine-level refactor that finally makes `thinking_token_budget` compatible
  with speculative decoding. Without that PR the budget knob is silently
  bypassed; *that* is the single most important upstream fix this image
  depends on.
- Related upstream issues: [#39573](https://github.com/vllm-project/vllm/issues/39573),
  [#20859](https://github.com/vllm-project/vllm/pull/20859) (original budget
  enforcement that PR #34668 obsoletes).

## vLLM PR #41703 — Gemma 4 DFlash batched verification

- PR: [vllm-project/vllm#41703](https://github.com/vllm-project/vllm/pull/41703)
- Status at time of writing: **open, not merged upstream**.
- License: Apache-2.0 (inherits from the vLLM project).
- Used: three files are cherry-picked from the PR head into the installed
  vLLM dist-package:
  - `vllm/model_executor/models/gemma4.py`
  - `vllm/v1/spec_decode/gemma4.py`
  - `vllm/model_executor/models/config.py`

  This image will pin the PR head as of the build date. When PR #41703 merges
  upstream and ships in a vLLM release, this layer should be dropped.

## AEON-7 / Qwen3.6-NVFP4-DFlash patch set

- Source: [github.com/AEON-7](https://github.com/AEON-7) — the
  `Qwen3.6-NVFP4-DFlash` patch set published as part of their NVFP4 DFlash
  serving image.
- License: Apache-2.0.
- Used: a single patch, `patches/patch_cudagraph_align.py`, adapted from
  AEON-7's overlay of the same name. It removes the `cudagraph_mode=FULL`
  gate on spec-decode capture-size alignment in `vllm/config/compilation.py`
  so PIECEWISE mode also gets the alignment, preventing
  `cudaErrorIllegalAddress` on partial-acceptance decode steps when
  `num_speculative_tokens=K` and capture sizes aren't multiples of `(1+K)`.
- Related upstream vLLM tickets: #28015, #28207, #29091, PR #29102, PR #23679.

The other patches in AEON-7's set (`patch_mrope_text_fallback.py`,
`patch_cuda_optional_import.py`, `patch_kv_cache_utils.py`,
`register_qwen3_5_text.py`, `strip_language_model_prefix.py`) are
Qwen3.6-specific or legacy backports for older vLLM bases and are **not** used
here. See `README.md` for the full skip list and reasoning.

## Models referenced

- **Gemma 4** — Google. Weights and license terms governed by Google's
  Gemma Terms of Use. This repository does not redistribute weights.
- **DFlash** — speculative-decoding drafter family from
  [z-lab/dflash](https://github.com/z-lab/dflash).

## Maintainer

- @davidzha712 — integration, testing on DGX Spark GB10, packaging.

The integration work itself (this Dockerfile, the public README, the
deployment template, and the test methodology in the README) is © 2026
David Zha, released under Apache-2.0.
