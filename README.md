# vllm-gemma4-dflash-budget

vLLM build that **actually enforces `thinking_token_budget`** while running
Gemma 4 (and Qwen 3.6) with DFlash speculative decoding on Blackwell-class GPUs
(sm_120 / sm_121a — DGX Spark GB10, GB200, …).

```
vllm/vllm-openai:v0.21.0          (PR #34668 — engine-level budget tracking)
   + cherry-pick vllm PR #41703   (Gemma 4 DFlash batched verification)
   + patches/patch_cudagraph_align.py  (PIECEWISE CUDA-graph alignment, k=15)
```

License: **Apache-2.0** (matches vLLM upstream).

> **Works for Qwen 3.6 too.** Empirical cross-image test on DGX Spark
> GB10 (2026-05-26) confirmed this image runs `Qwen/Qwen3.6-35B-A3B-FP8` +
> DFlash drafter with equivalent inference to a dedicated Qwen-patched build:
> identical budget gradient (10 → 30 chars; 1000 → ~3000 chars), identical
> DFlash acceptance (~30%), identical native architecture detection
> (`Qwen3_5MoeForConditionalGeneration`).
>
> Aeon-7's three Qwen-specific patches
> (`register_qwen3_5_text`, `patch_mrope_text_fallback`, `patch_kv_cache_utils`)
> were legacy backports for v0.18-era vLLM bases — v0.21.0 mainline includes
> all three fixes natively. Only `patch_cudagraph_align` is needed (already
> in this image). The companion repo
> [`vllm-qwen3-dflash-budget`](https://github.com/davidzha712/vllm-qwen3-dflash-budget)
> is now archival; use this image for both Gemma 4 and Qwen 3.6.

---

## The problem this solves

Anyone trying to run Gemma 4 with [DFlash](https://github.com/z-lab/dflash)
speculative decoding on Blackwell hits two independent issues:

1. **`thinking_token_budget` is silently ignored under speculative decoding.**
   vLLM's original budget enforcement (PR
   [#20859](https://github.com/vllm-project/vllm/pull/20859)) lived in a logits
   processor that was bypassed by multi-token spec-decode dispatch paths. See
   [issue #39573](https://github.com/vllm-project/vllm/issues/39573). Operators
   set `thinking_token_budget=10` and get 3000+ characters of reasoning — i.e.
   the knob is fake.

2. **Gemma 4 DFlash isn't merged upstream yet.** [PR
   #41703](https://github.com/vllm-project/vllm/pull/41703) adds the model
   wrapper and batched verifier; still open at time of writing.

3. **DFlash k=15 + default PIECEWISE CUDA graphs → `cudaErrorIllegalAddress`.**
   vLLM's capture-size alignment is gated to `cudagraph_mode=FULL` only, so
   PIECEWISE-mode runs hit misaligned graph dispatch on partial acceptance.
   See `patches/patch_cudagraph_align.py` for the full trace.

This image combines the three fixes into one runnable container.

## What's inside

| Layer | Source | Purpose |
|---|---|---|
| 1 | `vllm/vllm-openai:v0.21.0` | aarch64 wheels, SM_120 CUTLASS NVFP4, **PR #34668** (engine-level spec-decode + thinking_budget compatibility) |
| 2 | Cherry-pick of [vllm-project/vllm #41703](https://github.com/vllm-project/vllm/pull/41703) | Gemma 4 DFlash batched verification |
| 3 | `patches/patch_cudagraph_align.py` | DFlash k=15 PIECEWISE CUDA-graph capture-size alignment — prevents `cudaErrorIllegalAddress` |

All patches are **Python overlays** applied to the installed vLLM dist-package
files at image-build time. No `nvcc` / C++ rebuild needed. Build takes
≈15–20 minutes on a native arm64 host.

## Build

```bash
git clone https://github.com/davidzha712/vllm-gemma4-dflash-budget.git
cd vllm-gemma4-dflash-budget

# Native arm64 (recommended — on a Blackwell host)
docker buildx build --platform=linux/arm64 \
    -t ghcr.io/<you>/vllm-gemma4-dflash-budget:v0.21 \
    --load .

# Or cross-build + push to a registry
docker buildx build --platform=linux/arm64 \
    -t <registry>/vllm-gemma4-dflash-budget:v0.21 \
    --push .
```

A sanity-check stage in the Dockerfile verifies that
`SamplingParams.thinking_token_budget` is exposed and `ReasoningConfig` plus the
alignment patch loaded; failure here means one of the layers didn't take.

## Run

Point vLLM at your Gemma 4 weights and a DFlash drafter:

```bash
docker run --rm -it --gpus all -p 8000:8000 \
    -v $HOME/models:/models:ro \
    ghcr.io/<you>/vllm-gemma4-dflash-budget:v0.21 \
    vllm serve /models/gemma4-nvfp4 \
        --port 8000 \
        --served-model-name gemma4 \
        --trust-remote-code \
        --quantization modelopt \
        --max-model-len 16384 \
        --max-num-seqs 4 \
        --reasoning-parser gemma4 \
        --speculative-config='{"method":"dflash","model":"/models/gemma4-dflash","num_speculative_tokens":15}'
```

A Kubernetes deployment template (with placeholders) is in
[`deploy/test-deployment.yaml`](deploy/test-deployment.yaml).

## Verify the budget is real

The whole point of this image is that `thinking_token_budget` actually clips
reasoning length. Hit `/v1/chat/completions` with a reasoning-heavy prompt and
vary the budget:

```bash
for B in 10 50 200 1000 10000; do
    curl -sS http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"gemma4\",
            \"messages\":[{\"role\":\"user\",\"content\":\"Prove the sum of two odd numbers is always even.\"}],
            \"thinking_token_budget\": $B,
            \"max_tokens\": 4096,
            \"chat_template_kwargs\": {\"enable_thinking\": true}}" \
    | jq -r --arg b "$B" '"budget=" + $b + " reason_chars=" + (.choices[0].message.reasoning_content|length|tostring)'
done
```

On the reference build the reasoning length scales monotonically with budget:

| `thinking_token_budget` | `reason_chars` (Gemma 4 26B NVFP4 + DFlash k=15) |
|---:|---:|
| 10     | 28    |
| 50     | 197   |
| 200    | 656   |
| 1000   | 1722  |
| 10000  | 1609  (natural reasoning length below cap) |

If your numbers stay flat across budgets, one of the patch layers didn't apply.

## What's *not* included (and why)

The original AEON-7 / Qwen3.6-NVFP4-DFlash patch set ships several other
overlays. None of them are needed for Gemma 4 on vLLM v0.21.0; including them
would only add risk of conflict with upstream code that has already moved on:

| Skipped patch | Why |
|---|---|
| `patch_mrope_text_fallback.py` | Qwen3.6 multimodal-specific M-RoPE |
| `patch_cuda_optional_import.py` | Legacy backport; v0.21.0 has the upstream fix |
| `patch_kv_cache_utils.py` | Legacy for bases lacking `mamba_block_size` defaulting |
| `register_qwen3_5_text.py` | Qwen3.6 v1 weight-layout registration |
| `strip_language_model_prefix.py` | Weight conversion script, not a vLLM patch |

## Hardware tested

- DGX Spark GB10 (sm_121a, 128 GB unified memory), Ubuntu 24.04 ARM64
- DGX Spark cluster with NVFP4 Gemma 4 26B + DFlash drafter

Should also work on any sm_120 Blackwell (GB200, B100/B200) — same vLLM env
vars. Adjust `TORCH_CUDA_ARCH_LIST` if building for older arches.

## Acceptance rate

DFlash spec-decode still works after these patches — measured ~14% acceptance
rate on Gemma 4 26B NVFP4 at `num_speculative_tokens=15` with concurrency 1
on the verification prompt above. Production workloads with longer prompts
should see higher rates; tune `num_speculative_tokens` for your distribution.

## Contributing

Issues and PRs welcome. Things on the roadmap:

- [ ] Track when vLLM PR #41703 merges upstream so the cherry-pick layer can
      be dropped.
- [ ] Track when the CUDA-graph alignment fix lands upstream
      (vLLM #28015 / #29102 / #23679) so the overlay patch can be dropped too.
- [ ] Validate on Qwen3 / Llama 4 DFlash configs.

## Attribution

This is **not original research** — it's an integration of three pieces of
public work. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for full credits to:

- The vLLM project and the authors of PR #34668 and PR #41703
- The AEON-7 / Qwen3.6-NVFP4-DFlash patch set authors
- Google for Gemma 4 and the z-lab team for DFlash

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
