# Hardware notes

## The short version

This project is unusually forgiving of a modest GPU. A 48-token vocabulary and
a 25M-parameter model is a tiny workload by 2026 standards, and the real
bottleneck is not the GPU at all — it is **CPU-side verifier throughput**. The
self-improvement loop of §06 evaluates 64 candidates across 20k specs per
round, which is 1.28M simulations; the model's forward passes are the cheap
part. Budget optimisation effort accordingly: `cargo build --release` and
enough cores matter more than the accelerator.

## AMD Radeon RX 7600 (RDNA 3, `gfx1102`, 8 GB)

RDNA 3 is a considerably better position than RDNA 2 was. The 7600 is
`gfx1102`; ROCm's officially supported desktop list centres on `gfx1100` (the
7900 series), but 6.x supports `gfx1101`/`gfx1102` for the libraries that
matter here, and the override below covers the cases where a kernel is only
shipped for `gfx1100`.

```bash
# Linux only. ROCm on Windows is not viable for training.
# Ubuntu 24.04 or 22.04.

# Present gfx1102 as gfx1100 for any library that only ships gfx1100 kernels.
# Try WITHOUT this first on recent ROCm -- if it works unset, leave it unset.
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Keeps fragmentation down across the long allocations the sampling loop makes.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# verify
python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

| Concern | Reality on this card |
|---|---|
| VRAM | 8 GB. Comfortable for 25M parameters at seq 1568 with batch 16–24. If tight: gradient checkpointing, or crop each grid to its occupied bounding box and predict the box dims in the prefix — that cuts sequence length roughly 4x and removes most of the air. |
| bf16 | RDNA 3 *does* have proper bf16 matrix support, unlike RDNA 2. `torch.autocast` with bf16 is worth trying and should be a real speedup. If anything goes strange, fp32 at this model size is genuinely fine. |
| Flash attention | Do not count on it. Use PyTorch SDPA and let it fall back to the math kernel; at seq 1568 the difference is tolerable. |
| `torch.compile` | Try it, expect breakage on unsupported archs. Keep an eager-mode path as the default in configs — which is what the shipped configs do. |
| Verifier parallelism | **This is the real scaling axis.** The Rust core is already rayon-parallel across cores; give it all of them. 64 candidates x 16 truth-table rows per spec adds up fast. |
| Escape hatch | A single rented A100-hour trains the main model end to end. Keep the code device-agnostic so a stranger with an NVIDIA box can reproduce you — that matters for the repository's reach more than the local speedup does. |

## Measured verifier throughput

On the machine this was developed on (4 cores), `cargo test --release --test
golden` reports **~153 µs per evaluation** for 2–3 input circuits, single
threaded. A 4-input spec runs 16 rows through two passes and costs
proportionally more; all of it is comfortably inside the "well under a
millisecond" budget the design assumes, and it scales linearly with cores
through rayon.

For a full §06 round at 1.28M evaluations, that is roughly

```
1.28e6 x 200 µs / n_cores  ≈  4.3 core-hours
```

so about 30 minutes on 8 cores. That is the number to plan around, not the GPU
time.

## A note on the older RX 6700

The original design targeted an RX 6700 (RDNA 2, `gfx1031`), which is *not* on
AMD's supported list and needs `HSA_OVERRIDE_GFX_VERSION=10.3.0` to present as
`gfx1030`. It works in practice. The two differences that matter versus the
7600: RDNA 2 lacks proper bf16 matrix support, so prefer fp16 with a
`GradScaler` or plain fp32; and it has 10 GB rather than 8, which makes the
larger batch sizes easier.
