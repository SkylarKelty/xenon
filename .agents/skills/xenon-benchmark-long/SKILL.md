---
name: xenon-benchmark-long
description: Benchmark the Xenon LLM inference engine with long prompts (>256 tokens). Use when asked to benchmark, profile, or measure performance on prompts exceeding 256 tokens. Covers long-context attention paths, KV cache pressure, sliding window attention, and throughput at 300–4000 token prompts.
---

# Xenon Benchmark — Long Prompts (>256 tokens)

Benchmark the engine with prompts > 256 tokens. This exercises the long-context attention paths (`attn_flash_tc` for prefill, `attn_split_kv` for decode), KV cache pressure, and sliding window attention layers.

## Prerequisites

- CUDA device available and `xenon-cli` built (`cargo build --release --bin xenon-cli`)
- Model path configured (default: `~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b`)
- Sufficient VRAM for KV cache at target context length

## Workflow

1. **Check VRAM**: `cargo run --release --bin xenon-cli -- vram`
2. **Pick a prompt length** from the reference catalog (see `references/prompts.md`)
3. **Run benchmark**: `cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "..." --max-new 64 --warmup 1 --iters 3 --label long-<N>tok`
4. **Check output**: prefill ms, decode ms/step, tok/s
5. **Compare results**: Look at `results/bench-long-<N>tok-*.json`

## Key Parameters

| Flag | Default | Long Prompt Guidance |
|------|---------|---------------------|
| `--max-new` | 32 | Use 64 for long prompts to get meaningful decode samples |
| `--warmup` | 1 | Always do at least 1 warmup; 2 for stable clocks |
| `--iters` | 3 | 3 minimum; use 5 for statistical confidence |
| `--max-len` | 4096 | Must be ≥ prompt_len + max_new. Raise to 8192 if needed |
| `--chat` | off | Enable for instruction-tuned models |

## VRAM Planning

For Gemma-4-E4B with `max_len=4096`:
- KV cache: ~3.5 GiB at full context
- Prefill at 1000 tokens: ~1 GiB temporary during attention
- Always run `vram` before very long runs:
  ```bash
  cargo run --release --bin xenon-cli -- vram
  ```

## Quick Commands

```bash
MODEL=/home/k1811651/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b

# 400 tokens — medium long
P=$(cat <<'EOF'
Design a REST API for a task management application. The API should support creating tasks with title, description, priority (low/medium/high), due date, and assignee. Tasks can be organized into projects. Each project has a name, description, and owner. Users can comment on tasks and upload attachments. Implement filtering by status, priority, and assignee. Include pagination with cursor-based navigation. Document all endpoints with request/response examples and error codes. Consider rate limiting and authentication using JWT tokens. The API should follow REST conventions and use JSON for all exchanges. Include webhook support for real-time notifications on task changes.
EOF
)
cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "$P" --max-new 64 --label long-400tok

# 1000 tokens — long prefill
P=$(python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-4-4b-it')
paragraphs = [
    'Neural networks are computational models inspired by biological neural networks. '
    'They consist of interconnected groups of artificial neurons that process information using a connectionist approach to computation. '
    'In most cases, neural networks are adaptive systems that change their structure based on external or internal information that flows through the network.',
    'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. '
    'Learning can be supervised, semi-supervised, or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection, and board game programs.',
    'The adjective deep in deep learning refers to the use of multiple layers in the network. '
    'Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. '
    'Deep learning is a modern variation that is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions.',
]
text = '\n\n'.join(paragraphs)
tokens = tok.encode(text)
while len(tokens) < 1000:
    text += '\n\n' + paragraphs[len(tokens) % len(paragraphs)]
    tokens = tok.encode(text)
print(tok.decode(tokens[:1000]))
")
cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "$P" --max-new 32 --label long-1000tok

# 2000+ tokens — very long (use file-based approach)
cargo run --release --bin xenon-cli -- bench "$MODEL" \
  --prompt "$(cat /tmp/prompt_2000.txt)" \
  --max-new 32 \
  --warmup 1 \
  --iters 2 \
  --max-len 4096 \
  --label long-2000tok
```

## Attention Dispatch at Long Context

The engine uses three attention dispatch strategies:

1. **Prefill (T_q >= 16)**: `attn_flash_tc` — tensor-core flash attention
   - Best for long context prefill
   - Uses `cp.async` for K/V prefetching
   - O(BR×D) shared memory, dynamic up to 98K

2. **Decode (T_q = 1, small T_kv)**: `attn_split_kv` — splits KV across SMs
   - May shift to `attn_naive` as T_kv grows very large
   - Chunk size auto-tuned based on SM count

3. **Fallback**: `attn_naive` — full KV in shared memory (48K limit)
   - Rarely triggered at long context; used mainly for edge cases

At very long context (>2000 tokens), the bottleneck shifts from compute to memory bandwidth for KV cache reads.

## Diagnosing Issues

- **High prefill at >1000 tokens**: Expected — flash attention is still O(N²). Check if `attn_flash_tc` is dispatching.
- **Decode latency growing with context**: Normal for naive attention. Verify `attn_split_kv` is active.
- **OOM during prefill**: Reduce `--max-len` or use shorter prompt. Check if temporary attention buffers are oversized.
- **Incorrect outputs at long context**: May indicate KV cache corruption or sliding window misconfiguration.

## Prompt Catalog

See `references/prompts.md` for ready-made prompts at 300–600, 700–1500, and 2000–4000 tokens.
