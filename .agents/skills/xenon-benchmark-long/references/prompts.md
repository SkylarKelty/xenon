# Long Prompts Reference (> 256 tokens)

Prompts for benchmarking and verifying the Xenon engine with prompt lengths exceeding 256 tokens. These exercise the long-context attention paths, KV cache pressure, and sliding window attention layers.

## Prompt Suite

### 1. Medium Long (300–600 tokens)

| Name | Prompt | Approx Tokens | Use Case |
|------|--------|---------------|----------|
| code-function | A single well-documented Python function (~400 tokens including docstring, type hints, and inline comments). Tests code understanding and docstring generation. | ~350 | Code comprehension |
| story-excerpt | A 5-paragraph story excerpt (~500 tokens). Tests continuation quality and context retention across paragraphs. | ~480 | Creative writing |
| technical-article | First 3 paragraphs of a technical blog post about CUDA programming (~450 tokens). Tests summary or continuation. | ~420 | Technical writing |
| conversation-log | Simulated 4-turn conversation between a user and support agent (~400 tokens). Tests multi-turn context tracking. | ~380 | Conversation |
| bug-analysis | A detailed bug report with stack trace, environment details, and reproduction steps (~500 tokens). Tests root cause analysis. | ~460 | Debugging |
| code-review-full | A 300-line code review comment thread with suggestions and responses (~550 tokens). Tests review synthesis. | ~520 | Code review |
| api-documentation | API endpoint documentation with parameters, examples, and error codes (~450 tokens). Tests doc generation or Q&A. | ~410 | Documentation |
| config-explanation | A detailed YAML/JSON configuration file with inline comments explaining each field (~400 tokens). Tests config validation suggestions. | ~360 | System config |
| meeting-transcript | A 5-minute meeting transcript (~500 tokens). Tests action item extraction and summary. | ~480 | Transcript processing |
| research-summary | Abstract and introduction of a research paper (~550 tokens). Tests simplification or Q&A. | ~510 | Scientific text |
| email-thread | A 4-email thread with replies and forwards (~450 tokens). Tests thread summary or response drafting. | ~420 | Email processing |
| legal-clause | A contract clause with definitions and sub-clauses (~400 tokens). Tests simplification or risk analysis. | ~370 | Legal text |
| medical-case | A simplified patient case description with symptoms, history, and test results (~500 tokens). Tests differential diagnosis reasoning. | ~460 | Medical reasoning |
| product-spec | A product requirement document section with user stories and acceptance criteria (~550 tokens). Tests spec review or gap analysis. | ~510 | Product management |
| tutorial-section | A section from a programming tutorial with code examples and explanations (~450 tokens). Tests continuation or Q&A. | ~410 | Educational |

### 2. Long (700–1500 tokens)

| Name | Prompt | Approx Tokens | Use Case |
|------|--------|---------------|----------|
| multi-file-code | Concatenation of 3 related Python modules with imports and docstrings (~800 tokens). Tests cross-file reasoning. | ~780 | Large code context |
| chapter-excerpt | A full chapter excerpt from a novel (~1200 tokens). Tests long-range narrative consistency. | ~1150 | Literary analysis |
| paper-introduction | Full introduction section of a CS paper with related work (~1000 tokens). Tests summarization or critique. | ~950 | Academic |
| dialogue-script | A 10-minute screenplay scene with stage directions (~900 tokens). Tests character consistency. | ~860 | Script analysis |
| log-analysis | 200 lines of application logs with timestamps and error patterns (~1100 tokens). Tests anomaly detection. | ~1050 | Log processing |
| dataset-description | A detailed dataset card with statistics, provenance, and usage guidelines (~850 tokens). Tests data validation. | ~810 | Data science |
| architecture-doc | A microservice architecture document with diagrams described in text (~950 tokens). Tests design critique. | ~900 | System design |
| policy-document | A security policy with sections on access control, encryption, and incident response (~1200 tokens). Tests compliance check. | ~1150 | Policy review |
| interview-transcript | A 30-minute technical interview transcript (~1300 tokens). Tests candidate evaluation. | ~1250 | HR/evaluation |
| codebase-walkthrough | A guided walkthrough of a repository structure and key files (~1000 tokens). Tests understanding and suggestions. | ~960 | Codebase onboarding |
| multi-step-reasoning | A multi-step mathematical proof or logical argument (~900 tokens). Tests step-by-step verification. | ~860 | Reasoning |
| customer-feedback | 20 customer feedback entries with ratings and comments (~800 tokens). Tests sentiment analysis and theme extraction. | ~760 | Feedback analysis |
| news-article | A full news article with headline, byline, and body (~1100 tokens). Tests fact extraction or summary. | ~1050 | Journalism |
| scientific-method | A methods section from a biology paper (~950 tokens). Tests reproducibility assessment. | ~900 | Scientific method |
| onboarding-guide | A new engineer onboarding guide covering tools, processes, and codebase (~1400 tokens). Tests Q&A and navigation. | ~1350 | Onboarding |

### 3. Very Long (2000–4000 tokens)

| Name | Prompt | Approx Tokens | Use Case |
|------|--------|---------------|----------|
| novella-excerpt | A substantial novella excerpt (~2500 tokens). Tests very long context coherence and theme tracking. | ~2400 | Extended narrative |
| research-survey | A literature survey covering 10 papers with summaries and comparisons (~3000 tokens). Tests synthesis and cross-paper reasoning. | ~2900 | Research synthesis |
| codebase-analysis | An analysis of a medium-sized codebase architecture with file listings and relationships (~2800 tokens). Tests architectural reasoning. | ~2700 | Codebase analysis |
| legal-contract | A full software license agreement (~3500 tokens). Tests clause extraction and risk assessment. | ~3400 | Legal analysis |
| textbook-chapter | A textbook chapter on machine learning fundamentals (~4000 tokens). Tests pedagogical Q&A across the chapter. | ~3800 | Education |
| system-log-marathon | 500 lines of distributed system logs with interleaved services (~3200 tokens). Tests root cause analysis at scale. | ~3100 | Large-scale debugging |
| conversation-marathon | A 20-turn customer support conversation with context switches (~2500 tokens). Tests long conversation memory. | ~2400 | Support analytics |
| documentation-suite | Complete API documentation for a small service with all endpoints (~2800 tokens). Tests comprehensive Q&A. | ~2700 | API Q&A |
| multi-document-qa | 5 related short documents concatenated for cross-document Q&A (~3000 tokens). Tests cross-document reasoning. | ~2900 | Multi-document |
| debate-transcript | A full Oxford-style debate transcript (~2200 tokens). Tests argument tracking and evaluation. | ~2100 | Debate analysis |

## Generating Prompts of Exact Length

To generate a prompt of exactly N tokens:

```bash
python3 << 'PYEOF'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-4-4b-it')

# Method 1: Repeat a simple sentence until target length
base = "The quick brown fox jumps over the lazy dog. "
target = 512
tokens = []
while len(tokens) < target:
    tokens.extend(tok.encode(base, add_special_tokens=False))
prompt = tok.decode(tokens[:target])
print(prompt)
PYEOF
```

Or for realistic prompts:
```bash
python3 << 'PYEOF'
# Fetch a Wikipedia article section of approximate token count
import urllib.request, json, re

def get_wiki_paragraphs(title, min_tokens=300):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('google/gemma-4-4b-it')
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read())
    text = data.get('extract', '')
    # Fetch full content via parse API
    parse_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={title}&prop=text&format=json"
    with urllib.request.urlopen(parse_url) as r:
        parsed = json.loads(r.read())
    html = parsed['parse']['text']['*']
    # Strip HTML tags roughly
    clean = re.sub(r'<[^>]+>', ' ', html)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

# Example usage
# text = get_wiki_paragraphs("Artificial_intelligence")
# tok = AutoTokenizer.from_pretrained('google/gemma-4-4b-it')
# print(f"Token count: {len(tok.encode(text))}")
PYEOF
```

## Benchmarking Patterns

### Medium Long Benchmark (400 tokens)
```bash
MODEL=/home/k1811651/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b

# First create a prompt file with ~400 tokens
cat > /tmp/prompt_400.txt << 'EOF'
Design a REST API for a task management application. The API should support creating tasks with title, description, priority (low/medium/high), due date, and assignee. Tasks can be organized into projects. Each project has a name, description, and owner. Users can comment on tasks and upload attachments. Implement filtering by status, priority, and assignee. Include pagination with cursor-based navigation. Document all endpoints with request/response examples and error codes. Consider rate limiting and authentication using JWT tokens. The API should follow REST conventions and use JSON for all exchanges. Include webhook support for real-time notifications on task changes.
EOF

PROMPT=$(cat /tmp/prompt_400.txt)

cargo run --release --bin xenon-cli -- bench "$MODEL" \
  --prompt "$PROMPT" \
  --max-new 64 \
  --warmup 1 \
  --iters 3 \
  --label long-400tok
```

### Long Benchmark (1000 tokens)
```bash
# Generate a ~1000 token prompt programmatically or use a file
python3 << 'PYEOF'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-4-4b-it')
paragraphs = [
    "Neural networks are computational models inspired by biological neural networks. "
    "They consist of interconnected groups of artificial neurons that process information using a connectionist approach to computation. "
    "In most cases, neural networks are adaptive systems that change their structure based on external or internal information that flows through the network.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. "
    "Learning can be supervised, semi-supervised, or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection, and board game programs.",
    "The adjective deep in deep learning refers to the use of multiple layers in the network. "
    "Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. "
    "Deep learning is a modern variation that is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions.",
]
text = "\n\n".join(paragraphs)
tokens = tok.encode(text)
print(f"Token count: {len(tokens)}")
# If too short, repeat
while len(tokens) < 1000:
    text += "\n\n" + paragraphs[len(tokens) % len(paragraphs)]
    tokens = tok.encode(text)
print(tok.decode(tokens[:1000]))
PYEOF
```

### Very Long Benchmark (2000+ tokens)
```bash
# Use a file-based approach for very long prompts
cargo run --release --bin xenon-cli -- bench "$MODEL" \
  --prompt "$(cat /tmp/prompt_2000.txt)" \
  --max-new 32 \
  --warmup 1 \
  --iters 2 \
  --max-len 4096 \
  --label long-2000tok

# Note: very long prompts may hit KV cache limits;
# monitor with `vram` subcommand first.
```

## KV Cache Considerations

For the Gemma-4 model with `max_len=4096`:
- Each layer consumes KV space proportional to `max_len * head_dim * num_kv_heads * 2 * sizeof(bf16)`
- With KV sharing (18 of 42 layers reuse), ~24 unique KV slots
- At 4096 context length: ~3.5 GiB for KV cache alone
- Prefill at 1000 tokens: ~1 GiB temporarily during attention computation
- Always check VRAM before very long runs:
  ```bash
  cargo run --release --bin xenon-cli -- vram
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
