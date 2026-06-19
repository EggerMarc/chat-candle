# chat-candle

Local-inference **[chat-rs](https://github.com/eggermarc/chat-rs) provider** (and CLI)
for Qwen3 / Qwen2.5 / Llama / MiniCPM-family models, via
[candle](https://github.com/huggingface/candle). It implements
`CompletionProvider` + `StreamProvider`, so it drops into `chat_core::ChatBuilder`
and participates in the tool-calling, structured-output, and streaming chat loop —
the same surface as `chat-claude`, `chat-openai`, etc.

It owns the raw token loop (no daemon, no HTTP): tokenization, chat-templating,
sampling, KV cache, tool-call parsing, and JSON-constrained decoding all happen
in-process. It is the cross-platform sibling of
[`chat-mlx`](https://github.com/eggermarc/chat-mlx): same engine architecture and
provider wiring, but the kernel layer is candle (CPU / Metal / CUDA) instead of
MLX (Apple-only).

## Backends

candle owns the GPU/CPU kernels; the engine composes them. Pick a backend at
build time:

```bash
cargo run --release                      # CPU
cargo run --release --features metal     # Apple GPU
cargo run --release --features cuda      # NVIDIA GPU
cargo run --release --features accelerate # CPU + Apple Accelerate BLAS
```

Compute dtype follows the device: bf16 on CUDA, f16 on Metal, f32 on CPU.

## Layout

```
src/engine/         the inference core (no chat-rs types)
  config.rs         parse HF config.json -> ModelArgs
  model.rs          the architecture: Attention/Mlp/Decoder composed from candle_nn modules
  cache.rs          KV cache (seq-axis concat; candle tensors are immutable)
  sampler.rs        SampleOpts -> candle LogitsProcessor (greedy / temp / top-k / top-p)
  generate.rs       prefill + decode loop; plus generate_constrained (logit-masked)
  constraint.rs     LogitMask trait (constrained decoding hook)
  template.rs       ChatML / model jinja-template rendering
src/loader.rs       HF download + VarBuilder weight load (config-driven arch detection)
src/builder.rs      CandleBuilder (type-state) -> CandleClient
src/client.rs       CandleClient (Arc<Mutex<Model>>, Clone)
src/api/            CompletionProvider / StreamProvider impls + request/response mapping
src/parsers/        reasoning (<think>), tool (families + stripper), json (validator+mask), structured
src/main.rs         CLI over the lib
```

candle ships the nn bricks (`Linear`, `RmsNorm`, `Embedding`, `rotary_emb::rope`,
`ops::softmax_last_dim`); we compose the architecture. Struct field names in
`model.rs` mirror HF tensor keys so `VarBuilder` maps official weights with no
manual remapping.

## Use as a provider

```rust
use chat_core::builder::ChatBuilder;
use chat_core::types::messages::{Messages, content};
use chat_core::parts;
use chat_candle::CandleBuilder;

let client = CandleBuilder::new().with_model("Qwen/Qwen3-0.6B").build()?;
let mut chat = ChatBuilder::new().with_model(client).build();

let mut msgs = Messages::default();
msgs.push(content::from_user(parts!["Explain RoPE in one sentence."]));
let out = chat.complete(&mut msgs).await?;
```

Builder knobs: `with_max_context`, `with_sink_tokens`, `with_tokens_per_eval`,
`with_tool_format` / `with_tool_pattern`, `with_structured_mode`.

## CLI

```bash
cargo run --release --features metal -- --model Qwen/Qwen3-0.6B \
  --prompt "Explain RoPE in one sentence." --temp 0.7 --top-k 40
# flags: --model --system --prompt --max-tokens --temp --top-k --top-p
#        --max-context --sink-tokens --gguf --gguf-file
```

Default model: `Qwen/Qwen3-0.6B` (bf16/f16 safetensors).

Quantized (GGUF) — weights from `--gguf`, tokenizer/template from `--model`:

```bash
cargo run --release --features metal -- --model Qwen/Qwen3-0.6B \
  --gguf Qwen/Qwen3-0.6B-GGUF --gguf-file Qwen3-0.6B-Q8_0.gguf \
  --prompt "Explain RoPE in one sentence."
```

As a provider: `CandleBuilder::new().with_gguf("Qwen/Qwen3-0.6B-GGUF",
"Qwen3-0.6B-Q8_0.gguf").with_model("Qwen/Qwen3-0.6B").build()?`.

## Supported model families

Architecture is config-driven (`config.json`) with tensor-name probing, no
per-family source files:

- **Llama / MiniCPM** — GQA, SwiGLU, RoPE; bias per the `attention_bias` flag.
- **Qwen2 / Qwen2.5** — QKV bias (auto-detected from `q_proj.bias`) + tied
  embeddings (no shipped `lm_head.weight`).
- **Qwen3** — per-head QK-Norm (auto-detected from `q_norm.weight`), no QKV bias.

## Structured output

`ChatBuilder::with_structured_output::<T>()` works two ways, selected by
`CandleBuilder::with_structured_mode`:

- **`StructuredMode::Prompt`** (default) — inject the schema, parse the emitted
  JSON; the chat loop retries on a parse miss.
- **`StructuredMode::Constrained`** — mask logits each decode step so only tokens
  keeping the output a valid-JSON prefix can be sampled. Enforces JSON *syntax*;
  the schema's types/required fields are validated on the typed deserialize.

## Status

Throughput, Qwen3-0.6B, M-series, 48-token decode (Metal): fp16 **20.8 tok/s**;
Q8_0 GGUF **34.3 tok/s**; Q4_K_M GGUF **38.4 tok/s**.

- [x] bf16/f16 generate (Qwen3-0.6B), coherent output on CPU / Metal
- [x] config-driven QKV bias + QK-norm + tied embeddings (auto-detected)
- [x] **chat-rs provider**: completion, streaming, tool families, structured output
- [x] constrained (valid-JSON) decoding via logit masking
- [x] **quantization** — pre-quantized GGUF via `QMatMul` (`QLinear` enum: dense
      `Linear` | `QMatMul`). Architecture-agnostic: hyperparameters + arch read
      from GGUF metadata (`general.architecture` namespace), so any
      Llama/Qwen/Mistral-family GGUF loads, not just Qwen3.
- [x] q4_K / q8_0 GGUF verified on Metal (Q4_K_M, Q8_0)
- [ ] **rotating attention-sink KV cache** — today grows by concat (unbounded);
      port chat-mlx's bounded window.
- [ ] n-gram / prompt-lookup speculative decoding (`generate_ngram`)
- [ ] wasm32 + WebGPU target (candle compiles to wasm — the original motivation)
