//! The decoder architecture, composed from `candle_nn` bricks.
//!
//! Ported from chat-mlx's `engine/model.rs` (which composed `mlx-rs` nn modules)
//! to candle. Struct field names mirror HF tensor keys so `VarBuilder` maps
//! official weights with no manual remapping. Config-driven across the
//! Llama / Qwen2 / Qwen3 / MiniCPM families: QKV bias, QK-norm, and tied
//! embeddings are all toggled from `ModelArgs`.
//!
//! Two weight sources share one architecture and one forward:
//!  - **fp** safetensors via `candle_nn::VarBuilder` (HF tensor names),
//!  - **quantized** GGUF via `QMatMul` (llama.cpp tensor names + GGUF metadata).
//!
//! The projections are a `QLinear` enum (dense `Linear` or `QMatMul`) — the
//! candle analogue of chat-mlx's `MaybeQuantized<Linear>`. Only construction
//! differs between the two; `forward` is identical.

use std::io::{Read, Seek};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use candle_core::quantized::{QMatMul, QTensor, gguf_file};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_no_bias, rms_norm,
};

use super::cache::KvCache;
use super::config::ModelArgs;

/// RoPE table length. Bounds the addressable context; the rotating KV cache,
/// not this, is what bounds memory.
const MAX_SEQ: usize = 32_768;

/// Pick the compute device for the enabled backend feature.
pub fn default_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda(0);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Device::new_metal(0);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}

/// Compute dtype per device: bf16 on CUDA, f16 on Metal, f32 on CPU.
pub fn default_dtype(device: &Device) -> DType {
    match device {
        Device::Cuda(_) => DType::BF16,
        Device::Metal(_) => DType::F16,
        _ => DType::F32,
    }
}

/// A linear projection backed by either dense fp weights or a quantized matmul.
enum QLinear {
    Dense(Linear),
    Quant(QMatMul),
}

impl QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            QLinear::Dense(l) => l.forward(x),
            QLinear::Quant(q) => q.forward(x),
        }
    }
}

/// Shared RoPE tables (GPT-NeoX / "rotate-half" style, matching Qwen & Llama).
#[derive(Clone)]
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: usize, theta: f32, dtype: DType, dev: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let t = Tensor::arange(0u32, MAX_SEQ as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ, 1))?;
        let freqs = t.matmul(&inv_freq)?; // (MAX_SEQ, half)
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Rotate q and k at sequence position `offset`. Inputs are
    /// `(batch, heads, seq, head_dim)`.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

/// Expand grouped KV heads to match the query head count (GQA).
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, kv_heads, seq, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv_heads, n_rep, seq, d))?
        .reshape((b, kv_heads * n_rep, seq, d))
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rotary: Rotary,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(args: &ModelArgs, rotary: Rotary, vb: VarBuilder) -> Result<Self> {
        let head_dim = args.head_dim as usize;
        let n_heads = args.n_heads as usize;
        let n_kv = args.n_kv_heads as usize;
        let dim = args.dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;

        let mk = |i: usize, o: usize, bias: bool, vb: VarBuilder| -> Result<QLinear> {
            Ok(QLinear::Dense(if bias {
                linear(i, o, vb)?
            } else {
                linear_no_bias(i, o, vb)?
            }))
        };

        let (q_norm, k_norm) = if args.use_qk_norm {
            (
                Some(rms_norm(head_dim, args.norm_eps as f64, vb.pp("q_norm"))?),
                Some(rms_norm(head_dim, args.norm_eps as f64, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj: mk(dim, q_dim, args.attn_qkv_bias, vb.pp("q_proj"))?,
            k_proj: mk(dim, kv_dim, args.attn_qkv_bias, vb.pp("k_proj"))?,
            v_proj: mk(dim, kv_dim, args.attn_qkv_bias, vb.pp("v_proj"))?,
            o_proj: mk(q_dim, dim, args.attn_o_bias, vb.pp("o_proj"))?,
            q_norm,
            k_norm,
            rotary,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn new_gguf<R: Read + Seek>(
        args: &ModelArgs,
        rotary: Rotary,
        gg: &mut Gguf<R>,
        i: usize,
    ) -> Result<Self> {
        let p = format!("blk.{i}");
        let eps = args.norm_eps as f64;
        let (q_norm, k_norm) = if args.use_qk_norm {
            (
                Some(gg.rms(&format!("{p}.attn_q_norm.weight"), eps)?),
                Some(gg.rms(&format!("{p}.attn_k_norm.weight"), eps)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.attn_q.weight"))?),
            k_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.attn_k.weight"))?),
            v_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.attn_v.weight"))?),
            o_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.attn_output.weight"))?),
            q_norm,
            k_norm,
            rotary,
            n_heads: args.n_heads as usize,
            n_kv_heads: args.n_kv_heads as usize,
            head_dim: args.head_dim as usize,
            scale: (args.head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, cache: &mut KvCache) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let mut q = q
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(&q.contiguous()?)?;
            k = kn.forward(&k.contiguous()?)?;
        }

        let offset = cache.offset();
        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let (k, v) = cache.update_and_fetch(&k, &v)?;

        let n_rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, n_rep)?.contiguous()?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;

        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?; // (b, heads, l, head_dim)

        let out = out
            .transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

impl Mlp {
    fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let dim = args.dim as usize;
        let hidden = args.hidden_dim as usize;
        Ok(Self {
            gate_proj: QLinear::Dense(linear_no_bias(dim, hidden, vb.pp("gate_proj"))?),
            up_proj: QLinear::Dense(linear_no_bias(dim, hidden, vb.pp("up_proj"))?),
            down_proj: QLinear::Dense(linear_no_bias(hidden, dim, vb.pp("down_proj"))?),
        })
    }

    fn new_gguf<R: Read + Seek>(gg: &mut Gguf<R>, i: usize) -> Result<Self> {
        let p = format!("blk.{i}");
        Ok(Self {
            gate_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.ffn_gate.weight"))?),
            up_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.ffn_up.weight"))?),
            down_proj: QLinear::Quant(gg.qmatmul(&format!("{p}.ffn_down.weight"))?),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(args: &ModelArgs, rotary: Rotary, vb: VarBuilder) -> Result<Self> {
        let dim = args.dim as usize;
        let eps = args.norm_eps as f64;
        Ok(Self {
            self_attn: Attention::new(args, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(args, vb.pp("mlp"))?,
            input_layernorm: rms_norm(dim, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(dim, eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn new_gguf<R: Read + Seek>(
        args: &ModelArgs,
        rotary: Rotary,
        gg: &mut Gguf<R>,
        i: usize,
    ) -> Result<Self> {
        let eps = args.norm_eps as f64;
        let p = format!("blk.{i}");
        Ok(Self {
            input_layernorm: gg.rms(&format!("{p}.attn_norm.weight"), eps)?,
            self_attn: Attention::new_gguf(args, rotary, gg, i)?,
            post_attention_layernorm: gg.rms(&format!("{p}.ffn_norm.weight"), eps)?,
            mlp: Mlp::new_gguf(gg, i)?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, cache: &mut KvCache) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed, mask, cache)?;
        let h = (x + attn)?;
        let ff = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h + ff
    }
}

/// The full causal LM: embedding + decoder stack + final norm + lm head.
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QLinear,
    device: Device,
    dtype: DType,
    n_layers: usize,
}

impl Model {
    /// Build from fp safetensors (HF tensor names) via a `VarBuilder`.
    pub fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let dim = args.dim as usize;
        let vocab = args.vocab_size as usize;

        let rotary = Rotary::new(args.head_dim as usize, args.rope_theta, dtype, &device)?;

        let vm = vb.pp("model");
        let embed_tokens = embedding(vocab, dim, vm.pp("embed_tokens"))?;
        let layers = (0..args.n_layers as usize)
            .map(|i| DecoderLayer::new(args, rotary.clone(), vm.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let norm = rms_norm(dim, args.norm_eps as f64, vm.pp("norm"))?;

        // Tied embeddings: Qwen-style repos ship no `lm_head.weight`; reuse the
        // input embedding matrix (vocab, dim) as the output projection.
        let lm_head = if args.tie_word_embeddings || !vb.contains_tensor("lm_head.weight") {
            QLinear::Dense(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            QLinear::Dense(linear_no_bias(dim, vocab, vb.pp("lm_head"))?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
            dtype,
            n_layers: args.n_layers as usize,
        })
    }

    /// Build from a quantized GGUF file (llama.cpp tensor names + metadata).
    /// Returns the model, the `ModelArgs` reconstructed from GGUF metadata, and
    /// the architecture string. Quantized matmuls run f32 activations.
    ///
    /// Architecture-agnostic: metadata keys are namespaced under
    /// `general.architecture` (`llama.*` / `qwen2.*` / `qwen3.*` / `mistral.*`
    /// …), and the tensor names are llama.cpp's shared scheme — so the same
    /// path loads any Llama/Qwen/Mistral-family GGUF, not just Qwen3.
    pub fn from_gguf_reader<R: Read + Seek>(
        reader: R,
        device: &Device,
    ) -> Result<(Self, ModelArgs, String)> {
        let mut gg = Gguf::from_reader(reader, device)?;
        let dtype = DType::F32;

        let arch = gg.arch()?;
        let k = |s: &str| format!("{arch}.{s}");

        let n_heads = gg.u32(&k("attention.head_count"))? as usize;
        let n_kv = gg.u32_or(&k("attention.head_count_kv"), n_heads as u32) as usize;
        let n_layers = gg.u32(&k("block_count"))? as usize;
        let dim = gg.u32(&k("embedding_length"))? as usize;
        let hidden = gg.u32(&k("feed_forward_length"))? as usize;
        // key_length (head_dim) is explicit for Qwen3; Llama/Qwen2 omit it ->
        // derive from dim/heads. eps and rope base also carry family defaults.
        let head_dim = gg.u32_or(&k("attention.key_length"), (dim / n_heads) as u32) as usize;
        let eps = gg.f32_or(&k("attention.layer_norm_rms_epsilon"), 1e-5);
        let theta = gg.f32_or(&k("rope.freq_base"), 10_000.0);
        let use_qk_norm = gg.has("blk.0.attn_q_norm.weight");
        let tied = !gg.has("output.weight");

        // token_embd doubles as the (tied) lm head: keep the quantized tensor for
        // the matmul, dequantize a copy for the dense embedding lookup.
        let qt_embed = gg.read_qtensor("token_embd.weight")?;
        let embed_w = qt_embed.dequantize(device)?;
        let vocab = embed_w.dim(0)?;
        let embed_tokens = Embedding::new(embed_w, dim);

        let args = ModelArgs {
            dim: dim as i32,
            n_layers: n_layers as i32,
            n_heads: n_heads as i32,
            n_kv_heads: n_kv as i32,
            head_dim: head_dim as i32,
            hidden_dim: hidden as i32,
            vocab_size: vocab as i32,
            norm_eps: eps,
            rope_theta: theta,
            tie_word_embeddings: tied,
            use_qk_norm,
            attn_qkv_bias: false,
            attn_o_bias: false,
        };

        let rotary = Rotary::new(head_dim, theta, dtype, device)?;
        let layers = (0..n_layers)
            .map(|i| DecoderLayer::new_gguf(&args, rotary.clone(), &mut gg, i))
            .collect::<Result<Vec<_>>>()?;
        let norm = gg.rms("output_norm.weight", eps as f64)?;
        let lm_head = if tied {
            QLinear::Quant(QMatMul::from_qtensor(qt_embed)?)
        } else {
            QLinear::Quant(gg.qmatmul("output.weight")?)
        };

        Ok((
            Self {
                embed_tokens,
                layers,
                norm,
                lm_head,
                device: device.clone(),
                dtype,
                n_layers,
            },
            args,
            arch,
        ))
    }

    /// Native convenience: build from a GGUF file path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_gguf(path: &Path, device: &Device) -> Result<(Self, ModelArgs, String)> {
        Self::from_gguf_reader(std::fs::File::open(path)?, device)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn make_cache(&self, max_size: Option<i32>, keep: i32) -> Vec<KvCache> {
        (0..self.n_layers)
            .map(|_| KvCache::new(max_size, keep))
            .collect()
    }

    /// Forward `tokens` (shape `(batch, seq)`, dtype u32) through the stack,
    /// returning logits `(batch, seq, vocab)` in f32.
    pub fn forward(&self, tokens: &Tensor, cache: &mut [KvCache]) -> Result<Tensor> {
        let (_b, l) = tokens.dims2()?;
        let mut h = self.embed_tokens.forward(tokens)?;

        let offset = cache.first().map_or(0, |c| c.offset());
        let mask = if l > 1 {
            Some(causal_mask(l, offset, self.dtype, &self.device)?)
        } else {
            None
        };

        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), c)?;
        }

        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)?.to_dtype(DType::F32)
    }
}

/// Additive causal mask of shape `(seq, offset + seq)`: query `i` may attend to
/// key `j` iff `j <= i + offset`; disallowed positions are `-inf`.
fn causal_mask(seq: usize, offset: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let klen = offset + seq;
    let mut data = vec![0f32; seq * klen];
    for i in 0..seq {
        for j in 0..klen {
            if j > i + offset {
                data[i * klen + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (seq, klen), dev)?.to_dtype(dtype)
}

/// Reader over a GGUF source (a file on native, in-memory bytes on wasm):
/// pulls quantized tensors and metadata by name.
struct Gguf<R: Read + Seek> {
    content: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn from_reader(mut reader: R, device: &Device) -> Result<Self> {
        let content = gguf_file::Content::read(&mut reader)?;
        Ok(Self {
            content,
            reader,
            device: device.clone(),
        })
    }

    fn read_qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.content.tensor(&mut self.reader, name, &self.device)
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let qt = self.read_qtensor(name)?;
        QMatMul::from_qtensor(qt)
    }

    /// Dequantize a tensor (used for norms, whose weights are fp in GGUF).
    fn dense(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.read_qtensor(name)?;
        qt.dequantize(&self.device)
    }

    fn rms(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        Ok(RmsNorm::new(self.dense(name)?, eps))
    }

    fn has(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    /// The model architecture (`general.architecture`), used as the metadata
    /// key namespace and the reported `model_type`.
    fn arch(&self) -> Result<String> {
        match self.content.metadata.get("general.architecture") {
            Some(v) => Ok(v.to_string()?.to_string()),
            None => candle_core::bail!("gguf metadata missing general.architecture"),
        }
    }

    fn u32(&self, key: &str) -> Result<u32> {
        match self.content.metadata.get(key) {
            Some(v) => v.to_u32(),
            None => candle_core::bail!("gguf metadata missing key: {key}"),
        }
    }

    fn u32_or(&self, key: &str, default: u32) -> u32 {
        self.content
            .metadata
            .get(key)
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(default)
    }

    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.content
            .metadata
            .get(key)
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(default)
    }
}
