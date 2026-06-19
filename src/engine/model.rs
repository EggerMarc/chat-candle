//! The decoder architecture, composed from `candle_nn` bricks.
//!
//! Ported from chat-mlx's `engine/model.rs` (which composed `mlx-rs` nn modules)
//! to candle. Struct field names mirror HF tensor keys so `VarBuilder` maps
//! official weights with no manual remapping. Config-driven across the
//! Llama / Qwen2 / Qwen3 / MiniCPM families: QKV bias, QK-norm, and tied
//! embeddings are all toggled from `ModelArgs`.

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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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

        let mk = |i: usize, o: usize, bias: bool, vb: VarBuilder| -> Result<Linear> {
            if bias {
                linear(i, o, vb)
            } else {
                linear_no_bias(i, o, vb)
            }
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
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(args: &ModelArgs, vb: VarBuilder) -> Result<Self> {
        let dim = args.dim as usize;
        let hidden = args.hidden_dim as usize;
        Ok(Self {
            gate_proj: linear_no_bias(dim, hidden, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(dim, hidden, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(hidden, dim, vb.pp("down_proj"))?,
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
    lm_head: Linear,
    device: Device,
    dtype: DType,
    n_layers: usize,
}

impl Model {
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
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(dim, vocab, vb.pp("lm_head"))?
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
