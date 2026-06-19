//! Prefill + decode loop on candle.
//!
//! Mirrors chat-mlx's `generate` / `generate_constrained` surface (same
//! arguments, same `GenStats`, same `on_token` callback contract) so the API
//! layer is unchanged. Where mlx kept sampled tokens on-device and batched
//! `tokens_per_eval` evals to amortise host syncs, candle decodes one token per
//! forward; `tokens_per_eval` is accepted for signature parity but unused.
//!
//! The n-gram / prompt-lookup speculative path (`generate_ngram`) is not yet
//! ported — it needs cache truncation + multi-token verification; a follow-up.

use anyhow::Result;
use candle_core::{IndexOp, Tensor};

// `std::time::Instant` panics on wasm32-unknown-unknown (no clock). Time on
// native; report 0.0 in the browser (the JS side can measure wall-clock).
#[cfg(not(target_arch = "wasm32"))]
fn now() -> std::time::Instant {
    std::time::Instant::now()
}
#[cfg(not(target_arch = "wasm32"))]
fn secs(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64()
}
#[cfg(target_arch = "wasm32")]
fn now() {}
#[cfg(target_arch = "wasm32")]
fn secs(_: ()) -> f64 {
    0.0
}

use super::cache::KvCache;
use super::constraint::LogitMask;
use super::model::Model;
use super::sampler::SampleOpts;

pub struct GenStats {
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn generate<F: FnMut(u32) -> bool>(
    model: &Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SampleOpts,
    eos: &[u32],
    _tokens_per_eval: usize,
    cache: &mut [KvCache],
    mut on_token: F,
) -> Result<GenStats> {
    let dev = model.device();
    let mut lp = opts.processor(0);

    let t_prefill = now();
    let input = Tensor::new(prompt_ids, dev)?.unsqueeze(0)?;
    let logits = model.forward(&input, cache)?;
    let last = logits.i((0, logits.dim(1)? - 1))?;
    let mut next = lp.sample(&last)?;
    let prefill_secs = secs(t_prefill);

    let t_decode = now();
    let mut out = Vec::with_capacity(max_tokens);
    loop {
        if eos.contains(&next) {
            break;
        }
        out.push(next);
        if !on_token(next) || out.len() >= max_tokens {
            break;
        }
        let input = Tensor::new(&[next], dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, cache)?;
        let last = logits.i((0, 0))?;
        next = lp.sample(&last)?;
    }
    let decode_secs = secs(t_decode);

    Ok(GenStats {
        tokens: out,
        prefill_secs,
        decode_secs,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn generate_constrained<F: FnMut(u32) -> bool>(
    model: &Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SampleOpts,
    eos: &[u32],
    cache: &mut [KvCache],
    constraint: &mut dyn LogitMask,
    mut on_token: F,
) -> Result<GenStats> {
    let dev = model.device();
    let mut lp = opts.processor(0);

    let t_prefill = now();
    let input = Tensor::new(prompt_ids, dev)?.unsqueeze(0)?;
    let logits = model.forward(&input, cache)?;
    let last = constraint.mask(&logits.i((0, logits.dim(1)? - 1))?)?;
    let mut next = lp.sample(&last)?;
    let prefill_secs = secs(t_prefill);

    let t_decode = now();
    let mut out = Vec::with_capacity(max_tokens);
    loop {
        if eos.contains(&next) {
            break;
        }
        out.push(next);
        constraint.accept(next);
        if !on_token(next) || out.len() >= max_tokens {
            break;
        }
        let input = Tensor::new(&[next], dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, cache)?;
        let last = constraint.mask(&logits.i((0, 0))?)?;
        next = lp.sample(&last)?;
    }
    let decode_secs = secs(t_decode);

    Ok(GenStats {
        tokens: out,
        prefill_secs,
        decode_secs,
    })
}
