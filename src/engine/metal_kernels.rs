//! Custom fused Metal kernel: single-query (decode) attention.
//!
//! This is the test of "write our own kernel where candle's is weak". For the
//! `seq == 1` decode step, candle's eager path dispatches ~12 Metal kernels per
//! layer (repeat_kv ×2, q/k contiguous, transpose, qk^T matmul, scale, softmax,
//! pv matmul, …). Here we fuse the whole attention core — scale·QKᵀ → softmax →
//! ·V, with GQA handled inline — into a single dispatch. f32 activations only;
//! the quantized weight matmuls stay on candle's QMatMul.
//!
//! The kernel is intentionally simple (one thread per (batch, head), looping the
//! key dimension) — enough to test whether collapsing dispatches beats candle's
//! many small kernels.
//!
//! RESULT (Qwen3-0.6B Q4_K_M, M-series): numerically correct but ~2.5× SLOWER
//! than candle's path (14.6 vs 37 tok/s @48; 5.6 vs 29 @200, worsening with
//! length). Collapsing dispatches doesn't help when the fused kernel doesn't
//! saturate the GPU: one thread per (b,head) is ~16 threads total, while
//! candle's library matmul/softmax kernels are fully parallel. Beating candle
//! needs a flash-attention kernel: one threadgroup per (b,head), threads
//! splitting the key axis, threadgroup-memory reduction + online softmax.
//! Left here as a working CustomOp3 scaffold; not wired into `model::attend`.

use std::cell::RefCell;

use candle_core::backend::BackendStorage;
use candle_core::{CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_attn_decode_f32(
    device const float* q      [[buffer(0)]],   // [B, H, 1, D]
    device const float* k      [[buffer(1)]],   // [B, Hkv, S, D]
    device const float* v      [[buffer(2)]],   // [B, Hkv, S, D]
    device float*       out    [[buffer(3)]],   // [B, H, 1, D]
    constant uint&      B      [[buffer(4)]],
    constant uint&      H      [[buffer(5)]],
    constant uint&      Hkv    [[buffer(6)]],
    constant uint&      S      [[buffer(7)]],
    constant uint&      D      [[buffer(8)]],
    constant float&     scale  [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * H) return;
    uint b = gid / H;
    uint h = gid % H;
    uint hk = h / (H / Hkv);          // GQA: map query head -> kv head

    device const float* qp    = q + (uint)(b * H + h) * D;
    device const float* kbase = k + (uint)(b * Hkv + hk) * S * D;
    device const float* vbase = v + (uint)(b * Hkv + hk) * S * D;

    // pass 1: max score for numerical stability
    float m = -INFINITY;
    for (uint j = 0; j < S; j++) {
        device const float* kp = kbase + j * D;
        float dot = 0.0f;
        for (uint t = 0; t < D; t++) dot += qp[t] * kp[t];
        dot *= scale;
        m = max(m, dot);
    }

    // pass 2: softmax-weighted sum of V (D <= 128 for the target models)
    float acc[128];
    for (uint t = 0; t < D; t++) acc[t] = 0.0f;
    float denom = 0.0f;
    for (uint j = 0; j < S; j++) {
        device const float* kp = kbase + j * D;
        float dot = 0.0f;
        for (uint t = 0; t < D; t++) dot += qp[t] * kp[t];
        float w = exp(dot * scale - m);
        denom += w;
        device const float* vp = vbase + j * D;
        for (uint t = 0; t < D; t++) acc[t] += w * vp[t];
    }

    device float* op = out + (uint)(b * H + h) * D;
    float inv = 1.0f / denom;
    for (uint t = 0; t < D; t++) op[t] = acc[t] * inv;
}
"#;

thread_local! {
    // Compiled once per thread (the decode loop runs on one thread); metal
    // objects are !Send, so a thread-local cache avoids Send/Sync bounds.
    static PIPELINE: RefCell<Option<ComputePipeline>> = const { RefCell::new(None) };
}

struct FusedAttnDecode {
    scale: f32,
}

impl CustomOp3 for FusedAttnDecode {
    fn name(&self) -> &'static str {
        "fused-attn-decode"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("fused-attn-decode is metal-only")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        lq: &Layout,
        k: &MetalStorage,
        lk: &Layout,
        v: &MetalStorage,
        lv: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if q.dtype() != DType::F32 || k.dtype() != DType::F32 || v.dtype() != DType::F32 {
            candle_core::bail!("fused-attn-decode requires f32");
        }
        // q: (B, H, 1, D); k/v: (B, Hkv, S, D)
        let (bsz, h, _one, d) = lq.shape().dims4()?;
        let (_b2, hkv, s, _d2) = lk.shape().dims4()?;

        let device = q.device();
        let out_count = bsz * h * d;
        let output = device.new_buffer(out_count, DType::F32, "fused-attn-decode")?;

        PIPELINE.with(|cell| -> Result<()> {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let mdev = device.metal_device();
                let lib = mdev
                    .new_library_with_source(SRC, None)
                    .map_err(candle_core::Error::wrap)?;
                let func = lib
                    .get_function("fused_attn_decode_f32", None)
                    .map_err(candle_core::Error::wrap)?;
                let pipe = mdev
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(candle_core::Error::wrap)?;
                *slot = Some(pipe);
            }
            let pipeline = slot.as_ref().unwrap();

            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(pipeline);

            let f32_sz = DType::F32.size_in_bytes();
            encoder.set_buffer(0, Some(q.buffer()), lq.start_offset() * f32_sz);
            encoder.set_buffer(1, Some(k.buffer()), lk.start_offset() * f32_sz);
            encoder.set_buffer(2, Some(v.buffer()), lv.start_offset() * f32_sz);
            encoder.set_buffer(3, Some(&output), 0);

            encoder.set_bytes(4, &(bsz as u32));
            encoder.set_bytes(5, &(h as u32));
            encoder.set_bytes(6, &(hkv as u32));
            encoder.set_bytes(7, &(s as u32));
            encoder.set_bytes(8, &(d as u32));
            encoder.set_bytes(9, &self.scale);

            let total = bsz * h;
            let tg = total.clamp(1, 64);
            encoder.dispatch_threads(
                MTLSize {
                    width: total,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        })?;

        let shape = Shape::from_dims(&[bsz, h, 1, d]);
        let storage = MetalStorage::new(output, device.clone(), out_count, DType::F32);
        Ok((storage, shape))
    }
}

/// Fused single-query attention. `q`: (B, H, 1, D); `k`/`v`: (B, Hkv, S, D),
/// already gathered from the KV cache. Returns (B, H, 1, D). The single query
/// attends to all `S` cached keys (no causal mask needed for decode).
pub fn fused_attn_decode(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    q.contiguous()?.apply_op3_no_bwd(
        &k.contiguous()?,
        &v.contiguous()?,
        &FusedAttnDecode { scale },
    )
}
