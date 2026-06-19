//! Sampling config and its mapping onto candle's `LogitsProcessor`.
//!
//! chat-mlx sampled on-device with hand-written mlx ops; candle ships a
//! `LogitsProcessor` (temperature / top-k / top-p, seeded) that we drive
//! instead. `SampleOpts` stays the carrier type the API layer lowers requests
//! into, so `api/types/request.rs` is unchanged.

use candle_transformers::generation::{LogitsProcessor, Sampling};

#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temp: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl SampleOpts {
    /// Build a fresh, seeded `LogitsProcessor` for one generation. Held across
    /// the decode loop so its RNG advances per token.
    pub fn processor(&self, seed: u64) -> LogitsProcessor {
        let sampling = if self.temp <= 0.0 {
            Sampling::ArgMax
        } else {
            let temperature = self.temp as f64;
            match (self.top_k, self.top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p: p as f64,
                    temperature,
                },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP {
                    p: p as f64,
                    temperature,
                },
                (None, None) => Sampling::All { temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    }
}
