use candle_core::{Result, Tensor};

/// A per-step logit transform for constrained decoding: `mask` restricts which
/// tokens may be sampled next, and `accept` advances internal state with the
/// token that was chosen. Implemented by `parsers::json::JsonConstraint`.
pub trait LogitMask {
    /// Return `logits` with disallowed tokens pushed to `-inf`.
    fn mask(&self, logits: &Tensor) -> Result<Tensor>;

    /// Record the token that was actually sampled.
    fn accept(&mut self, token: u32);
}
