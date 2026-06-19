use candle_core::{Result, Tensor};

pub trait LogitMask {
    fn mask(&self, logits: &Tensor) -> Result<Tensor>;

    fn accept(&mut self, token: u32);
}
