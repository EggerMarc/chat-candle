#![allow(clippy::result_large_err)]

//! Local-inference chat-rs provider for Qwen3 / Llama / MiniCPM-family models
//! via candle, on CPU / Metal / CUDA. The browser/WebGPU path lives in the
//! sibling `chat-wgpu` crate.

pub mod api;
pub mod engine;
pub mod loader;
pub mod parsers;

mod builder;
mod client;

pub use builder::{CandleBuilder, WithModel, WithoutModel};
pub use client::{CandleClient, StructuredMode};
pub use loader::Quantize;
