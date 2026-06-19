#![allow(clippy::result_large_err)]

// The engine is backend-agnostic and compiles everywhere, including wasm32.
// The parsers (tool/json/reasoning/structured) are only consumed by the
// provider layer, so they ride along with it. The chat-rs provider stack and
// the HF loader are native-only.
pub mod engine;

#[cfg(feature = "provider")]
pub mod parsers;

#[cfg(feature = "loader-hf")]
pub mod loader;

#[cfg(feature = "provider")]
pub mod api;
#[cfg(feature = "provider")]
mod builder;
#[cfg(feature = "provider")]
mod client;

#[cfg(feature = "provider")]
pub use builder::{CandleBuilder, WithModel, WithoutModel};
#[cfg(feature = "provider")]
pub use client::{CandleClient, StructuredMode};
#[cfg(feature = "loader-hf")]
pub use loader::Quantize;

// Browser entry point: a wasm-bindgen API over the engine.
#[cfg(target_arch = "wasm32")]
pub mod wasm;
