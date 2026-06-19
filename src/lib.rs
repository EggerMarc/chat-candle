#![allow(clippy::result_large_err)]

pub mod api;
pub mod engine;
pub mod loader;
pub mod parsers;

mod builder;
mod client;

pub use builder::{CandleBuilder, WithModel, WithoutModel};
pub use client::{CandleClient, StructuredMode};
pub use loader::Quantize;
