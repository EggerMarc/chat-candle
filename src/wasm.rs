//! Browser entry point: a `wasm-bindgen` API over the engine.
//!
//! The browser has no filesystem or HF hub, so weights and tokenizer arrive as
//! byte arrays from JS (fetched + cached on the JS side, e.g. in IndexedDB).
//! Inference runs on candle's CPU backend (wasm32 has no WebGPU backend in
//! candle 0.10.2 yet — that's the next milestone). Tokens stream out through a
//! JS callback.

use std::io::Cursor;

use candle_core::Device;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

use crate::engine::{generate, model::Model, sampler::SampleOpts, template};

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// A loaded local chat model, callable from JavaScript.
#[wasm_bindgen]
pub struct LocalChat {
    model: Model,
    tokenizer: Tokenizer,
    eos: Vec<u32>,
}

#[wasm_bindgen]
impl LocalChat {
    /// Load from quantized GGUF bytes + `tokenizer.json` bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(gguf: &[u8], tokenizer_json: &[u8]) -> Result<LocalChat, JsValue> {
        let device = Device::Cpu;
        let (model, _args, _arch) = Model::from_gguf_reader(Cursor::new(gguf.to_vec()), &device)
            .map_err(err)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json).map_err(err)?;

        let mut eos = Vec::new();
        for t in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = tokenizer.token_to_id(t) {
                eos.push(id);
            }
        }
        Ok(LocalChat {
            model,
            tokenizer,
            eos,
        })
    }

    /// Generate a reply to `prompt` (with optional `system` message). `on_token`
    /// is a JS function called with each decoded text piece as it is produced.
    /// Returns the full generated text. `temperature` 0.0 = greedy.
    pub fn generate(
        &self,
        prompt: &str,
        system: Option<String>,
        max_tokens: usize,
        temperature: f32,
        on_token: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let mut turns = Vec::new();
        if let Some(sys) = system {
            turns.push(template::Turn {
                role: "system",
                content: sys,
            });
        }
        turns.push(template::Turn {
            role: "user",
            content: prompt.to_string(),
        });
        let rendered = template::chatml(&turns);

        let encoding = self.tokenizer.encode(rendered, true).map_err(err)?;
        let ids = encoding.get_ids();

        let opts = SampleOpts {
            temp: temperature,
            top_k: None,
            top_p: None,
        };
        let mut cache = self.model.make_cache(None, 0);
        let mut decoder = self.tokenizer.decode_stream(true);
        let mut full = String::new();
        let this = JsValue::null();

        generate::generate(
            &self.model,
            ids,
            max_tokens,
            &opts,
            &self.eos,
            1,
            &mut cache,
            |id| {
                if let Ok(Some(piece)) = decoder.step(id) {
                    full.push_str(&piece);
                    let _ = on_token.call1(&this, &JsValue::from_str(&piece));
                }
                true
            },
        )
        .map_err(err)?;

        Ok(full)
    }
}

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}
