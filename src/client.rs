use std::sync::{Arc, Mutex, OnceLock};

use chat_core::types::provider_meta::ProviderMeta;
use tokenizers::Tokenizer;

use crate::engine::config::ModelArgs;
use crate::engine::model::Model;
use crate::engine::template::ChatTemplate;
use crate::parsers::tool::ToolFormat;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StructuredMode {
    #[default]
    Prompt,
    Constrained,
}

#[derive(Clone)]
pub struct CandleClient {
    pub(crate) model: Arc<Mutex<Model>>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) args: ModelArgs,
    pub(crate) eos: Vec<u32>,
    pub(crate) model_id: String,
    pub(crate) tokens_per_eval: usize,
    pub(crate) max_context: Option<i32>,
    pub(crate) sink_tokens: i32,
    pub(crate) format: Arc<dyn ToolFormat>,
    pub(crate) template: Arc<ChatTemplate>,
    pub(crate) structured_mode: StructuredMode,
    pub(crate) token_strings: Arc<OnceLock<Arc<Vec<String>>>>,
    pub(crate) meta: Arc<ProviderMeta>,
}

impl CandleClient {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    pub fn provider_meta(&self) -> &ProviderMeta {
        &self.meta
    }

    pub(crate) fn token_strings(&self) -> Arc<Vec<String>> {
        self.token_strings
            .get_or_init(|| {
                let vocab = self.args.vocab_size.max(0) as u32;
                let strings = (0..vocab)
                    .map(|id| self.tokenizer.decode(&[id], false).unwrap_or_default())
                    .collect();
                Arc::new(strings)
            })
            .clone()
    }
}
