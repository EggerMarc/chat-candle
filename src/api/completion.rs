use async_trait::async_trait;

use chat_core::error::{ChatError, ChatFailure};
use chat_core::traits::CompletionProvider;
use chat_core::types::messages::Messages;
use chat_core::types::options::ChatOptions;
use chat_core::types::provider_meta::ProviderMeta;
use chat_core::types::response::ChatResponse;
use chat_core::types::tools::ToolDeclarations;

use crate::api::types::{error, request, response};
use crate::client::{CandleClient, StructuredMode};
use crate::engine::generate;
use crate::parsers::json::JsonConstraint;
use crate::parsers::{reasoning, structured};

#[async_trait]
impl CompletionProvider for CandleClient {
    async fn complete(
        &mut self,
        messages: &mut Messages,
        tool_declarations: Option<&dyn ToolDeclarations>,
        options: Option<&ChatOptions>,
        structured_output: Option<&schemars::Schema>,
    ) -> Result<ChatResponse, ChatFailure> {
        let tools = match tool_declarations {
            Some(d) => Some(
                d.json()
                    .map_err(|e| error::provider(format!("tool declarations: {e}")))?,
            ),
            None => None,
        };

        let prepared = request::from_core(
            messages,
            options,
            structured_output,
            tools.as_ref(),
            &*self.format,
            &self.template,
        )?;

        let want_structured = structured_output.is_some();
        let tools_present = tools.is_some();
        let constrained = want_structured && self.structured_mode == StructuredMode::Constrained;
        // Build the JSON grammar mask up front (decoding the vocab is independent
        // of the model lock).
        let token_strings = constrained.then(|| self.token_strings());

        // Run the synchronous, GPU-bound decode off the async runtime so the
        // caller's executor stays free (e.g. to service a concurrent input
        // stream). The model is held behind `Arc<Mutex<…>>`, so the mutex
        // serialises decodes across clones while the runtime keeps moving.
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let eos = self.eos.clone();
        let sampler = prepared.sampler.clone();
        let prompt = prepared.prompt;
        let max_tokens = prepared.max_tokens;
        let tokens_per_eval = self.tokens_per_eval;
        let max_context = self.max_context;
        let sink_tokens = self.sink_tokens;

        let decode = tokio::task::spawn_blocking(move || -> Result<(String, usize, usize), ChatError> {
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|e| ChatError::InvalidResponse(format!("tokenizer encode: {e}")))?;
            let ids = encoding.get_ids();
            let input_tokens = ids.len();

            let mut model = model
                .lock()
                .map_err(|_| ChatError::Provider("model mutex poisoned".into()))?;
            let mut cache = model.make_cache(max_context, sink_tokens);

            let stats = match token_strings {
                Some(ts) => {
                    let mut con = JsonConstraint::new(ts, eos.clone());
                    generate::generate_constrained(
                        &mut model, ids, max_tokens, &sampler, &eos, &mut cache, &mut con, |_| true,
                    )
                }
                None => generate::generate(
                    &mut model,
                    ids,
                    max_tokens,
                    &sampler,
                    &eos,
                    tokens_per_eval,
                    &mut cache,
                    |_| true,
                ),
            }
            .map_err(|e| ChatError::Provider(format!("generation failed: {e}")))?;
            drop(model);

            let raw = tokenizer
                .decode(&stats.tokens, true)
                .map_err(|e| ChatError::InvalidResponse(format!("tokenizer decode: {e}")))?;
            Ok((raw, input_tokens, stats.tokens.len()))
        });

        let (raw, input_tokens, output_tokens) = decode
            .await
            .map_err(|e| error::provider(format!("decode task failed: {e}")))?
            .map_err(ChatFailure::from_err)?;

        let (reasoning_text, body) = reasoning::split(&raw);

        let mut structured = None;
        let mut calls = Vec::new();
        let mut text = String::new();

        if want_structured {
            // Both modes parse the emitted JSON; constrained decoding has
            // already guaranteed it is well-formed. On a parse miss, hand back
            // the raw text so the chat loop's retry can take over.
            match structured::extract(&body) {
                Some(v) => structured = Some(v),
                None => text = body,
            }
        } else if tools_present {
            let parsed = self.format.parse(&body);
            calls = parsed.calls;
            text = parsed.text;
        } else {
            text = body;
        }

        Ok(response::build(
            &self.model_id,
            reasoning_text,
            text,
            calls,
            structured,
            input_tokens,
            output_tokens,
            max_tokens,
        ))
    }

    fn metadata(&self) -> Option<&ProviderMeta> {
        Some(&self.meta)
    }
}
