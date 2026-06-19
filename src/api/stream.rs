use async_trait::async_trait;
use chat_core::error::ChatError;
use chat_core::traits::StreamProvider;
use chat_core::types::messages::Messages;
use chat_core::types::options::ChatOptions;
use chat_core::types::response::StreamEvent;
use chat_core::types::tools::ToolDeclarations;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;

use crate::api::types::{request, response};
use crate::client::{CandleClient, StructuredMode};
use crate::engine::generate;
use crate::parsers::json::JsonConstraint;
use crate::parsers::reasoning::{Chunk, ReasoningSplitter};
use crate::parsers::structured;
use crate::parsers::tool::ToolCallStripper;

#[async_trait]
impl StreamProvider for CandleClient {
    async fn stream(
        &mut self,
        messages: &mut Messages,
        tool_declarations: Option<&dyn ToolDeclarations>,
        options: Option<&ChatOptions>,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ChatError>>, ChatError> {
        let tools = match tool_declarations {
            Some(d) => Some(
                d.json()
                    .map_err(|e| ChatError::Provider(format!("tool declarations: {e}")))?,
            ),
            None => None,
        };

        let prepared =
            request::from_core(messages, options, None, tools.as_ref(), &*self.format, &self.template)
                .map_err(|f| f.err)?;
        let tools_present = tools.is_some();

        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let format = self.format.clone();
        let eos = self.eos.clone();
        let model_id = self.model_id.clone();
        let tokens_per_eval = self.tokens_per_eval;
        let max_context = self.max_context;
        let sink_tokens = self.sink_tokens;
        let max_tokens = prepared.max_tokens;
        let sampler = prepared.sampler.clone();
        let prompt = prepared.prompt;

        let (tx, mut rx) = mpsc::unbounded_channel::<Result<StreamEvent, ChatError>>();

        // The decode loop is synchronous and blocking; run it off the async
        // runtime and forward events through the channel. The mutex guard is
        // confined to this blocking task.
        tokio::task::spawn_blocking(move || {
            let encoding = match tokenizer.encode(prompt, true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(Err(ChatError::InvalidResponse(format!(
                        "tokenizer encode: {e}"
                    ))));
                    return;
                }
            };
            let ids = encoding.get_ids();
            let input_tokens = ids.len();

            let mut model = match model.lock() {
                Ok(m) => m,
                Err(_) => {
                    let _ = tx.send(Err(ChatError::Provider("model mutex poisoned".into())));
                    return;
                }
            };
            let mut cache = model.make_cache(max_context, sink_tokens);
            let mut decoder = tokenizer.decode_stream(true);
            let mut splitter = ReasoningSplitter::new();
            // While tools are active, hide in-progress `<tool_call>` markup from
            // the live text; the calls still surface as `Tool` parts at `Done`.
            let mut stripper = if tools_present {
                format
                    .call_delimiters()
                    .map(|(o, c)| ToolCallStripper::new(o, c))
            } else {
                None
            };

            let route = |chunk: Chunk,
                         stripper: &mut Option<ToolCallStripper>,
                         tx: &mpsc::UnboundedSender<Result<StreamEvent, ChatError>>| {
                match chunk {
                    Chunk::Reasoning(s) => {
                        let _ = tx.send(Ok(StreamEvent::ReasoningChunk(s)));
                    }
                    Chunk::Text(s) => {
                        let shown = match stripper.as_mut() {
                            Some(st) => st.push(&s),
                            None => s,
                        };
                        if !shown.is_empty() {
                            let _ = tx.send(Ok(StreamEvent::TextChunk(shown)));
                        }
                    }
                }
            };

            let result = generate::generate(
                &mut model,
                ids,
                max_tokens,
                &sampler,
                &eos,
                tokens_per_eval,
                &mut cache,
                |id| {
                    if let Ok(Some(piece)) = decoder.step(id) {
                        for chunk in splitter.push(&piece) {
                            route(chunk, &mut stripper, &tx);
                        }
                    }
                    // Stop generating if the consumer dropped the stream (e.g.
                    // a new message restarted the exchange).
                    !tx.is_closed()
                },
            );

            let stats = match result {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(ChatError::Provider(format!("generation failed: {e}"))));
                    return;
                }
            };
            for chunk in splitter.flush() {
                route(chunk, &mut stripper, &tx);
            }
            if let Some(st) = stripper.as_mut() {
                let tail = st.flush();
                if !tail.is_empty() {
                    let _ = tx.send(Ok(StreamEvent::TextChunk(tail)));
                }
            }
            drop(model);

            let reasoning_text = std::mem::take(&mut splitter.reasoning);
            let body = std::mem::take(&mut splitter.text);
            let (calls, text) = if tools_present {
                let parsed = format.parse(&body);
                (parsed.calls, parsed.text)
            } else {
                (Vec::new(), body)
            };

            // Surface the parsed calls to the consumer before the terminal
            // event; the chat loop executes them off `Done`'s content.
            for call in &calls {
                let _ = tx.send(Ok(StreamEvent::ToolCall(call.clone())));
            }

            let resp = response::build(
                &model_id,
                reasoning_text,
                text,
                calls,
                None,
                input_tokens,
                stats.tokens.len(),
                max_tokens,
            );
            let _ = tx.send(Ok(StreamEvent::Done(resp)));
        });

        let s = async_stream::stream! {
            while let Some(ev) = rx.recv().await {
                yield ev;
            }
        };
        Ok(s.boxed())
    }
}

impl CandleClient {
    /// Stream a structured-output generation. The JSON is produced live as
    /// `TextChunk`s (any `<think>` reasoning as `ReasoningChunk`), and a final
    /// `StreamEvent::Structured(value)` is emitted just before `Done`. `mode`
    /// selects prompt-and-parse vs. grammar-constrained decoding.
    ///
    /// This is a provider-native method: chat-core's `StreamProvider::stream`
    /// carries no schema, so structured streaming can't go through the generic
    /// trait.
    pub fn stream_structured(
        &self,
        messages: &Messages,
        options: Option<&ChatOptions>,
        schema: &schemars::Schema,
        mode: StructuredMode,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ChatError>>, ChatError> {
        let prepared = request::from_core(
            messages,
            options,
            Some(schema),
            None,
            &*self.format,
            &self.template,
        )
        .map_err(|f| f.err)?;

        // Build the grammar mask up front (decoding the vocab is independent of
        // the model lock).
        let token_strings = match mode {
            StructuredMode::Constrained => Some(self.token_strings()),
            StructuredMode::Prompt => None,
        };

        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let eos = self.eos.clone();
        let model_id = self.model_id.clone();
        let tokens_per_eval = self.tokens_per_eval;
        let max_context = self.max_context;
        let sink_tokens = self.sink_tokens;
        let max_tokens = prepared.max_tokens;
        let sampler = prepared.sampler.clone();
        let prompt = prepared.prompt;

        let (tx, mut rx) = mpsc::unbounded_channel::<Result<StreamEvent, ChatError>>();

        tokio::task::spawn_blocking(move || {
            let encoding = match tokenizer.encode(prompt, true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(Err(ChatError::InvalidResponse(format!(
                        "tokenizer encode: {e}"
                    ))));
                    return;
                }
            };
            let ids = encoding.get_ids();
            let input_tokens = ids.len();

            let mut model = match model.lock() {
                Ok(m) => m,
                Err(_) => {
                    let _ = tx.send(Err(ChatError::Provider("model mutex poisoned".into())));
                    return;
                }
            };
            let mut cache = model.make_cache(max_context, sink_tokens);
            let mut decoder = tokenizer.decode_stream(true);
            let mut splitter = ReasoningSplitter::new();

            let emit = |id: u32| {
                if let Ok(Some(piece)) = decoder.step(id) {
                    for chunk in splitter.push(&piece) {
                        let _ = tx.send(Ok(chunk_event(chunk)));
                    }
                }
                !tx.is_closed()
            };

            let result = match token_strings {
                Some(ts) => {
                    let mut con = JsonConstraint::new(ts, eos.clone());
                    generate::generate_constrained(
                        &mut model, ids, max_tokens, &sampler, &eos, &mut cache, &mut con, emit,
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
                    emit,
                ),
            };

            let stats = match result {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(ChatError::Provider(format!("generation failed: {e}"))));
                    return;
                }
            };
            for chunk in splitter.flush() {
                let _ = tx.send(Ok(chunk_event(chunk)));
            }
            drop(model);

            let reasoning_text = std::mem::take(&mut splitter.reasoning);
            let body = std::mem::take(&mut splitter.text);
            let structured = structured::extract(&body);
            if let Some(v) = &structured {
                let _ = tx.send(Ok(StreamEvent::Structured(v.clone())));
            }
            let text = if structured.is_some() {
                String::new()
            } else {
                body
            };
            let resp = response::build(
                &model_id,
                reasoning_text,
                text,
                Vec::new(),
                structured,
                input_tokens,
                stats.tokens.len(),
                max_tokens,
            );
            let _ = tx.send(Ok(StreamEvent::Done(resp)));
        });

        let s = async_stream::stream! {
            while let Some(ev) = rx.recv().await {
                yield ev;
            }
        };
        Ok(s.boxed())
    }
}

fn chunk_event(chunk: Chunk) -> StreamEvent {
    match chunk {
        Chunk::Reasoning(s) => StreamEvent::ReasoningChunk(s),
        Chunk::Text(s) => StreamEvent::TextChunk(s),
    }
}
