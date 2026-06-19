use chat_core::error::ChatFailure;
use chat_core::types::messages::Messages;
use chat_core::types::messages::content::RoleEnum;
use chat_core::types::messages::parts::PartEnum;
use chat_core::types::options::ChatOptions;
use serde_json::Value;

use crate::api::types::error::{invalid, unsupported};
use crate::engine::sampler::SampleOpts;
use crate::engine::template::{ChatTemplate, Turn};
use crate::parsers::structured;
use crate::parsers::tool::ToolFormat;

const DEFAULT_MAX_TOKENS: usize = 512;

/// A request lowered from chat-core types into what the engine consumes.
pub struct Prepared {
    pub prompt: String,
    pub sampler: SampleOpts,
    pub max_tokens: usize,
}

/// Lower chat-core `Messages` + `ChatOptions` into a prompt string and sampling
/// config. When `tools` is `Some`, the declarations are advertised in the
/// system prompt and prior `Tool` parts (calls + results) are rendered back
/// using `format`. When `structured_output` is `Some`, its schema is appended
/// to the system prompt as an instruction (both structured modes do this; the
/// constrained mode additionally masks logits during decoding).
pub fn from_core(
    messages: &Messages,
    options: Option<&ChatOptions>,
    structured_output: Option<&schemars::Schema>,
    tools: Option<&Value>,
    format: &dyn ToolFormat,
    template: &ChatTemplate,
) -> Result<Prepared, ChatFailure> {
    let instr = match structured_output {
        Some(schema) => {
            let value = serde_json::to_value(schema)
                .map_err(|e| invalid(format!("structured-output schema: {e}")))?;
            Some(structured::instruction(&value))
        }
        None => None,
    };
    let has_system_extras = tools.is_some() || instr.is_some();

    let mut turns: Vec<Turn> = Vec::new();
    let mut injected = false;

    for content in &messages.0 {
        match content.role {
            RoleEnum::System => {
                let base = flatten_text(&content.parts.0);
                turns.push(Turn {
                    role: "system",
                    content: compose_system(&base, tools, instr.as_deref(), format),
                });
                injected = has_system_extras;
            }
            RoleEnum::User => turns.push(Turn {
                role: "user",
                content: flatten_text(&content.parts.0),
            }),
            RoleEnum::Model => {
                let (assistant, results) = render_model(&content.parts.0, format)?;
                turns.push(Turn {
                    role: "assistant",
                    content: assistant,
                });
                turns.extend(results);
            }
        }
    }

    // Extras to advertise but no system message to host them: prepend one.
    if has_system_extras && !injected {
        turns.insert(
            0,
            Turn {
                role: "system",
                content: compose_system("", tools, instr.as_deref(), format),
            },
        );
    }

    let prompt = template.render(&turns);
    let (sampler, max_tokens) = sampler_from_options(options);
    Ok(Prepared {
        prompt,
        sampler,
        max_tokens,
    })
}

/// Build a system message body from base text + optional tool advert + optional
/// structured-output instruction.
fn compose_system(
    base: &str,
    tools: Option<&Value>,
    instr: Option<&str>,
    format: &dyn ToolFormat,
) -> String {
    let mut body = match tools {
        Some(t) => format.system_with_tools(base, t),
        None => base.to_string(),
    };
    if let Some(i) = instr {
        if !body.is_empty() {
            body.push_str("\n\n");
        }
        body.push_str(i);
    }
    body
}

fn flatten_text(parts: &[PartEnum]) -> String {
    let mut buf = String::new();
    for part in parts {
        if let PartEnum::Text(t) = part {
            append_line(&mut buf, t.as_str());
        }
    }
    buf
}

/// Render a model turn: text parts + tool calls go into the assistant content;
/// each resolved tool result becomes its own follow-up turn.
fn render_model(
    parts: &[PartEnum],
    format: &dyn ToolFormat,
) -> Result<(String, Vec<Turn>), ChatFailure> {
    let mut content = String::new();
    let mut results = Vec::new();
    for part in parts {
        match part {
            PartEnum::Text(t) => append_line(&mut content, t.as_str()),
            // Prior chain-of-thought is not re-fed to the model.
            PartEnum::Reasoning(_) => {}
            PartEnum::Tool(tool) => {
                let (call, response) = tool.to_tuple();
                append_line(&mut content, &format.render_call(&call));
                if let Some(resp) = response {
                    let (role, body) = format.render_result(&resp);
                    results.push(Turn {
                        role,
                        content: body,
                    });
                }
            }
            PartEnum::Structured(_) => {}
            PartEnum::File(_) => return Err(unsupported("file parts")),
            PartEnum::Embeddings(_) => return Err(unsupported("embedding parts in input")),
        }
    }
    Ok((content, results))
}

fn append_line(buf: &mut String, s: &str) {
    if s.is_empty() {
        return;
    }
    if !buf.is_empty() {
        buf.push('\n');
    }
    buf.push_str(s);
}

fn sampler_from_options(options: Option<&ChatOptions>) -> (SampleOpts, usize) {
    let mut sampler = SampleOpts {
        temp: 0.0,
        top_k: None,
        top_p: None,
    };
    let mut max_tokens = DEFAULT_MAX_TOKENS;

    if let Some(o) = options {
        if let Some(t) = o.temperature {
            sampler.temp = t;
        }
        if let Some(p) = o.top_p {
            sampler.top_p = Some(p);
        }
        if let Some(m) = o.max_tokens {
            max_tokens = m as usize;
        }
        if let Some(k) = o.metadata.get("top_k").and_then(|v| v.as_u64()) {
            sampler.top_k = Some(k as usize);
        }
    }

    (sampler, max_tokens)
}
