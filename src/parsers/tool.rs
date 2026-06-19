use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;
use tools_rs::{FunctionCall, FunctionResponse};

pub struct ParsedTools {
    pub calls: Vec<FunctionCall>,
    pub text: String,
}

pub trait ToolFormat: Send + Sync {
    fn system_with_tools(&self, base: &str, tools: &Value) -> String;
    fn render_call(&self, call: &FunctionCall) -> String;
    fn render_result(&self, resp: &FunctionResponse) -> (&'static str, String);
    fn parse(&self, text: &str) -> ParsedTools;
    fn call_delimiters(&self) -> Option<(String, String)> {
        None
    }
}

pub fn detect(model_type: &str) -> Arc<dyn ToolFormat> {
    let mt = model_type.to_lowercase();
    if mt.contains("mistral") || mt.contains("mixtral") {
        Arc::new(Mistral)
    } else if mt.contains("llama") {
        Arc::new(Json)
    } else {
        Arc::new(Hermes)
    }
}

#[derive(Deserialize)]
struct RawCall {
    name: String,
    #[serde(default, alias = "parameters")]
    arguments: Value,
}

fn raw_to_call(raw: &str) -> Option<FunctionCall> {
    let parsed: RawCall = serde_json::from_str(raw.trim()).ok()?;
    let arguments = match parsed.arguments {
        Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
        other => other,
    };
    Some(FunctionCall {
        id: None,
        name: parsed.name,
        arguments,
    })
}

fn extract_spans(text: &str, open: &str, close: &str) -> (Vec<String>, String) {
    let mut inners = Vec::new();
    let mut residual = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        residual.push_str(&rest[..start]);
        let after = &rest[start + open.len()..];
        match after.find(close) {
            Some(end) => {
                inners.push(after[..end].to_string());
                rest = &after[end + close.len()..];
            }
            None => {
                residual.push_str(&rest[start..]);
                rest = "";
                break;
            }
        }
    }
    residual.push_str(rest);
    (inners, residual)
}

fn balanced_end(b: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &c) in b.iter().enumerate().skip(start) {
        if in_str {
            match c {
                _ if esc => esc = false,
                b'\\' => esc = true,
                b'"' => in_str = false,
                _ => {}
            }
        } else {
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Pull every balanced `{…}` that parses as a `{name, arguments|parameters}`
/// call out of free text (no delimiters), returning the calls and the residual.
fn parse_json_calls(text: &str) -> ParsedTools {
    let b = text.as_bytes();
    let mut calls = Vec::new();
    let mut residual = String::new();
    let mut i = 0;
    let mut copy_from = 0;
    while i < b.len() {
        if b[i] == b'{'
            && let Some(end) = balanced_end(b, i)
            && let Some(call) = raw_to_call(&text[i..=end])
        {
            residual.push_str(&text[copy_from..i]);
            calls.push(call);
            i = end + 1;
            copy_from = i;
            continue;
        }
        i += 1;
    }
    residual.push_str(&text[copy_from..]);
    ParsedTools {
        calls,
        text: residual.trim().to_string(),
    }
}

pub struct Hermes;

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

impl ToolFormat for Hermes {
    fn system_with_tools(&self, base: &str, tools: &Value) -> String {
        let mut lines = String::new();
        if let Value::Array(arr) = tools {
            for decl in arr {
                let wrapped = serde_json::json!({ "type": "function", "function": decl });
                lines.push_str(&wrapped.to_string());
                lines.push('\n');
            }
        }
        let mut out = String::new();
        if !base.is_empty() {
            out.push_str(base);
            out.push_str("\n\n");
        }
        out.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
        );
        out.push_str(&lines);
        out.push_str(
            "</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );
        out
    }

    fn render_call(&self, call: &FunctionCall) -> String {
        let obj = serde_json::json!({ "name": call.name, "arguments": call.arguments });
        format!("{TOOL_CALL_OPEN}\n{obj}\n{TOOL_CALL_CLOSE}")
    }

    fn render_result(&self, resp: &FunctionResponse) -> (&'static str, String) {
        (
            "user",
            format!("<tool_response>\n{}\n</tool_response>", resp.result),
        )
    }

    fn parse(&self, text: &str) -> ParsedTools {
        let (inners, residual) = extract_spans(text, TOOL_CALL_OPEN, TOOL_CALL_CLOSE);
        let calls = inners.iter().filter_map(|s| raw_to_call(s)).collect();
        ParsedTools {
            calls,
            text: residual.trim().to_string(),
        }
    }

    fn call_delimiters(&self) -> Option<(String, String)> {
        Some((TOOL_CALL_OPEN.to_string(), TOOL_CALL_CLOSE.to_string()))
    }
}

pub struct Pattern {
    pub open: String,
    pub close: String,
}

impl ToolFormat for Pattern {
    fn system_with_tools(&self, base: &str, tools: &Value) -> String {
        let mut out = String::new();
        if !base.is_empty() {
            out.push_str(base);
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "You can call functions. Available functions (JSON):\n{tools}\n\n\
             To call one, emit {}{{\"name\": <name>, \"arguments\": <args>}}{}",
            self.open, self.close
        ));
        out
    }

    fn render_call(&self, call: &FunctionCall) -> String {
        let obj = serde_json::json!({ "name": call.name, "arguments": call.arguments });
        format!("{}{}{}", self.open, obj, self.close)
    }

    fn render_result(&self, resp: &FunctionResponse) -> (&'static str, String) {
        ("user", format!("[tool result] {}", resp.result))
    }

    fn parse(&self, text: &str) -> ParsedTools {
        let (inners, residual) = extract_spans(text, &self.open, &self.close);
        let calls = inners.iter().filter_map(|s| raw_to_call(s)).collect();
        ParsedTools {
            calls,
            text: residual.trim().to_string(),
        }
    }

    fn call_delimiters(&self) -> Option<(String, String)> {
        Some((self.open.clone(), self.close.clone()))
    }
}

pub struct Json;

impl ToolFormat for Json {
    fn system_with_tools(&self, base: &str, tools: &Value) -> String {
        let mut out = String::new();
        if !base.is_empty() {
            out.push_str(base);
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "You have access to the following functions (JSON Schema):\n{tools}\n\n\
             To call a function, respond with a single JSON object of the form \
             {{\"name\": <function-name>, \"parameters\": <arguments-json-object>}} and nothing else.",
        ));
        out
    }

    fn render_call(&self, call: &FunctionCall) -> String {
        serde_json::json!({ "name": call.name, "parameters": call.arguments }).to_string()
    }

    fn render_result(&self, resp: &FunctionResponse) -> (&'static str, String) {
        ("tool", resp.result.to_string())
    }

    fn parse(&self, text: &str) -> ParsedTools {
        parse_json_calls(text)
    }
}

pub struct Mistral;

const MISTRAL_MARKER: &str = "[TOOL_CALLS]";

impl ToolFormat for Mistral {
    fn system_with_tools(&self, base: &str, tools: &Value) -> String {
        let mut out = String::new();
        if !base.is_empty() {
            out.push_str(base);
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "You have access to the following functions (JSON Schema):\n{tools}\n\n\
             To call functions, respond with {MISTRAL_MARKER} followed by a JSON array of \
             {{\"name\": <function-name>, \"arguments\": <arguments-json-object>}} objects.",
        ));
        out
    }

    fn render_call(&self, call: &FunctionCall) -> String {
        let obj = serde_json::json!([{ "name": call.name, "arguments": call.arguments }]);
        format!("{MISTRAL_MARKER}{obj}")
    }

    fn render_result(&self, resp: &FunctionResponse) -> (&'static str, String) {
        ("tool", resp.result.to_string())
    }

    fn parse(&self, text: &str) -> ParsedTools {
        match text.find(MISTRAL_MARKER) {
            Some(idx) => {
                let calls = parse_json_calls(&text[idx + MISTRAL_MARKER.len()..]).calls;
                ParsedTools {
                    calls,
                    text: text[..idx].trim().to_string(),
                }
            }
            None => ParsedTools {
                calls: Vec::new(),
                text: text.trim().to_string(),
            },
        }
    }
}

pub struct ToolCallStripper {
    open: String,
    close: String,
    inside: bool,
    pending: String,
}

impl ToolCallStripper {
    pub fn new(open: String, close: String) -> Self {
        Self {
            open,
            close,
            inside: false,
            pending: String::new(),
        }
    }

    pub fn push(&mut self, piece: &str) -> String {
        self.pending.push_str(piece);
        let mut out = String::new();
        loop {
            if !self.inside {
                if let Some(i) = self.pending.find(&self.open) {
                    out.push_str(&self.pending[..i]);
                    self.pending.drain(..i + self.open.len());
                    self.inside = true;
                    continue;
                }
                let keep = super::partial_suffix_len(&self.pending, &self.open);
                let n = self.pending.len() - keep;
                let emit: String = self.pending.drain(..n).collect();
                out.push_str(&emit);
                break;
            }
            if let Some(i) = self.pending.find(&self.close) {
                self.pending.drain(..i + self.close.len());
                self.inside = false;
                continue;
            }
            // Still inside a call: discard all but a possible partial close.
            let keep = super::partial_suffix_len(&self.pending, &self.close);
            let n = self.pending.len() - keep;
            self.pending.drain(..n);
            break;
        }
        out
    }

    pub fn flush(&mut self) -> String {
        if self.inside {
            self.pending.clear();
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermes_parses_call_and_residual() {
        let out = Hermes.parse(
            "Sure.<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(out.calls[0].arguments["city"], "Paris");
        assert_eq!(out.text, "Sure.");
    }

    /// Feed the text one byte at a time: the tool-call span must be fully
    /// suppressed even though the delimiters are split across pushes.
    #[test]
    fn stripper_hides_call_across_boundaries() {
        let (open, close) = Hermes.call_delimiters().unwrap();
        let mut st = ToolCallStripper::new(open, close);
        let input = "Hi <tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call> done";
        let mut shown = String::new();
        for ch in input.chars() {
            shown.push_str(&st.push(&ch.to_string()));
        }
        shown.push_str(&st.flush());
        assert_eq!(shown, "Hi  done");
    }

    #[test]
    fn pattern_strips_custom_delimiters() {
        let p = Pattern {
            open: "[[".into(),
            close: "]]".into(),
        };
        let out = p.parse("a[[{\"name\":\"f\",\"arguments\":{}}]]b");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.text, "ab");
    }

    #[test]
    fn json_parses_bare_object_with_parameters() {
        // Llama-style: bare JSON object using the `parameters` key.
        let out = Json.parse(
            "Let me check. {\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}",
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(out.calls[0].arguments["city"], "Paris");
        assert_eq!(out.text, "Let me check.");
    }

    #[test]
    fn json_ignores_non_call_objects() {
        let out = Json.parse("{\"unrelated\": 1}");
        assert!(out.calls.is_empty());
        assert_eq!(out.text, "{\"unrelated\": 1}");
    }

    #[test]
    fn mistral_parses_tool_calls_array() {
        let out = Mistral.parse(
            "sure[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}]",
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(out.calls[0].arguments["city"], "Paris");
        assert_eq!(out.text, "sure");
    }

    #[test]
    fn detect_routes_by_family() {
        // Smoke-test routing via each format's distinctive render.
        assert!(detect("llama").render_call(&fc()).starts_with('{'));
        assert!(
            detect("mistral")
                .render_call(&fc())
                .starts_with("[TOOL_CALLS]")
        );
        assert!(
            detect("qwen3")
                .render_call(&fc())
                .starts_with("<tool_call>")
        );
    }

    fn fc() -> FunctionCall {
        FunctionCall {
            id: None,
            name: "f".into(),
            arguments: serde_json::json!({}),
        }
    }
}
