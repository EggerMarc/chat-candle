use chat_core::error::{ChatError, ChatFailure};

/// A provider-side failure (model/engine error).
pub fn provider(msg: impl Into<String>) -> ChatFailure {
    ChatFailure::from_err(ChatError::Provider(msg.into()))
}

/// A malformed / unexpected result (tokenizer or parsing bug, not a model failure).
pub fn invalid(msg: impl Into<String>) -> ChatFailure {
    ChatFailure::from_err(ChatError::InvalidResponse(msg.into()))
}

/// A capability not yet implemented by this provider.
pub fn unsupported(what: &str) -> ChatFailure {
    ChatFailure::from_err(ChatError::Provider(format!(
        "chat-candle does not yet support {what}"
    )))
}
