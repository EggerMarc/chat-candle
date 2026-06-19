use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone)]
pub struct ModelArgs {
    pub dim: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub hidden_dim: i32,
    pub vocab_size: i32,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub use_qk_norm: bool,
    pub attn_qkv_bias: bool,
    pub attn_o_bias: bool,
}

impl From<HfConfig> for ModelArgs {
    fn from(c: HfConfig) -> Self {
        let head_dim = c.head_dim.unwrap_or(c.hidden_size / c.num_attention_heads);
        let bias = c.attention_bias.unwrap_or(false);
        Self {
            dim: c.hidden_size,
            n_layers: c.num_hidden_layers,
            n_heads: c.num_attention_heads,
            n_kv_heads: c.num_key_value_heads,
            head_dim,
            hidden_dim: c.intermediate_size,
            vocab_size: c.vocab_size,
            norm_eps: c.rms_norm_eps,
            rope_theta: c.rope_theta,
            tie_word_embeddings: c.tie_word_embeddings,
            use_qk_norm: false,
            attn_qkv_bias: bias,
            attn_o_bias: bias,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct GenerationConfig {
    #[serde(default, deserialize_with = "de_eos")]
    pub eos_token_id: Vec<u32>,
}

fn de_eos<'de, D>(d: D) -> Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::Number(n) => Ok(vec![n.as_u64().unwrap_or(0) as u32]),
        serde_json::Value::Array(a) => Ok(a
            .into_iter()
            .filter_map(|x| x.as_u64().map(|n| n as u32))
            .collect()),
        serde_json::Value::Null => Ok(vec![]),
        other => Err(D::Error::custom(format!(
            "unexpected eos_token_id: {other}"
        ))),
    }
}

impl GenerationConfig {
    pub fn eos_or_default(&self) -> Vec<u32> {
        if self.eos_token_id.is_empty() {
            vec![1, 130073]
        } else {
            self.eos_token_id.clone()
        }
    }
}
