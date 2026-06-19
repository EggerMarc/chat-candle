use std::collections::{BTreeSet, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_nn::VarBuilder;
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiRepo},
};
use tokenizers::Tokenizer;

use crate::engine::config::{GenerationConfig, HfConfig, ModelArgs};
use crate::engine::model::{self, Model};
use crate::engine::template::ChatTemplate;

pub struct Loaded {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub args: ModelArgs,
    pub eos: Vec<u32>,
    pub model_type: String,
    pub chat_template: ChatTemplate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantize {
    Q2,
    Q3,
    Q4,
    Q8,
}

impl Quantize {
    pub fn bits(self) -> i32 {
        match self {
            Quantize::Q2 => 2,
            Quantize::Q3 => 3,
            Quantize::Q4 => 4,
            Quantize::Q8 => 8,
        }
    }

    pub fn group_size(self) -> i32 {
        64
    }

    pub fn from_bits(bits: i32) -> Option<Quantize> {
        match bits {
            2 => Some(Quantize::Q2),
            3 => Some(Quantize::Q3),
            4 => Some(Quantize::Q4),
            8 => Some(Quantize::Q8),
            _ => None,
        }
    }
}

pub fn load(model_id: &str, quant: Option<Quantize>) -> Result<Loaded> {
    if quant.is_some() {
        // TODO: load pre-quantized GGUF via `candle_core::quantized` (the q4 GEMV
        // lever). Runtime quantization of fp safetensors isn't candle's model;
        // for now we load fp weights and ignore the flag.
        eprintln!("[warn] --quantize not yet supported on the candle backend; loading fp weights");
    }

    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json").context("fetch config.json")?;
    let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
    let weight_paths = resolve_weights(&repo).context("resolve model weight files")?;

    let gen_cfg: GenerationConfig = match repo.get("generation_config.json") {
        Ok(p) => serde_json::from_reader(std::fs::File::open(p)?)?,
        Err(_) => GenerationConfig::default(),
    };

    let hf: HfConfig = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let model_type = hf.model_type.clone();
    let cfg_tied = hf.tie_word_embeddings;
    let mut args: ModelArgs = hf.into();

    let arch = detect_arch(&repo, &weight_paths)?;
    args.use_qk_norm = arch.use_qk_norm;
    args.attn_qkv_bias = arch.attn_qkv_bias;
    args.attn_o_bias = arch.attn_o_bias;
    args.tie_word_embeddings = cfg_tied || arch.no_lm_head;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
    let (chat_template, eos_token) = read_chat_config(&repo);

    let mut eos = gen_cfg.eos_token_id.clone();
    if eos.is_empty()
        && !eos_token.is_empty()
        && let Some(id) = tokenizer.token_to_id(&eos_token)
    {
        eos.push(id);
    }

    // Build the model directly from the mmaped weights: candle fuses
    // construction and weight-load through the `VarBuilder` (keyed by HF tensor
    // names), so there is no separate `load_safetensors` / `tie_lm_head` step —
    // tying is handled inside `Model::new`.
    let device = model::default_device()?;
    let dtype = model::default_dtype(&device);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, &device)? };
    let model = Model::new(&args, vb)
        .context("build model (struct field names must match HF tensor keys)")?;

    Ok(Loaded {
        model,
        tokenizer,
        args,
        eos,
        model_type,
        chat_template,
    })
}

struct Arch {
    use_qk_norm: bool,
    attn_qkv_bias: bool,
    attn_o_bias: bool,
    no_lm_head: bool,
}

fn detect_arch(repo: &ApiRepo, weight_paths: &[PathBuf]) -> Result<Arch> {
    let names = tensor_names(repo, weight_paths)?;
    let any = |suffix: &str| names.iter().any(|n| n.ends_with(suffix));
    Ok(Arch {
        use_qk_norm: any(".self_attn.q_norm.weight"),
        attn_qkv_bias: any(".self_attn.q_proj.bias"),
        attn_o_bias: any(".self_attn.o_proj.bias"),
        no_lm_head: !names.contains("lm_head.weight"),
    })
}

fn tensor_names(repo: &ApiRepo, weight_paths: &[PathBuf]) -> Result<HashSet<String>> {
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;
        if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
            return Ok(map.keys().cloned().collect());
        }
    }
    let first = weight_paths.first().context("no weight files")?;
    safetensors_header_keys(first)
}

fn safetensors_header_keys(path: &Path) -> Result<HashSet<String>> {
    let mut f = std::fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len) as usize;
    let mut hdr = vec![0u8; n];
    f.read_exact(&mut hdr)?;
    let v: serde_json::Value = serde_json::from_slice(&hdr)?;
    Ok(v.as_object()
        .map(|o| {
            o.keys()
                .filter(|k| k.as_str() != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default())
}

fn read_chat_config(repo: &ApiRepo) -> (ChatTemplate, String) {
    let cfg = repo
        .get("tokenizer_config.json")
        .ok()
        .and_then(|p| std::fs::File::open(p).ok())
        .and_then(|f| serde_json::from_reader::<_, serde_json::Value>(f).ok());

    let Some(cfg) = cfg else {
        return (ChatTemplate::chatml_only(), String::new());
    };

    let template = cfg
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(String::from);
    let eos = token_str(cfg.get("eos_token"));
    (
        ChatTemplate::new(template, token_str(cfg.get("bos_token")), eos.clone()),
        eos,
    )
}

fn token_str(v: Option<&serde_json::Value>) -> String {
    match v {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(o)) => o
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

pub fn resolve_weights(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index.json missing weight_map")?;
        let shards: BTreeSet<String> = map
            .values()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        let mut paths = Vec::with_capacity(shards.len());
        for shard in shards {
            paths.push(repo.get(&shard).with_context(|| format!("fetch {shard}"))?);
        }
        return Ok(paths);
    }
    Ok(vec![
        repo.get("model.safetensors")
            .context("fetch model.safetensors")?,
    ])
}
