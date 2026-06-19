use std::io::Write;

use anyhow::Result;
use clap::Parser;

use chat_candle::engine::{generate, sampler::SampleOpts, template};
use chat_candle::loader;

#[derive(Parser)]
#[command(about = "Standalone candle inference for Qwen3 / Llama / MiniCPM-family models")]
struct Cli {
    /// HF repo id of the model (fp16/bf16 safetensors).
    #[clap(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,

    /// User message.
    #[clap(
        long,
        default_value = "Explain rotary position embeddings in one sentence."
    )]
    prompt: String,

    /// Optional system prompt.
    #[clap(long)]
    system: Option<String>,

    /// Max tokens to generate.
    #[clap(long, default_value = "256")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[clap(long, default_value = "0.0")]
    temp: f32,

    #[clap(long)]
    top_k: Option<usize>,

    #[clap(long)]
    top_p: Option<f32>,

    /// Max tokens retained in the KV cache. 0 = unbounded.
    #[clap(long, default_value = "4096")]
    max_context: i32,

    /// Leading tokens pinned as attention sinks when the window rotates.
    #[clap(long, default_value = "4")]
    sink_tokens: i32,

    /// GGUF repo for quantized weights (e.g. `Qwen/Qwen3-0.6B-GGUF`). The
    /// tokenizer is taken from `--model`.
    #[clap(long)]
    gguf: Option<String>,

    /// GGUF filename within `--gguf` (e.g. `Qwen3-0.6B-Q4_K_M.gguf`).
    #[clap(long)]
    gguf_file: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let loaded = match (&cli.gguf, &cli.gguf_file) {
        (Some(repo), Some(file)) => {
            eprintln!("[info] loading quantized {file} from {repo} …");
            loader::load_gguf(repo, file, &cli.model)?
        }
        (None, None) => {
            eprintln!("[info] loading {} …", cli.model);
            loader::load(&cli.model, None)?
        }
        _ => anyhow::bail!("--gguf and --gguf-file must be provided together"),
    };
    let args = &loaded.args;
    eprintln!(
        "[info] model: dim={} layers={} heads={}/{} head_dim={} vocab={} | device={:?}",
        args.dim,
        args.n_layers,
        args.n_heads,
        args.n_kv_heads,
        args.head_dim,
        args.vocab_size,
        loaded.model.device(),
    );
    let m = loaded.model;
    let tokenizer = loaded.tokenizer;
    let eos = loaded.eos;

    let mut turns = Vec::new();
    if let Some(sys) = cli.system.as_deref() {
        turns.push(template::Turn {
            role: "system",
            content: sys.to_string(),
        });
    }
    turns.push(template::Turn {
        role: "user",
        content: cli.prompt.clone(),
    });
    let prompt = loaded.chat_template.render(&turns);
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let ids = encoding.get_ids();
    eprintln!("[info] prompt tokens: {}", ids.len());

    print!("{}", cli.prompt);
    let _ = std::io::stdout().flush();

    let opts = SampleOpts {
        temp: cli.temp,
        top_k: cli.top_k,
        top_p: cli.top_p,
    };
    let mut stream = tokenizer.decode_stream(true);
    let mut emit = |id: u32| -> bool {
        if let Ok(Some(s)) = stream.step(id) {
            print!("{s}");
            let _ = std::io::stdout().flush();
        }
        true
    };

    let max_context = (cli.max_context > 0).then_some(cli.max_context);
    let mut kv_cache = m.make_cache(max_context, cli.sink_tokens);
    let stats = generate::generate(
        &m,
        ids,
        cli.max_tokens,
        &opts,
        &eos,
        1,
        &mut kv_cache,
        &mut emit,
    )?;
    println!();

    let n = stats.tokens.len();
    eprintln!(
        "[info] prefill {} tok in {:.3}s ({:.1} tok/s) | decode {} tok in {:.3}s ({:.1} tok/s)",
        ids.len(),
        stats.prefill_secs,
        ids.len() as f64 / stats.prefill_secs.max(1e-9),
        n,
        stats.decode_secs,
        n as f64 / stats.decode_secs.max(1e-9),
    );

    Ok(())
}
