/// <reference lib="webworker" />
//
// Runs the model off the main thread so generation (a synchronous wasm call)
// never freezes the UI and tokens can stream.

import init, { LocalChat } from "./pkg/chat_candle.js";
import wasmUrl from "./pkg/chat_candle_bg.wasm?url";

// Quantized weights + tokenizer, fetched from the HF CDN on first load and
// cached (Cache API) so later loads are instant.
const GGUF_URL =
  "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf";
const TOKENIZER_URL =
  "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json";
const CACHE = "chat-candle-v1";

let chat: LocalChat | null = null;

type ToMain =
  | { type: "status"; status: string }
  | { type: "progress"; label: string; received: number; total: number }
  | { type: "ready" }
  | { type: "token"; piece: string }
  | { type: "done"; tokens: number; secs: number }
  | { type: "error"; error: string };

const post = (m: ToMain) => (self as DedicatedWorkerGlobalScope).postMessage(m);

async function fetchBytes(url: string, label: string): Promise<Uint8Array> {
  const cache = await caches.open(CACHE);
  let resp = await cache.match(url);
  if (!resp) {
    const net = await fetch(url);
    if (!net.ok) throw new Error(`${label}: HTTP ${net.status}`);
    await cache.put(url, net.clone());
    resp = net;
  }
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body!.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    post({ type: "progress", label, received, total });
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      post({ type: "status", status: "initializing wasm" });
      await init(wasmUrl);
      const tok = await fetchBytes(TOKENIZER_URL, "tokenizer");
      const gguf = await fetchBytes(GGUF_URL, "weights");
      post({ type: "status", status: "building model" });
      chat = new LocalChat(gguf, tok);
      post({ type: "ready" });
    } else if (msg.type === "generate") {
      if (!chat) throw new Error("model not loaded");
      const t0 = performance.now();
      let n = 0;
      chat.generate(
        msg.prompt,
        undefined,
        msg.maxTokens ?? 256,
        msg.temperature ?? 0.7,
        (piece: string) => {
          n++;
          post({ type: "token", piece });
        },
      );
      post({ type: "done", tokens: n, secs: (performance.now() - t0) / 1000 });
    }
  } catch (err) {
    post({ type: "error", error: String(err) });
  }
};
