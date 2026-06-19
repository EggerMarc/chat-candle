import { useEffect, useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; text: string };
type Progress = Record<string, { received: number; total: number }>;

const mb = (n: number) => (n / 1024 / 1024).toFixed(0);

export function App() {
  const worker = useRef<Worker | null>(null);
  const [phase, setPhase] = useState("starting…");
  const [progress, setProgress] = useState<Progress>({});
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("What is the capital of France?");
  const [messages, setMessages] = useState<Msg[]>([]);

  useEffect(() => {
    const w = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    worker.current = w;
    w.onmessage = (e) => {
      const m = e.data;
      switch (m.type) {
        case "status":
          setPhase(m.status);
          break;
        case "progress":
          setProgress((p) => ({
            ...p,
            [m.label]: { received: m.received, total: m.total },
          }));
          setPhase(`downloading ${m.label}…`);
          break;
        case "ready":
          setReady(true);
          setPhase("ready");
          break;
        case "token":
          setMessages((ms) => {
            const upd = [...ms];
            const last = upd[upd.length - 1];
            upd[upd.length - 1] = { ...last, text: last.text + m.piece };
            return upd;
          });
          break;
        case "done":
          setBusy(false);
          setPhase(
            `ready · ${m.tokens} tok in ${m.secs.toFixed(1)}s (${(
              m.tokens / m.secs
            ).toFixed(1)} tok/s)`,
          );
          break;
        case "error":
          setBusy(false);
          setPhase("error: " + m.error);
          break;
      }
    };
    w.postMessage({ type: "load" });
    return () => w.terminate();
  }, []);

  const send = () => {
    const prompt = input.trim();
    if (!ready || busy || !prompt) return;
    setMessages((ms) => [
      ...ms,
      { role: "user", text: prompt },
      { role: "assistant", text: "" },
    ]);
    setInput("");
    setBusy(true);
    setPhase("generating…");
    worker.current!.postMessage({
      type: "generate",
      prompt,
      maxTokens: 256,
      temperature: 0.7,
    });
  };

  return (
    <main style={styles.main}>
      <h1 style={styles.h1}>chat-candle</h1>
      <p style={styles.muted}>
        Qwen3-0.6B (Q4_K_M GGUF) running locally via candle compiled to wasm.
        Weights auto-download from the HF CDN on first load and cache in the
        browser. CPU-only (no WebGPU yet) — so it's slow but real.
      </p>

      <div style={styles.status}>
        <strong>{phase}</strong>
        {Object.entries(progress).map(([label, p]) => (
          <div key={label} style={styles.bar}>
            <span>{label}</span>
            <progress value={p.received} max={p.total || undefined} />
            <span>
              {mb(p.received)}
              {p.total ? ` / ${mb(p.total)} MB` : " MB"}
            </span>
          </div>
        ))}
      </div>

      <div style={styles.chat}>
        {messages.map((m, i) => (
          <div key={i} style={m.role === "user" ? styles.user : styles.bot}>
            {m.text || (m.role === "assistant" ? "…" : "")}
          </div>
        ))}
      </div>

      <div style={styles.composer}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          placeholder={ready ? "Ask something…" : "loading model…"}
          style={styles.textarea}
        />
        <button onClick={send} disabled={!ready || busy} style={styles.button}>
          {busy ? "…" : "Send"}
        </button>
      </div>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  main: {
    font: "15px/1.5 system-ui, sans-serif",
    maxWidth: 720,
    margin: "2rem auto",
    padding: "0 1rem",
  },
  h1: { fontSize: "1.3rem", marginBottom: 0 },
  muted: { color: "#777", fontSize: "0.85rem" },
  status: {
    background: "#f4f4f4",
    borderRadius: 6,
    padding: "0.6rem 0.8rem",
    fontSize: "0.85rem",
    margin: "1rem 0",
  },
  bar: { display: "flex", gap: "0.6rem", alignItems: "center", marginTop: 4 },
  chat: { display: "flex", flexDirection: "column", gap: "0.6rem" },
  user: {
    alignSelf: "flex-end",
    background: "#2563eb",
    color: "#fff",
    padding: "0.5rem 0.8rem",
    borderRadius: 12,
    maxWidth: "80%",
    whiteSpace: "pre-wrap",
  },
  bot: {
    alignSelf: "flex-start",
    background: "#f0f0f0",
    padding: "0.5rem 0.8rem",
    borderRadius: 12,
    maxWidth: "80%",
    whiteSpace: "pre-wrap",
  },
  composer: { display: "flex", gap: "0.5rem", marginTop: "1rem" },
  textarea: { flex: 1, padding: "0.5rem", boxSizing: "border-box" },
  button: { padding: "0.5rem 1.2rem" },
};
