const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

pub enum Chunk {
    Text(String),
    Reasoning(String),
}

#[derive(Default)]
pub struct ReasoningSplitter {
    in_think: bool,
    pending: String,
    pub reasoning: String,
    pub text: String,
}

impl ReasoningSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, piece: &str) -> Vec<Chunk> {
        self.pending.push_str(piece);
        let mut out = Vec::new();

        loop {
            let (marker, emit_reasoning) = if self.in_think {
                (CLOSE, true)
            } else {
                (OPEN, false)
            };

            if let Some(i) = self.pending.find(marker) {
                let before: String = self.pending.drain(..i).collect();
                self.pending.drain(..marker.len());
                self.in_think = !self.in_think;
                if !before.is_empty() {
                    out.push(self.record(before, emit_reasoning));
                }
                continue;
            }

            let keep = super::partial_suffix_len(&self.pending, marker);
            let emit_len = self.pending.len() - keep;
            if emit_len > 0 {
                let s: String = self.pending.drain(..emit_len).collect();
                out.push(self.record(s, emit_reasoning));
            }
            break;
        }

        out
    }

    pub fn flush(&mut self) -> Vec<Chunk> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let s = std::mem::take(&mut self.pending);
        vec![self.record(s, self.in_think)]
    }

    fn record(&mut self, s: String, reasoning: bool) -> Chunk {
        if reasoning {
            self.reasoning.push_str(&s);
            Chunk::Reasoning(s)
        } else {
            self.text.push_str(&s);
            Chunk::Text(s)
        }
    }
}

pub fn split(text: &str) -> (String, String) {
    let mut s = ReasoningSplitter::new();
    let _ = s.push(text);
    let _ = s.flush();
    (s.reasoning, s.text)
}
