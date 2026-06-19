use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::engine::constraint::LogitMask;

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

#[derive(Clone, Copy)]
enum Num {
    NeedInt,            // just saw '-'
    Zero,               // saw a leading 0
    Int,                // integer digits
    NeedFrac,           // saw '.', need a digit
    Frac,               // fractional digits
    NeedExpSignOrDigit, // saw e/E
    NeedExpDigit,       // saw e/E then sign
    Exp,                // exponent digits
}

fn num_complete(n: Num) -> bool {
    matches!(n, Num::Zero | Num::Int | Num::Frac | Num::Exp)
}

enum NumStep {
    Cont(Num),
    EndReprocess,
    Invalid,
}

fn num_feed(n: Num, c: char) -> NumStep {
    use Num::*;
    use NumStep::*;
    match n {
        NeedInt => match c {
            '0' => Cont(Zero),
            '1'..='9' => Cont(Int),
            _ => Invalid,
        },
        Zero => match c {
            '.' => Cont(NeedFrac),
            'e' | 'E' => Cont(NeedExpSignOrDigit),
            _ => EndReprocess,
        },
        Int => match c {
            '0'..='9' => Cont(Int),
            '.' => Cont(NeedFrac),
            'e' | 'E' => Cont(NeedExpSignOrDigit),
            _ => EndReprocess,
        },
        NeedFrac => match c {
            '0'..='9' => Cont(Frac),
            _ => Invalid,
        },
        Frac => match c {
            '0'..='9' => Cont(Frac),
            'e' | 'E' => Cont(NeedExpSignOrDigit),
            _ => EndReprocess,
        },
        NeedExpSignOrDigit => match c {
            '+' | '-' => Cont(NeedExpDigit),
            '0'..='9' => Cont(Exp),
            _ => Invalid,
        },
        NeedExpDigit => match c {
            '0'..='9' => Cont(Exp),
            _ => Invalid,
        },
        Exp => match c {
            '0'..='9' => Cont(Exp),
            _ => EndReprocess,
        },
    }
}

#[derive(Clone)]
enum St {
    Start,
    Done,
    ExpectValue,
    ArrStart,
    ObjStart,
    ObjKey,
    Str { esc: bool, key: bool, hex: u8 },
    ExpectColon,
    Num(Num),
    Lit { lit: &'static str, i: usize },
    AfterObj,
    AfterArr,
}

#[derive(Clone)]
pub struct JsonState {
    stack: Vec<bool>,
    st: St,
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonState {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            st: St::Start,
        }
    }

    pub fn feed(&mut self, c: char) -> bool {
        loop {
            match self.st {
                St::Str { esc, key, hex } => return self.feed_string(c, esc, key, hex),
                St::Lit { lit, i } => return self.feed_lit(c, lit, i),
                St::Num(n) => match num_feed(n, c) {
                    NumStep::Cont(n2) => {
                        self.st = St::Num(n2);
                        return true;
                    }
                    NumStep::Invalid => return false,
                    NumStep::EndReprocess => {
                        self.complete_value();
                        continue;
                    }
                },
                _ => {}
            }

            if is_ws(c) {
                return true;
            }

            return match self.st {
                St::Start | St::ExpectValue => self.begin_value(c),
                St::ArrStart => {
                    if c == ']' {
                        self.close_container();
                        true
                    } else {
                        self.begin_value(c)
                    }
                }
                St::ObjStart => match c {
                    '}' => {
                        self.close_container();
                        true
                    }
                    '"' => {
                        self.st = St::Str {
                            esc: false,
                            key: true,
                            hex: 0,
                        };
                        true
                    }
                    _ => false,
                },
                St::ObjKey => {
                    if c == '"' {
                        self.st = St::Str {
                            esc: false,
                            key: true,
                            hex: 0,
                        };
                        true
                    } else {
                        false
                    }
                }
                St::ExpectColon => {
                    if c == ':' {
                        self.st = St::ExpectValue;
                        true
                    } else {
                        false
                    }
                }
                St::AfterObj => match c {
                    ',' => {
                        self.st = St::ObjKey;
                        true
                    }
                    '}' => {
                        self.close_container();
                        true
                    }
                    _ => false,
                },
                St::AfterArr => match c {
                    ',' => {
                        self.st = St::ExpectValue;
                        true
                    }
                    ']' => {
                        self.close_container();
                        true
                    }
                    _ => false,
                },
                St::Done => false,
                _ => false,
            };
        }
    }

    pub fn can_terminate(&self) -> bool {
        if !self.stack.is_empty() {
            return false;
        }
        match self.st {
            St::Done => true,
            St::Num(n) => num_complete(n),
            _ => false,
        }
    }

    pub fn allows(&self, s: &str) -> bool {
        let mut probe = self.clone();
        s.chars().all(|c| probe.feed(c))
    }

    fn begin_value(&mut self, c: char) -> bool {
        self.st = match c {
            '"' => St::Str {
                esc: false,
                key: false,
                hex: 0,
            },
            '{' => {
                self.stack.push(true);
                St::ObjStart
            }
            '[' => {
                self.stack.push(false);
                St::ArrStart
            }
            '-' => St::Num(Num::NeedInt),
            '0' => St::Num(Num::Zero),
            '1'..='9' => St::Num(Num::Int),
            't' => St::Lit { lit: "true", i: 1 },
            'f' => St::Lit { lit: "false", i: 1 },
            'n' => St::Lit { lit: "null", i: 1 },
            _ => return false,
        };
        true
    }

    fn complete_value(&mut self) {
        self.st = match self.stack.last() {
            None => St::Done,
            Some(true) => St::AfterObj,
            Some(false) => St::AfterArr,
        };
    }

    fn close_container(&mut self) {
        self.stack.pop();
        self.complete_value();
    }

    fn feed_string(&mut self, c: char, esc: bool, key: bool, hex: u8) -> bool {
        if hex > 0 {
            if c.is_ascii_hexdigit() {
                self.st = St::Str {
                    esc: false,
                    key,
                    hex: hex - 1,
                };
                true
            } else {
                false
            }
        } else if esc {
            match c {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                    self.st = St::Str {
                        esc: false,
                        key,
                        hex: 0,
                    };
                    true
                }
                'u' => {
                    self.st = St::Str {
                        esc: false,
                        key,
                        hex: 4,
                    };
                    true
                }
                _ => false,
            }
        } else {
            match c {
                '\\' => {
                    self.st = St::Str {
                        esc: true,
                        key,
                        hex: 0,
                    };
                    true
                }
                '"' => {
                    if key {
                        self.st = St::ExpectColon;
                    } else {
                        self.complete_value();
                    }
                    true
                }
                c if (c as u32) < 0x20 => false,
                _ => true,
            }
        }
    }

    fn feed_lit(&mut self, c: char, lit: &'static str, i: usize) -> bool {
        if lit.as_bytes().get(i).copied() != Some(c as u8) {
            return false;
        }
        let next = i + 1;
        if next == lit.len() {
            self.complete_value();
        } else {
            self.st = St::Lit { lit, i: next };
        }
        true
    }
}

pub struct JsonConstraint {
    state: JsonState,
    token_strings: Arc<Vec<String>>,
    eos: Vec<u32>,
}

impl JsonConstraint {
    pub fn new(token_strings: Arc<Vec<String>>, eos: Vec<u32>) -> Self {
        Self {
            state: JsonState::new(),
            token_strings,
            eos,
        }
    }
}

impl LogitMask for JsonConstraint {
    fn mask(&self, logits: &Tensor) -> Result<Tensor> {
        let vocab = self.token_strings.len();
        let mut add = vec![f32::NEG_INFINITY; vocab];
        for (id, s) in self.token_strings.iter().enumerate() {
            if self.state.allows(s) {
                add[id] = 0.0;
            }
        }
        let stop = self.state.can_terminate();
        for &e in &self.eos {
            if let Some(slot) = add.get_mut(e as usize) {
                *slot = if stop { 0.0 } else { f32::NEG_INFINITY };
            }
        }
        // `logits` is the 1-D (vocab,) row for the current step.
        let mask = Tensor::from_vec(add, vocab, logits.device())?;
        logits.add(&mask)
    }

    fn accept(&mut self, token: u32) {
        if let Some(s) = self.token_strings.get(token as usize) {
            for c in s.chars() {
                let _ = self.state.feed(c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(s: &str) -> bool {
        let mut st = JsonState::new();
        s.chars().all(|c| st.feed(c)) && st.can_terminate()
    }

    fn prefix_ok(s: &str) -> bool {
        let mut st = JsonState::new();
        s.chars().all(|c| st.feed(c))
    }

    #[test]
    fn accepts_well_formed() {
        assert!(accepts(
            r#"{"name":"Ada","age":36,"tags":["a","b"],"ok":true}"#
        ));
        assert!(accepts(r#"[1,2,3]"#));
        assert!(accepts(r#""hello\nthere""#));
        assert!(accepts(r#"-12.5e3"#));
        assert!(accepts(r#"{ "k" : null }"#));
    }

    #[test]
    fn rejects_malformed() {
        assert!(!prefix_ok(r#"{"a":1,}"#)); // trailing comma
        assert!(!prefix_ok(r#"{'a':1}"#)); // single quotes
        assert!(!prefix_ok(r#"[1,,2]"#)); // empty element
        assert!(!prefix_ok(r#"01"#)); // leading zero
        assert!(!prefix_ok(r#"tru e"#)); // broken literal
    }

    #[test]
    fn partial_is_a_valid_prefix_but_not_terminable() {
        assert!(prefix_ok(r#"{"name":"#));
        assert!(!accepts(r#"{"name":"#)); // incomplete
        assert!(prefix_ok(r#"{"n"#));
        assert!(prefix_ok(r#"-12."#)); // valid prefix, needs a digit
        assert!(!accepts(r#"-12."#));
    }
}
