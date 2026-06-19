use minijinja::{Environment, Error, ErrorKind, context};

pub struct Turn {
    pub role: &'static str,
    pub content: String,
}

pub struct ChatTemplate {
    template: Option<String>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn new(template: Option<String>, bos_token: String, eos_token: String) -> Self {
        Self {
            template,
            bos_token,
            eos_token,
        }
    }

    pub fn chatml_only() -> Self {
        Self {
            template: None,
            bos_token: String::new(),
            eos_token: String::new(),
        }
    }

    pub fn render(&self, turns: &[Turn]) -> String {
        if let Some(tmpl) = &self.template
            && let Ok(rendered) = self.render_jinja(tmpl, turns)
        {
            return rendered;
        }
        chatml(turns)
    }

    fn render_jinja(&self, tmpl: &str, turns: &[Turn]) -> Result<String, Error> {
        let mut env = Environment::new();
        env.add_function("raise_exception", |msg: String| {
            Err::<String, _>(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |_fmt: String| Ok::<_, Error>(String::new()));

        let messages: Vec<serde_json::Value> = turns
            .iter()
            .map(|t| serde_json::json!({ "role": t.role, "content": t.content }))
            .collect();

        env.render_str(
            tmpl,
            context! {
                messages => messages,
                add_generation_prompt => true,
                bos_token => self.bos_token,
                eos_token => self.eos_token,
            },
        )
    }
}

pub fn chatml(turns: &[Turn]) -> String {
    let mut s = String::new();
    for t in turns {
        s.push_str("<|im_start|>");
        s.push_str(t.role);
        s.push('\n');
        s.push_str(&t.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_turn() -> Vec<Turn> {
        vec![Turn {
            role: "user",
            content: "hi".into(),
        }]
    }

    #[test]
    fn renders_model_jinja_template() {
        let tmpl = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}\
                    {% if add_generation_prompt %}[assistant]{% endif %}";
        let ct = ChatTemplate::new(Some(tmpl.into()), "<bos>".into(), "<eos>".into());
        assert_eq!(ct.render(&user_turn()), "[user] hi\n[assistant]");
    }

    #[test]
    fn falls_back_to_chatml_without_template() {
        let out = ChatTemplate::chatml_only().render(&user_turn());
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn falls_back_on_broken_template() {
        let ct = ChatTemplate::new(Some("{% for %}".into()), String::new(), String::new());
        assert!(ct.render(&user_turn()).contains("<|im_start|>"));
    }
}
