//! Llama 3.2 sometimes spells a tool call as a Python dict literal
//! (`{'name': 'add', 'parameters': {'a': 5}}`). Python's `Llama32Detector`
//! reads that as a call; Dynamo's `llama3_json` parser reads only JSON, so the
//! call is respelled as JSON before that parser sees it.

use std::collections::HashMap;

use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_protocols::types::{ChatCompletionMessageContent, CreateChatCompletionStreamResponse};
use futures::{Stream, StreamExt};
use serde_json::Value;

const BOT_TOKEN: &str = "<|python_tag|>";

/// What the text an output opens with turned out to be.
enum Opening {
    /// A Python dict call: its JSON spelling, and the bytes of `body` it spans.
    Call { json: String, end: usize },
    /// More text decides.
    Undecided,
    /// Anything else, JSON calls included; it passes as the model wrote it.
    Other,
}

/// Split the optional `<|python_tag|>` off an output; `None` while only part
/// of the token has arrived.
fn split_bot_token(text: &str) -> Option<(&str, &str)> {
    if let Some(body) = text.strip_prefix(BOT_TOKEN) {
        Some((BOT_TOKEN, body))
    } else if BOT_TOKEN.starts_with(text) {
        None
    } else {
        Some(("", text))
    }
}

/// Classify `body`, the output after any `<|python_tag|>`. Only `{` followed
/// by a single quote is scanned, which no JSON object opens with.
fn opening(body: &str) -> Opening {
    let Some(rest) = body.strip_prefix('{') else {
        return if body.is_empty() {
            Opening::Undecided
        } else {
            Opening::Other
        };
    };
    match rest.trim_start().chars().next() {
        None => Opening::Undecided,
        Some('\'') => python_dict_call(body),
        Some(_) => Opening::Other,
    }
}

/// Respell the Python dict literal `body` opens with as JSON: single-quoted
/// strings become JSON strings, and `True`/`False`/`None` their JSON names.
/// It is a call only if the result is a JSON object with a string `name`.
fn python_dict_call(body: &str) -> Opening {
    let mut json = String::with_capacity(body.len());
    let mut depth = 0usize;
    let mut chars = body.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        match c {
            '\'' | '"' => {
                json.push('"');
                loop {
                    match chars.next() {
                        None => return Opening::Undecided,
                        Some((_, '\\')) => match chars.next() {
                            None => return Opening::Undecided,
                            // JSON has no `\'` escape.
                            Some((_, '\'')) => json.push('\''),
                            Some((_, escaped)) => json.extend(['\\', escaped]),
                        },
                        Some((_, quote)) if quote == c => break,
                        Some((_, '"')) => json.push_str("\\\""),
                        Some((_, other)) => json.push(other),
                    }
                }
                json.push('"');
            }
            '{' | '[' => {
                depth += 1;
                json.push(c);
            }
            '}' | ']' => {
                json.push(c);
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let is_call = matches!(
                        serde_json::from_str(&json),
                        Ok(Value::Object(call)) if call.get("name").is_some_and(Value::is_string)
                    );
                    return if is_call {
                        Opening::Call {
                            json,
                            end: at + c.len_utf8(),
                        }
                    } else {
                        Opening::Other
                    };
                }
            }
            c if c.is_ascii_alphabetic() => {
                let mut word = String::from(c);
                while let Some((_, next)) = chars.next_if(|(_, next)| next.is_ascii_alphanumeric())
                {
                    word.push(next);
                }
                json.push_str(match word.as_str() {
                    "True" => "true",
                    "False" => "false",
                    "None" => "null",
                    other => other,
                });
            }
            _ => json.push(c),
        }
    }
    Opening::Undecided
}

/// The text to forward for the output so far, or `None` while its opening is
/// undecided. `last` marks the whole output, which nothing more can decide.
fn decided(text: &str, last: bool) -> Option<String> {
    match split_bot_token(text).map(|(bot_token, body)| (bot_token, body, opening(body))) {
        Some((bot_token, body, Opening::Call { json, end })) => {
            Some(format!("{bot_token}{json}{}", &body[end..]))
        }
        Some((_, _, Opening::Other)) => Some(text.to_owned()),
        _ => last.then(|| text.to_owned()),
    }
}

/// `text` with the Python dict call it opens with respelled as JSON.
pub(super) fn respell_python_dict_call(text: &str) -> String {
    decided(text, true).unwrap_or_else(|| text.to_owned())
}

/// [`respell_python_dict_call`] over a stream. A choice's opening is held back
/// until it is decided, then released in one delta: respelled if it is a
/// Python dict call, else exactly as the model wrote it.
pub(super) fn respell_python_dict_calls<S>(
    stream: S,
) -> impl Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send
where
    S: Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send + 'static,
{
    async_stream::stream! {
        // The held opening per choice; `None` once it has been released.
        let mut held: HashMap<u32, Option<String>> = HashMap::new();
        futures::pin_mut!(stream);
        while let Some(mut item) = stream.next().await {
            for choice in item.data.iter_mut().flat_map(|chunk| chunk.choices.iter_mut()) {
                let opening = held.entry(choice.index).or_insert_with(|| Some(String::new()));
                let Some(text) = opening else {
                    continue;
                };
                if let Some(ChatCompletionMessageContent::Text(delta)) = &choice.delta.content {
                    text.push_str(delta);
                    choice.delta.content = None;
                }
                if let Some(released) = decided(text, choice.finish_reason.is_some()) {
                    *opening = None;
                    choice.delta.content = (!released.is_empty())
                        .then_some(ChatCompletionMessageContent::Text(released));
                }
            }
            yield item;
        }
    }
}

#[cfg(test)]
mod tests {
    use dynamo_parsers::ToolDefinition;
    use dynamo_protocols::types::{
        ChatChoiceStream, ChatCompletionStreamResponseDelta, ChatCompletionToolChoiceOption,
        FinishReason, Role,
    };
    use serde_json::json;

    use super::*;
    use crate::parser::tools::{parse_tool_calls, tool_call_stream};

    #[allow(deprecated)]
    fn chunk(
        text: Option<&str>,
        finish_reason: Option<FinishReason>,
    ) -> Annotated<CreateChatCompletionStreamResponse> {
        Annotated {
            data: Some(CreateChatCompletionStreamResponse {
                id: "chatcmpl-test".into(),
                choices: vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: text.map(|text| ChatCompletionMessageContent::Text(text.into())),
                        function_call: None,
                        tool_calls: None,
                        role: Some(Role::Assistant),
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason,
                    logprobs: None,
                }],
                created: 1,
                model: "model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            }),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn add_tool() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "add".into(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            })),
            strict: Some(true),
        }]
    }

    /// The stream's `(tool name, joined arguments, joined content, finish reason)`.
    fn stream(chunks: &[&str]) -> (Option<String>, String, String, Option<FinishReason>) {
        let mut items: Vec<_> = chunks.iter().map(|text| chunk(Some(text), None)).collect();
        items.push(chunk(None, Some(FinishReason::Stop)));
        let parsed: Vec<_> = futures::executor::block_on(
            tool_call_stream(
                "llama3",
                Some(ChatCompletionToolChoiceOption::Auto),
                Some(add_tool()),
                true,
                futures::stream::iter(items),
            )
            .collect(),
        );
        let (mut name, mut arguments, mut content, mut finish_reason) =
            (None, String::new(), String::new(), None);
        for choice in parsed
            .iter()
            .flat_map(|item| &item.data.as_ref().unwrap().choices)
        {
            if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                content.push_str(text);
            }
            for function in choice
                .delta
                .tool_calls
                .iter()
                .flatten()
                .flat_map(|call| &call.function)
            {
                name = function.name.clone().or(name);
                arguments.push_str(function.arguments.as_deref().unwrap_or_default());
            }
            finish_reason = choice.finish_reason.or(finish_reason);
        }
        (name, arguments, content, finish_reason)
    }

    fn unary_names(text: &str) -> Vec<String> {
        let (calls, _) =
            futures::executor::block_on(parse_tool_calls("llama3", text, Some(&add_tool())))
                .unwrap();
        calls.into_iter().map(|call| call.function.name).collect()
    }

    /// A call the model spells as a Python dict is a tool call, as it is on
    /// the Python server, not content with finish reason `stop`.
    #[test]
    fn python_dict_call_is_a_tool_call() {
        // Llama-3.2-1B's tokens for this output.
        let tokens = [
            "{'",
            "name",
            "':",
            " '",
            "add",
            "',",
            " '",
            "parameters",
            "':",
            " {'",
            "a",
            "':",
            " ",
            "5",
            ",",
            " '",
            "b",
            "':",
            " ",
            "7",
            "}}",
        ];
        for bot_token in [None, Some(BOT_TOKEN)] {
            let chunks: Vec<&str> = bot_token.into_iter().chain(tokens).collect();
            let (name, arguments, content, finish_reason) = stream(&chunks);
            assert_eq!(name.as_deref(), Some("add"));
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                json!({"a": 5, "b": 7})
            );
            assert_eq!(content, "");
            assert_eq!(finish_reason, Some(FinishReason::ToolCalls));
            assert_eq!(unary_names(&chunks.concat()), ["add"]);
        }
    }

    /// A Python dict that is not a call reaches the client as the model wrote
    /// it: held text is released unchanged, never respelled.
    #[test]
    fn python_dict_that_is_not_a_call_stays_content() {
        let chunks = [
            "{'",
            "type",
            "': '",
            "function",
            "', 'function': 'add'}",
            " ok",
        ];
        let (name, _, content, finish_reason) = stream(&chunks);
        assert_eq!(name, None);
        assert_eq!(content, chunks.concat());
        assert_eq!(finish_reason, Some(FinishReason::Stop));
        assert!(unary_names(&chunks.concat()).is_empty());

        // Never closed: released when the stream ends.
        let (name, _, content, _) = stream(&["{'", "name", "': 'add"]);
        assert_eq!((name, content.as_str()), (None, "{'name': 'add"));
    }

    #[test]
    fn respells_quotes_escapes_and_literals() {
        let python = concat!(
            r#"<|python_tag|>{'name': 'say', 'parameters': "#,
            r#"{'text': 'it\'s "ok"', "other": "don't", 'flags': [True, False, None]}} tail"#,
        );
        let json = concat!(
            r#"<|python_tag|>{"name": "say", "parameters": "#,
            r#"{"text": "it's \"ok\"", "other": "don't", "flags": [true, false, null]}} tail"#,
        );
        assert_eq!(respell_python_dict_call(python), json);
        // JSON calls and prose are not touched.
        for text in [r#"{"name": 'add'}"#, "Paris is 'nice'", "<|python_"] {
            assert_eq!(respell_python_dict_call(text), text);
        }
    }
}
