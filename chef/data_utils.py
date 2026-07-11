"""Shared helpers for formatting {input, output, category} samples into the
on-disk training formats (chat-template text for the tokenizer/model, and
OpenAI-style messages for interop). Used by dataset generators, e.g.
milkshake_data.py.
"""


def format_sample(s):
    return (
        f"<|im_start|>user\n{s['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{s['output']}<|im_end|>"
    )


def to_openai(s):
    return {"messages": [
        {"role": "user", "content": s["input"]},
        {"role": "assistant", "content": s["output"]},
    ]}
