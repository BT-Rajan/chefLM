"""Shared helpers for formatting {input, output, category} samples into the
on-disk training formats (chat-template text for the tokenizer/model, and
OpenAI-style messages for interop). Used by dataset generators, e.g.
milkshake_data.py.
"""

LANG_TAGS = {"en": "<|lang_en|>", "ar": "<|lang_ar|>"}


def format_sample(s):
    # The lang tag sits right after the assistant preamble and before the
    # actual reply text. Training the model to always follow "<|lang_ar|>"
    # with Arabic text (and "<|lang_en|>" with English text) means the
    # *output* language can be forced at inference time by feeding that tag
    # as the last token of the prompt — regardless of what script the
    # user's own message happened to be written in. That's what makes a
    # simple EN/AR toggle in the UI reliable, rather than hoping the model
    # infers the right language from the input alone.
    lang_tag = LANG_TAGS.get(s.get("lang", "en"), LANG_TAGS["en"])
    return (
        f"<|im_start|>user\n{s['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{lang_tag}{s['output']}<|im_end|>"
    )


def to_openai(s):
    return {"messages": [
        {"role": "user", "content": s["input"]},
        {"role": "assistant", "content": s["output"]},
    ]}
