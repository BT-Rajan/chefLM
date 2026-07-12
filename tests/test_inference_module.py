import inspect

from chef import inference


def test_inference_module_imports_persona():
    # Regression check: inference.py + persona.py used to live at the repo
    # root (outside the chef package) with a `from .persona import`
    # relative import that could never resolve there. Both now live inside
    # chef/, so this import — and the wiring below — should work.
    assert hasattr(inference, "Persona")
    assert hasattr(inference, "NONE_PERSONA")


def test_chat_completion_accepts_persona_kwarg():
    params = inspect.signature(inference.ChefInference.chat_completion).parameters
    assert "persona" in params
    assert params["persona"].default is None


def test_chat_completion_accepts_lang_kwarg_defaulting_to_english():
    params = inspect.signature(inference.ChefInference.chat_completion).parameters
    assert "lang" in params
    assert params["lang"].default == "en"


def test_format_prompt_injects_matching_lang_tag():
    # _format_prompt doesn't need a loaded model/tokenizer, so call it
    # directly on the class rather than constructing a full ChefInference.
    format_prompt = inference.ChefInference._format_prompt
    messages = [{"role": "user", "content": "hi"}]

    en_prompt = format_prompt(None, messages, lang="en")
    assert en_prompt.endswith("<|im_start|>assistant\n<|lang_en|>")

    ar_prompt = format_prompt(None, messages, lang="ar")
    assert ar_prompt.endswith("<|im_start|>assistant\n<|lang_ar|>")

    # Same user turn either way — only the trailing tag should differ
    assert en_prompt.replace("<|lang_en|>", "") == ar_prompt.replace("<|lang_ar|>", "")
