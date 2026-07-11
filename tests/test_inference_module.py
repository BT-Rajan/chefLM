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
