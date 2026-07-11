from chef.grammar import correct_grammar


def test_empty_text_returned_unchanged():
    assert correct_grammar("") == ""
    assert correct_grammar(None) is None


def test_correction_never_raises_without_languagetool_installed():
    # In this test environment language-tool-python is a dev-only extra
    # (see requirements-dev.txt), so this also exercises the fail-soft
    # path described in grammar.py's module docstring.
    text = "this are a test sentance"
    result = correct_grammar(text)
    assert isinstance(result, str)
