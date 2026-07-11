"""Grammar/sentence check applied to generated replies before they're shown.

Uses LanguageTool (via the `language_tool_python` package) for real grammar
correction — catches things like subject-verb agreement, capitalization,
double spaces, missing punctuation, etc. LanguageTool itself is a Java
program; `language_tool_python` downloads it automatically the first time
it runs (~200MB) and needs a JRE installed on the machine.

This is intentionally isolated in its own module and fails soft: if Java
isn't installed, or the first-run download can't reach the network, the
rest of the model still works — replies just won't be grammar-corrected,
with a one-time warning instead of a crash.
"""

_tool = None
_load_failed = False


def _get_tool():
    """Lazily create the LanguageTool instance (slow: ~1-2s first call,
    plus a one-time ~200MB download + Java runtime check on first ever use
    on a machine)."""
    global _tool, _load_failed
    if _tool is not None or _load_failed:
        return _tool
    try:
        import language_tool_python
        _tool = language_tool_python.LanguageTool("en-US")
    except Exception as e:
        _load_failed = True
        print(f"[grammar] LanguageTool unavailable ({e}); "
              f"continuing without grammar correction.")
        _tool = None
    return _tool


def correct_grammar(text):
    """Run grammar correction on `text`. Returns the corrected string, or
    the original string unchanged if LanguageTool isn't available."""
    if not text:
        return text
    tool = _get_tool()
    if tool is None:
        return text
    try:
        return tool.correct(text)
    except Exception as e:
        print(f"[grammar] correction failed ({e}); returning original text.")
        return text
