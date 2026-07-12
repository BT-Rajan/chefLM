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

Bilingual (English/Arabic): each language gets its own lazily-created
LanguageTool instance ("en-US" or "ar"), keyed in _tools by lang code, since
running English grammar rules against Arabic text (or vice versa) would
misapply rules rather than actually correct anything. If Arabic support
specifically fails to load (e.g. an older LanguageTool bundle without an
Arabic model), that language falls back to returning the text unchanged —
same fail-soft behavior as the rest of this module, just scoped per
language instead of globally.
"""

_LANGUAGE_TOOL_CODES = {"en": "en-US", "ar": "ar"}

_tools = {}
_load_failed = set()


def _get_tool(lang="en"):
    """Lazily create the LanguageTool instance for `lang` (slow: ~1-2s
    first call per language, plus a one-time ~200MB download + Java
    runtime check on first ever use on a machine)."""
    if lang in _tools:
        return _tools[lang]
    if lang in _load_failed:
        return None
    try:
        import language_tool_python
        tool_code = _LANGUAGE_TOOL_CODES.get(lang, _LANGUAGE_TOOL_CODES["en"])
        tool = language_tool_python.LanguageTool(tool_code)
        _tools[lang] = tool
        return tool
    except Exception as e:
        _load_failed.add(lang)
        print(f"[grammar] LanguageTool unavailable for lang={lang!r} ({e}); "
              f"continuing without grammar correction for this language.")
        return None


def correct_grammar(text, lang="en"):
    """Run grammar correction on `text` in the given language ("en" or
    "ar"). Returns the corrected string, or the original string unchanged
    if LanguageTool isn't available for that language."""
    if not text:
        return text
    tool = _get_tool(lang)
    if tool is None:
        return text
    try:
        return tool.correct(text)
    except Exception as e:
        print(f"[grammar] correction failed ({e}); returning original text.")
        return text
