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

# Words that are intentional (persona flavor, loanwords, brand names,
# informal terms, etc.) but aren't in LanguageTool's English dictionary,
# so it would otherwise "correct" them into an unrelated real word — e.g.
# "vanakkam" (a Tamil greeting used in some training answers, see
# milkshake_data.py) getting rewritten to "Tanaka".
#
# This list was built by extracting every distinct word used across the
# English training outputs (see milkshake_data.py), running them through
# aspell to shortlist words not in a standard English dictionary, then
# actually running each shortlisted word through LanguageTool to confirm
# it gets mangled (rather than assuming aspell's gaps = LanguageTool's
# gaps — some words aspell flags, like "chai" or "oreo", LanguageTool
# handles fine and don't need protecting). Only words confirmed to
# actually change under LanguageTool are listed here:
#   - drink/dessert loanwords: affogato, cortado, granita, horchata (ok,
#     kept out — see note below), lassi, leche, matcha, paleta
#   - informal/compound terms: citrusy, creamsicle, eggier, thickshake,
#     thru, thrus (as in "drive-thru")
#   - a proper noun: boba, horlick (as in "William Horlick", credited
#     with inventing malted milk powder — LanguageTool capitalizes
#     "William" correctly on its own, only "Horlick" needed protecting)
#
# Checked case-insensitively; original casing in the text is preserved.
# Add to this set if a new protected word starts getting mangled — see
# the comment above _protect_words for how to re-run this audit.
PROTECTED_WORDS = {
    "vanakkam", "affogato", "boba", "citrusy", "cortado", "creamsicle",
    "eggier", "granita", "horlick", "lassi", "leche", "matcha", "paleta",
    "thickshake", "thru", "thrus",
}

# Placeholder substituted in for each protected word before running
# LanguageTool, then swapped back after. Alphabetic-only so LanguageTool
# has no digits/punctuation to flag, and distinctive enough to be very
# unlikely to collide with real reply text.
_PLACEHOLDER_TEMPLATE = "Xprotectedword{}X"


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


def _protect_words(text):
    """Replace whole-word, case-insensitive occurrences of PROTECTED_WORDS
    with placeholders, so LanguageTool can't "correct" them into a real
    but unrelated word. Returns (protected_text, restore_map), where
    restore_map maps each placeholder back to the original (as-cased)
    matched text."""
    if not PROTECTED_WORDS:
        return text, {}
    import re
    restore_map = {}
    counter = 0

    def repl(match):
        nonlocal counter
        placeholder = _PLACEHOLDER_TEMPLATE.format(counter)
        restore_map[placeholder] = match.group(0)
        counter += 1
        return placeholder

    pattern = r"\b(" + "|".join(re.escape(w) for w in PROTECTED_WORDS) + r")\b"
    protected_text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return protected_text, restore_map


def _restore_words(text, restore_map):
    for placeholder, original in restore_map.items():
        text = text.replace(placeholder, original)
    return text


def correct_grammar(text, lang="en"):
    """Run grammar correction on `text` in the given language ("en" or
    "ar"). Returns the corrected string, or the original string unchanged
    if LanguageTool isn't available for that language.

    Words in PROTECTED_WORDS (intentional persona vocabulary that isn't
    in LanguageTool's dictionary) are swapped out for placeholders before
    correction and restored after, so LanguageTool can't rewrite them into
    an unrelated real word.
    """
    if not text:
        return text
    tool = _get_tool(lang)
    if tool is None:
        return text
    protected_text, restore_map = _protect_words(text)
    try:
        corrected = tool.correct(protected_text)
    except Exception as e:
        print(f"[grammar] correction failed ({e}); returning original text.")
        return text
    return _restore_words(corrected, restore_map)
