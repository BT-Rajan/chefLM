"""
persona.py — output-side persona shaping.

Runs AFTER the model generates text and AFTER grammar.py cleans it up.
Does not touch model.py, train.py, milkshake_data.py, or the (nonexistent)
system prompt — see README "Why no system prompt?" for why that path
doesn't work at this model size. This is a small, deterministic
text-rewriting layer: it takes whatever the model already said and
adds Indian-English conversational flavor to it. No retraining required
to use or tweak this file.

It is intentionally light-touch:
- adds a tag word ("yaar", "no?", "only", ...) to some sentences
- occasionally prepends a conversational opener ("arre, ", "acha, ")
- swaps a small set of stock words for their Indian-English equivalents
It never invents new claims or changes the factual content of a reply.
"""

import random
import re

# Sentence-final tags. Kept short so they don't read as a non-sequitur
# tacked onto any sentence.
TAG_WORDS = ["yaar", "no?", "only", "actually", "boss", "na"]

# Conversational openers, applied to the first sentence of a reply only.
OPENERS = ["arre, ", "acha, ", "arre yaar, "]

# Small vocabulary swaps — applied whole-word, case-insensitive, and
# only when the replacement doesn't collide with a word already used
# elsewhere in the sentence (avoids "very very").
WORD_SWAPS = {
    "very": "very very",
    "really": "really only",
    "yes": "yes yes",
}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class Persona:
    """A configurable, swappable persona layer.

    Usage:
        persona = Persona(name="indian", intensity=0.4)
        styled_text = persona.apply(model_output_text)

    intensity: 0.0 (no change, passthrough) to 1.0 (tag on nearly
    every sentence). 0.3-0.5 reads as flavorful without becoming
    exhausting; keep it in that range for normal use.
    """

    def __init__(self, name="indian", intensity=0.4, seed=None):
        self.name = name
        self.intensity = max(0.0, min(1.0, intensity))
        self._rng = random.Random(seed)

    def apply(self, text):
        if self.name == "none" or not text or self.intensity == 0.0:
            return text
        if self.name != "indian":
            raise ValueError(f"Unknown persona: {self.name!r} (only 'indian' and 'none' exist so far)")

        sentences = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s]
        if not sentences:
            return text

        styled = []
        for i, sentence in enumerate(sentences):
            s = sentence
            if i == 0 and self._rng.random() < self.intensity * 0.5:
                s = self._rng.choice(OPENERS) + s[0].lower() + s[1:]
            if self._rng.random() < self.intensity:
                s = self._add_tag(s)
            styled.append(s)

        return " ".join(styled)

    def _add_tag(self, sentence):
        tag = self._rng.choice(TAG_WORDS)
        stripped = sentence.rstrip()
        if not stripped:
            return sentence
        end_punct = stripped[-1] if stripped[-1] in ".!?" else "."
        body = stripped[:-1] if stripped[-1] in ".!?" else stripped
        # Don't double up if the sentence already ends in the same tag.
        if body.lower().endswith(tag.lower()):
            return stripped if stripped[-1] in ".!?" else stripped + end_punct
        return f"{body}, {tag}{end_punct}"


def apply_word_swaps(text):
    """Optional extra flavor pass — small vocabulary substitutions.

    Kept separate from Persona.apply() so it can be toggled independently;
    repeating "very very" or "yes yes" is a strong, distinctive marker and
    some users will want it off even with the persona otherwise on.

    Idempotent: safe to call more than once on the same text (e.g. if a
    caller applies it, then later re-applies it to already-swapped text).
    Each swap's pattern refuses to match if the word is already followed
    by its own swapped continuation ("very very", "yes yes", "really
    only"), so a second call is a no-op instead of compounding into
    "very very very very". Verified via test_persona.py.
    """
    def repl(match):
        word = match.group(0)
        lower = word.lower()
        if lower in WORD_SWAPS:
            return WORD_SWAPS[lower]
        return word

    # Per-word negative lookahead/lookbehind: don't match `word` if it's
    # already adjacent to whatever continuation its own swap would add
    # (e.g. don't match "very" if already followed BY "very"; don't match
    # "really" if already followed by "only"). For swaps where the
    # continuation repeats the word itself ("very"->"very very", "yes"->
    # "yes yes"), a lookahead alone isn't enough — the newly-*inserted*
    # duplicate has nothing after it, so on a second call it would match
    # and compound ("very very very"). A lookbehind for "already preceded
    # by this same word" catches that second instance too. Swaps whose
    # continuation is a different word ("really"->"really only") don't
    # need the lookbehind since "only" alone was never a matchable key.
    parts = []
    for word, replacement in WORD_SWAPS.items():
        continuation = replacement[len(word):].strip()
        lookahead = rf"(?!\s+{re.escape(continuation)}\b)"
        if continuation.lower() == word.lower():
            lookbehind = rf"(?<!\b{re.escape(word)}\s)"
        else:
            lookbehind = ""
        parts.append(rf"{lookbehind}\b{re.escape(word)}\b{lookahead}")
    pattern = "|".join(parts)
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


NONE_PERSONA = Persona(name="none", intensity=0.0)
