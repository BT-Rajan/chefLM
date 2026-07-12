"""
guardrails.py — lightweight, dependency-free checks that keep ChefLM's
replies on-topic and well-formed.

ChefLM is a small transformer trained on ~1200 hand-written milkshake Q&A
samples (see milkshake_data.py). It has no real language understanding —
on inputs far from anything it was trained on, it doesn't "know it
doesn't know"; it just samples tokens, which can produce fluent-looking
but nonsensical or off-topic text. There's already a "redirect" category
in the training data for known off-topic questions, but that only helps
for inputs that resemble those ~90 examples — anything novel can still
slip through as a wrong or made-up answer.

This module adds two independent layers on top of generation:

1. topic gate (pre-generation): compare the incoming message against the
   real (non-redirect) training questions. If it isn't close to anything
   ChefLM actually learned to answer, skip generation entirely and
   return a canned redirect — cheaper and more reliable than hoping the
   model redirects itself.

2. output sanitizer (post-generation): catch degenerate generation —
   empty replies, leaked special tokens (<|im_end|> etc. that weren't
   fully stripped), or the model looping on one word — and swap in the
   same fallback instead of returning it to the user.

Both are plain stdlib (re) and reuse the existing training data, so
there's nothing new to install or retrain.
"""

import re

from .milkshake_data import SAMPLES

_WORD_RE = re.compile(r"[a-zA-Z\u0600-\u06FF]+")

# Question words and other high-frequency function words. Nearly every
# training question contains several of these ("what is...", "do you
# like...", "how do i..."), so without filtering them out, an unrelated
# question built from the same common words (e.g. "what is the weather
# today") scores a misleadingly high overlap even though it shares no
# actual topic words with anything ChefLM was trained on. Stripping these
# leaves only the content words ("weather", "milkshake", "chocolate", ...)
# that the score should actually be based on.
_STOPWORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "do", "does",
    "did", "you", "your", "i", "it", "this", "that", "to", "of", "in",
    "on", "for", "and", "or", "how", "about", "can", "could", "would",
    "should", "have", "has", "had", "be", "with", "if", "me", "my",
    "like", "good", "not", "at", "as", "so", "there", "any", "some",
}


def _tokens(text):
    return set(w.lower() for w in _WORD_RE.findall(text or "")) - _STOPWORDS


# Discriminative domain vocabulary, built once at import time.
#
# A plain "does this message overlap with any training question" check
# doesn't work well on short text: generic words like "today" or "good"
# show up in BOTH real milkshake questions and the redirect examples
# ("what should i do today" vs. "what's the news today"), so they can't
# tell on-topic from off-topic on their own.
#
# Instead: collect the (stopword-filtered) words used in genuine
# milkshake/banter questions, then subtract any word that also shows up
# in a redirect question. What's left is vocabulary that's actually
# discriminative — "milkshake", "vanilla", "blend", "topping", "hi",
# "thanks" — words a real redirect example never uses. A message is
# on-topic if it contains at least one of these.
def _words_by_category(lang, categories=None, exclude_categories=None):
    words = set()
    for s in SAMPLES:
        if s.get("lang", "en") != lang:
            continue
        cat = s.get("category")
        if categories is not None and cat not in categories:
            continue
        if exclude_categories is not None and cat in exclude_categories:
            continue
        words |= _tokens(s["input"])
    return words


def _build_domain_vocab(lang):
    redirect_words = _words_by_category(lang, categories={"redirect"})
    onto_topic_words = _words_by_category(lang, exclude_categories={"redirect"})
    return onto_topic_words - redirect_words


_DOMAIN_VOCAB = {lang: _build_domain_vocab(lang) for lang in ("en", "ar")}

FALLBACK = {
    "en": "i only really know about milkshakes — ask me about flavors, "
          "ingredients, or how to make one.",
    "ar": "أنا فقط أعرف عن الميلك شيك — اسألني عن النكهات أو المكونات أو "
          "كيفية تحضيره.",
}


def topic_score(message, lang="en"):
    """Number of the message's words that fall in the discriminative
    domain vocabulary for `lang`. 0 means nothing in the message looks
    milkshake- or banter-related; the higher it is, the more of the
    message is made of words that only show up in genuine (non-redirect)
    training questions."""
    msg_tokens = _tokens(message)
    vocab = _DOMAIN_VOCAB.get(lang, _DOMAIN_VOCAB["en"])
    return len(msg_tokens & vocab)


def is_on_topic(message, lang="en", threshold=1):
    """threshold is a word count, not a fraction: 1 means "at least one
    word in the message is domain vocabulary". Raise it to require a
    stronger signal (fewer false positives on borderline input, more
    real milkshake questions getting redirected); lower it (0) to
    disable the gate entirely."""
    return topic_score(message, lang=lang) >= threshold


_MAX_REPEAT_RUN = 4  # same word 4+ times in a row -> looping


def looks_degenerate(text):
    """Catch generation failures: empty output, leftover special tokens
    that truncation missed, or the model looping on one word/phrase."""
    if not text or not text.strip():
        return True
    if "<|" in text or "|>" in text:
        return True
    words = text.split()
    run = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            run += 1
            if run >= _MAX_REPEAT_RUN:
                return True
        else:
            run = 1
    return False
