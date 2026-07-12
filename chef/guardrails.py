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

This module adds three independent layers on top of generation:

1. retrieval lookup (pre-generation): if the incoming message closely
   matches a training question (by word overlap), return that question's
   stored answer directly instead of generating — guaranteed correct and
   on-topic, since it sidesteps sampling entirely for anything the model
   was explicitly taught.

2. topic gate (pre-generation, for anything retrieval didn't catch):
   compare the message against the real (non-redirect) training
   questions' vocabulary. If it isn't close to anything ChefLM actually
   learned to answer, skip generation entirely and return a canned
   redirect — cheaper and more reliable than hoping the model redirects
   itself.

3. output sanitizer (post-generation, for whatever's left): catch
   degenerate generation — empty replies, leaked special tokens
   (<|im_end|> etc. that weren't fully stripped), or the model looping on
   one word — and swap in the same fallback instead of returning it to
   the user.

All three are plain stdlib (re) and reuse the existing training data, so
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
#
# Includes a second group of generic/temporal filler words ("today",
# "now", "please", "really", ...) added after finding that "today" alone
# — present in both a redirect example ("what's the news today") and a
# banter example ("what should i do today") — was enough to push an
# unrelated message ("what is the weather today") over best_match's
# similarity threshold via pure word overlap, with no actual topical
# relevance. These carry no domain signal in either direction, so they're
# excluded the same way question words are.
_STOPWORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "do", "does",
    "did", "you", "your", "i", "it", "this", "that", "to", "of", "in",
    "on", "for", "and", "or", "how", "about", "can", "could", "would",
    "should", "have", "has", "had", "be", "with", "if", "me", "my",
    "like", "good", "not", "at", "as", "so", "there", "any", "some",
    # generic/temporal filler — see comment above
    "today", "now", "tomorrow", "yesterday", "please", "really", "very",
    "just", "know", "think", "want", "tell", "get",
}


def _stem(word):
    """Very small, deliberately conservative suffix-stripping stemmer —
    not a real linguistic stemmer, just enough to collapse simple
    plural/singular variants (ingredient/ingredients, topping/toppings,
    flavor/flavors) so retrieval doesn't miss a match purely because of
    a trailing "s". English/ASCII words only: Arabic tokens are returned
    unchanged, since this suffix logic doesn't apply to Arabic morphology
    and would misfire on it.

    Deliberately just the one rule rather than a fuller stemmer: an
    earlier version also stripped "-ing", but that broke noun forms that
    happen to end in it within this domain — "topping" (the noun) was
    stemmed to "topp", which then *stopped* matching its own plural
    "toppings" (which only loses the trailing "s", not "-ing", since it
    ends in "-ings" rather than "-ing"). Singular and plural need to land
    on the same stem, and only stripping "s" guarantees that; stripping
    "-ing" too made it inconsistent depending on which form the word
    happened to start as. If a specific missed match comes up later,
    extend this narrowly rather than swapping in a general-purpose
    stemmer.
    """
    if not word.isascii() or len(word) <= 4:
        return word
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text):
    return set(_stem(w.lower()) for w in _WORD_RE.findall(text or "")) - _STOPWORDS


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


# Retrieval reference: every non-redirect training question, tokenized
# once at import time, alongside its stored answer. "redirect" examples
# are excluded here too — they're handled by is_on_topic, and their
# "answers" are canned deflections, not real content to retrieve.
_ANSWER_REFERENCE = [
    (_tokens(s["input"]), s["output"], s.get("lang", "en"))
    for s in SAMPLES
    if s.get("category") != "redirect"
]


def best_match(message, lang="en", min_similarity=0.6):
    """Find the closest training question to `message` by Jaccard
    similarity over stopword-filtered words, and return its stored
    answer if it's close enough to trust.

    This is a literal-overlap lookup, not a paraphrase or semantic
    matcher — it only fires when the wording is genuinely close to
    something in milkshake_data.py. That's the point: for those inputs,
    returning the human-written answer directly is strictly more
    reliable than generating, since it sidesteps sampling entirely.
    Anything phrased differently enough to miss the threshold falls
    through to generation as before.

    Returns (answer, similarity) on a hit, or (None, best_similarity_seen)
    on a miss — the second value is useful for logging/tuning even when
    there's no match to return.
    """
    msg_tokens = _tokens(message)
    if not msg_tokens:
        return None, 0.0
    best_score = 0.0
    best_answer = None
    for ref_tokens, ref_output, ref_lang in _ANSWER_REFERENCE:
        if ref_lang != lang or not ref_tokens:
            continue
        overlap = msg_tokens & ref_tokens
        if not overlap:
            continue
        score = len(overlap) / len(msg_tokens | ref_tokens)
        if score > best_score:
            best_score = score
            best_answer = ref_output
            if best_score == 1.0:
                break
    if best_score >= min_similarity:
        return best_answer, best_score
    return None, best_score


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
