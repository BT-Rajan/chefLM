"""
guardrails.py — checks that keep ChefLM's replies on-topic and well-formed.

ChefLM is a small transformer trained on ~1200 hand-written milkshake Q&A
samples (see milkshake_data.py). It has no real language understanding —
on inputs far from anything it was trained on, it doesn't "know it
doesn't know"; it just samples tokens, which can produce fluent-looking
but nonsensical or off-topic text.

This module adds two independent layers on top of generation:

1. retrieval + topic gate (pre-generation): every incoming message is
   compared against every training question using TF-IDF cosine
   similarity — a proper weighted vector-space comparison, not literal
   word overlap. The nearest training question decides what happens:
     - close match, non-redirect category -> return that question's
       stored answer directly, sidestepping generation entirely.
     - close match, redirect category -> return the canned fallback,
       skipping generation for a known-off-topic message.
     - no close match either way -> fall through to generation, letting
       the model take a shot at genuinely novel on-topic phrasing.

   Why TF-IDF cosine similarity instead of exact/Jaccard word overlap
   (this module's previous approach): word overlap requires every
   phrasing of every question to be hand-added to milkshake_data.py to
   be recognized — "list ingredients" not matching "what ingredients go
   in a milkshake" is exactly this failure, and fixing it one phrasing
   at a time doesn't generalize to the next unseen phrasing. TF-IDF
   weighting automatically down-ranks generic words ("what", "is", "a")
   without a hand-maintained stopword list (kept below anyway, as an
   extra precision boost — see _STOPWORDS), and cosine similarity over
   that weighted space recognizes a paraphrase as long as it shares
   enough *weighted* vocabulary with a training question, whether or
   not that exact phrasing was ever written down. This is what makes
   the retrieval layer generalize instead of requiring one dataset edit
   per new phrasing.

2. output sanitizer (post-generation, for whatever generation produces):
   catch degenerate output — empty replies, leaked special tokens
   (<|im_end|> etc. that weren't fully stripped), or the model looping
   on one word — and swap in the same fallback instead of returning it
   to the user.

Both reuse the existing training data (nothing new to hand-label), and
add one real dependency: scikit-learn (TfidfVectorizer, cosine_similarity)
— see requirements.txt.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .milkshake_data import SAMPLES

_WORD_RE = re.compile(r"[a-zA-Z\u0600-\u06FF]+")

# Question words and other high-frequency function words. TF-IDF already
# down-weights these automatically (they appear in nearly every training
# question, so they get a low idf score on their own) — this stopword
# list is a belt-and-suspenders precision boost on top of that, not the
# only thing standing between "milkshake" and "weather" the way it was
# under the old word-overlap scorer.
_STOPWORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "do", "does",
    "did", "you", "your", "i", "it", "this", "that", "to", "of", "in",
    "on", "for", "and", "or", "how", "about", "can", "could", "would",
    "should", "have", "has", "had", "be", "with", "if", "me", "my",
    "like", "good", "not", "at", "as", "so", "there", "any", "some",
    "today", "now", "tomorrow", "yesterday", "please", "really", "very",
    "just", "know", "think", "want", "tell", "get", "give", "got",
}


def _stem(word):
    """Very small, deliberately conservative suffix-stripping stemmer —
    not a real linguistic stemmer, just enough to collapse simple
    plural/singular variants (ingredient/ingredients, topping/toppings,
    flavor/flavors) onto the same TF-IDF feature, so the vectorizer
    doesn't treat them as two unrelated words with independent (and each
    individually weaker) idf weight. English/ASCII words only: Arabic
    tokens are returned unchanged, since this suffix logic doesn't apply
    to Arabic morphology and would misfire on it.

    Deliberately just the one rule (strip a single trailing "s") rather
    than a fuller stemmer — see git history for why a "-ing" rule was
    tried and reverted (it broke "topping" as a noun). If a specific
    missed match comes up later, extend this narrowly.
    """
    if not word.isascii() or len(word) <= 4:
        return word
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text):
    """Tokenizer fed to TfidfVectorizer (via the `tokenizer=` param, with
    `token_pattern=None` to disable sklearn's own regex splitting) so
    retrieval reuses the exact same word-extraction, stemming, and
    stopword-filtering as before — TF-IDF changes how those tokens get
    *weighted and compared*, not how they're produced."""
    return [
        stemmed
        for w in _WORD_RE.findall(text or "")
        for stemmed in [_stem(w.lower())]
        if stemmed not in _STOPWORDS
    ]


FALLBACK = {
    "en": "i only really know about milkshakes — ask me about flavors, "
          "ingredients, or how to make one.",
    "ar": "أنا فقط أعرف عن الميلك شيك — اسألني عن النكهات أو المكونات أو "
          "كيفية تحضيره.",
}


def _identity(x):
    return x


def _make_vectorizer():
    # A fresh TfidfVectorizer per language index (see _build_index) since
    # sklearn fits vocabulary/idf per instance — sharing one across
    # languages would mix English and Arabic vocabulary into one idf
    # weighting, which is meaningless (the two never compete against each
    # other for a match anyway, since lookups are always filtered to one
    # lang). token_pattern=None is required alongside a custom tokenizer,
    # otherwise sklearn warns it's being ignored.
    return TfidfVectorizer(
        tokenizer=_tokenize, preprocessor=_identity, token_pattern=None,
        lowercase=False, ngram_range=(1, 1), sublinear_tf=True,
    )


def _build_index(lang):
    """Build one TF-IDF index per language, once at import time, over
    EVERY training question (redirect included) — retrieval and topic
    gating are now one lookup instead of two separately-tuned mechanisms
    (see module docstring): whichever training question is the nearest
    neighbor, its category (redirect or not) decides the outcome."""
    docs, meta = [], []
    for s in SAMPLES:
        if s.get("lang", "en") != lang:
            continue
        docs.append(s["input"])
        meta.append((s["output"], s.get("category")))
    if not docs:
        return None
    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(docs)
    return {"vectorizer": vectorizer, "matrix": matrix, "meta": meta}


_INDEX = {lang: _build_index(lang) for lang in ("en", "ar")}


def _nearest(message, lang="en"):
    """Return (answer, category, similarity) for the single closest
    training question to `message` in `lang`'s index, or (None, None,
    0.0) if the index is empty or the message tokenizes to nothing this
    vocabulary has ever seen (cosine similarity to everything is 0)."""
    idx = _INDEX.get(lang) or _INDEX.get("en")
    if idx is None:
        return None, None, 0.0
    q_vec = idx["vectorizer"].transform([message])
    if q_vec.nnz == 0:
        return None, None, 0.0
    sims = cosine_similarity(q_vec, idx["matrix"])[0]
    best_i = sims.argmax()
    answer, category = idx["meta"][best_i]
    return answer, category, float(sims[best_i])


# Calibrated against tests/test_guardrails.py (see there for the concrete
# cases these were tuned against): TF-IDF cosine similarity on short,
# largely-content-word queries tends to land noticeably lower than
# Jaccard word-overlap did for a "real, close" match, and lower still for
# genuinely unrelated text (idf-weighted vectors of two short unrelated
# sentences rarely share much weighted mass at all) — so both thresholds
# are lower than the old 0.6, not directly comparable numbers.
RETRIEVAL_THRESHOLD = 0.6
TOPIC_THRESHOLD = 0.12


def topic_score(message, lang="en"):
    """Cosine similarity (0.0-1.0) of `message` to the closest training
    question, regardless of category. Higher means the message's
    (weighted) vocabulary looks more like something ChefLM was actually
    trained on, whether that's a genuine milkshake question or a known
    redirect example — see is_on_topic for how category then splits
    those two cases."""
    _answer, _category, score = _nearest(message, lang=lang)
    return score


def is_on_topic(message, lang="en", threshold=TOPIC_THRESHOLD):
    """A message is on-topic if its nearest training-question neighbor
    is close enough (>= threshold) AND that neighbor isn't itself a
    redirect example — i.e. the message doesn't just resemble *a*
    training question, it resembles a genuine milkshake/banter one."""
    _answer, category, score = _nearest(message, lang=lang)
    if score < threshold:
        return False
    return category != "redirect"


def best_match(message, lang="en", min_similarity=RETRIEVAL_THRESHOLD):
    """Find the closest training question to `message` by TF-IDF cosine
    similarity, and return its stored answer if it's close enough to
    trust AND it isn't a redirect example (redirect "answers" are canned
    deflections, not real content to retrieve — is_on_topic/FALLBACK
    handles that case instead).

    Returns (answer, similarity) on a hit, or (None, best_similarity_seen)
    on a miss — the second value is useful for logging/tuning even when
    there's no match to return.
    """
    answer, category, score = _nearest(message, lang=lang)
    if score >= min_similarity and category != "redirect":
        return answer, score
    return None, score


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
