from chef.guardrails import best_match, is_on_topic, looks_degenerate, topic_score


def test_real_milkshake_question_is_on_topic():
    assert is_on_topic("what is your favorite milkshake flavor")
    assert is_on_topic("how do i make a chocolate milkshake")
    assert is_on_topic("what ingredients go in a milkshake")


def test_greeting_is_on_topic():
    assert is_on_topic("hi there")
    assert is_on_topic("thanks")


def test_clearly_unrelated_question_is_off_topic():
    assert not is_on_topic("what is the weather today")
    assert not is_on_topic("who is the president of the united states")
    assert not is_on_topic("help me write some python code")
    assert not is_on_topic("tell me about basketball")


def test_generic_words_shared_with_redirect_examples_dont_count():
    # "today" appears in both banter ("what should i do today") and
    # redirect ("what's the news today") training questions. Under the
    # TF-IDF/cosine-similarity scorer, topic_score is now "similarity to
    # the single nearest training question, whatever its category" (see
    # guardrails.py) rather than a raw domain-word count, so a message
    # that's actually closest to a *redirect* example correctly scores
    # HIGH (it's a strong match — just to the wrong category), not zero.
    # is_on_topic is what actually enforces "don't treat this as a real
    # milkshake question" by checking that category, and that's the
    # behavior this test is really guarding.
    assert topic_score("what should i wear today") > 0
    assert not is_on_topic("what should i wear today")


def test_empty_output_is_degenerate():
    assert looks_degenerate("")
    assert looks_degenerate("   ")


def test_repeated_word_loop_is_degenerate():
    assert looks_degenerate("milkshake milkshake milkshake milkshake milkshake")


def test_leaked_special_tokens_are_degenerate():
    assert looks_degenerate("leftover <|im_start|> token")


def test_normal_reply_is_not_degenerate():
    assert not looks_degenerate("chocolate is a great flavor for a milkshake")


def test_exact_training_question_retrieves_stored_answer():
    answer, score = best_match("what is your favorite milkshake flavor")
    assert answer == "chocolate. it always wins."
    assert score == 1.0


def test_close_paraphrase_still_retrieves():
    answer, score = best_match("whats your favorite milkshake flavor")
    assert answer == "chocolate. it always wins."
    assert score >= 0.6


def test_loosely_related_question_does_not_retrieve():
    answer, score = best_match("how do i make a really good chocolate milkshake at home")
    assert answer is None
    assert score < 0.6


def test_off_topic_question_does_not_retrieve():
    answer, _score = best_match("what is the weather today")
    assert answer is None
