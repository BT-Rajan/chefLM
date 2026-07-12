from chef.guardrails import is_on_topic, looks_degenerate, topic_score


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
    # redirect ("what's the news today") training questions, so on its
    # own it must not be treated as a domain signal.
    assert topic_score("what should i wear today") == 0
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
