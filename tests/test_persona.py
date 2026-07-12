from chef.persona import NONE_PERSONA, Persona, apply_word_swaps


def test_none_persona_is_passthrough():
    text = "Vanilla milkshakes are the best. They are very reliable."
    assert NONE_PERSONA.apply(text) == text


def test_zero_intensity_is_passthrough():
    persona = Persona(name="indian", intensity=0.0)
    text = "Vanilla milkshakes are the best."
    assert persona.apply(text) == text


def test_persona_apply_is_deterministic_with_seed():
    text = "Vanilla is a classic flavor. It pairs well with almost anything."
    a = Persona(name="indian", intensity=0.5, seed=7).apply(text)
    b = Persona(name="indian", intensity=0.5, seed=7).apply(text)
    assert a == b


def test_unknown_persona_name_raises():
    persona = Persona(name="robot", intensity=0.5)
    try:
        persona.apply("hello")
        assert False, "expected ValueError for unknown persona name"
    except ValueError:
        pass


def test_word_swaps_only_touches_known_words():
    text = "This is a really good, very cold milkshake. Yes indeed."
    swapped = apply_word_swaps(text)
    assert "really only" in swapped
    assert "very very" in swapped
    assert "yes yes" in swapped
    assert "milkshake" in swapped


def test_word_swaps_are_idempotent():
    # Calling apply_word_swaps more than once on the same text must not
    # compound the swap (e.g. "very" -> "very very" -> "very very very
    # very" on a second call). This can happen in practice if a caller
    # accidentally re-applies it to already-swapped text.
    text = "This is a really good, very cold milkshake. Yes indeed."
    once = apply_word_swaps(text)
    twice = apply_word_swaps(once)
    thrice = apply_word_swaps(twice)
    assert once == twice == thrice
