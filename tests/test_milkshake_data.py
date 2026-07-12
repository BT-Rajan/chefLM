from chef import milkshake_data

EXPECTED_CATEGORIES = {
    "flavor", "ingredients", "howto", "topping", "temperature", "ordering",
    "health", "comparison", "opinion", "funfact", "recipe", "nutrition",
    "banter", "redirect",
}


def _all_samples():
    # milkshake_data.py exposes its samples under different names across
    # phases; grab whichever list-of-dicts attribute is present.
    for attr in ("SAMPLES", "DATA", "MILKSHAKE_DATA"):
        samples = getattr(milkshake_data, attr, None)
        if samples:
            return samples
    raise AttributeError(
        "Could not find the sample list in chef.milkshake_data; "
        "update this test's attribute name if the module was renamed."
    )


def test_dataset_has_expected_categories():
    samples = _all_samples()
    categories = {sample["category"] for sample in samples}
    assert categories == EXPECTED_CATEGORIES, (
        "README documents 14 topics — if this fails, either the dataset "
        "changed or README needs updating to match"
    )


def test_dataset_samples_have_input_and_output():
    samples = _all_samples()
    assert len(samples) > 0
    for sample in samples[:20]:
        assert sample["input"]
        assert sample["output"]


GREETING_INPUTS = {"hi", "hello", "hey", "hey chef", "hi chef", "hiya",
                    "greetings", "good morning", "good afternoon", "good evening"}
FAREWELL_INPUTS = {"bye", "goodbye", "see you later", "see you",
                    "take care", "i'll talk to you later", "i'm done for now"}


def test_greetings_use_the_welcome_script():
    samples = {s["input"]: s["output"] for s in _all_samples()}
    for greeting in GREETING_INPUTS:
        assert greeting in samples, f"missing greeting sample: {greeting!r}"
        output = samples[greeting]
        assert "vanakkam" in output
        assert "milkshake mystery" in output
        assert "how can i help you today" in output


def test_farewells_use_the_polite_sign_off():
    samples = {s["input"]: s["output"] for s in _all_samples()}
    for farewell in FAREWELL_INPUTS:
        assert farewell in samples, f"missing farewell sample: {farewell!r}"
        output = samples[farewell]
        assert "stay safe" in output
        assert "have a great day" in output
        assert "bye" in output
