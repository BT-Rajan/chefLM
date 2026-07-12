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


# ── Bilingual (English/Arabic) coverage ─────────────────────────────────

AR_GREETING_INPUTS = {"مرحبا", "أهلا", "هاي", "أهلا شيف", "مرحبا شيف",
                       "السلام عليكم", "صباح الخير", "مساء الخير"}
AR_FAREWELL_INPUTS = {"مع السلامة", "إلى اللقاء", "وداعا", "باي",
                       "أراك لاحقا", "اعتني بنفسك", "لقد انتهيت الآن"}


def test_samples_default_to_english_when_lang_is_absent():
    # Only the new Arabic samples set "lang" explicitly; the original 1059
    # English samples rely on data_utils.format_sample defaulting an absent
    # "lang" key to "en" — this locks in that every sample resolves to one
    # of the two known languages, not silently to something else.
    for s in _all_samples():
        assert s.get("lang", "en") in ("en", "ar")


def test_dataset_has_a_meaningful_arabic_set():
    samples = _all_samples()
    ar_samples = [s for s in samples if s.get("lang") == "ar"]
    ar_categories = {s["category"] for s in ar_samples}
    assert len(ar_samples) >= 100, (
        "expected a substantial initial Arabic set, not a token handful"
    )
    assert ar_categories == EXPECTED_CATEGORIES, (
        "Arabic samples should cover the same 14 categories as English"
    )


def test_arabic_greetings_use_the_welcome_script():
    samples = {s["input"]: s["output"] for s in _all_samples() if s.get("lang") == "ar"}
    for greeting in AR_GREETING_INPUTS:
        assert greeting in samples, f"missing Arabic greeting sample: {greeting!r}"
        output = samples[greeting]
        assert "فانكم" in output
        assert "ميلك شيك ميستري" in output
        assert "كيف يمكنني مساعدتكم اليوم" in output


def test_arabic_farewells_use_the_polite_sign_off():
    samples = {s["input"]: s["output"] for s in _all_samples() if s.get("lang") == "ar"}
    for farewell in AR_FAREWELL_INPUTS:
        assert farewell in samples, f"missing Arabic farewell sample: {farewell!r}"
        output = samples[farewell]
        assert "حافظ على سلامتك" in output
        assert "يوماً رائعاً" in output
        assert "اللقاء" in output


def test_arabic_samples_are_not_just_english_reencoded():
    # Cheap guard against accidentally shipping transliterated/Latin-script
    # "Arabic" samples: every Arabic-tagged input and output should contain
    # actual Arabic-script characters (U+0600-U+06FF), not Latin letters.
    import re
    arabic_re = re.compile(r"[\u0600-\u06FF]")
    latin_re = re.compile(r"[A-Za-z]")
    for s in _all_samples():
        if s.get("lang") != "ar":
            continue
        assert arabic_re.search(s["input"]), f"non-Arabic input: {s['input']!r}"
        assert arabic_re.search(s["output"]), f"non-Arabic output: {s['output']!r}"
        assert not latin_re.search(s["input"]), f"Latin characters in Arabic input: {s['input']!r}"
        assert not latin_re.search(s["output"]), f"Latin characters in Arabic output: {s['output']!r}"


def test_format_sample_embeds_matching_lang_tag():
    from chef.data_utils import format_sample
    en_sample = {"input": "hi", "output": "hello", "category": "banter", "lang": "en"}
    ar_sample = {"input": "مرحبا", "output": "أهلا", "category": "banter", "lang": "ar"}
    no_lang_sample = {"input": "hi", "output": "hello", "category": "banter"}

    assert "<|lang_en|>" in format_sample(en_sample)
    assert "<|lang_ar|>" in format_sample(ar_sample)
    assert "<|lang_ar|>" not in format_sample(en_sample)
    # Absent "lang" key defaults to English, same as the original samples
    assert "<|lang_en|>" in format_sample(no_lang_sample)
