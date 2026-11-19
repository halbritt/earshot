from earshot_bench.text import words, read_reference, score, align, leaked_words


def test_words_ignore_case_and_punctuation_and_split_hyphens():
    assert words("Clank-clank, they came; it's 3 o'clock—now.") == ["clank", "clank", "they", "came", "it's", "3", "o'clock", "now"]


def test_a_reference_without_markers_is_one_section():
    assert read_reference("One two.\nThree.") == [("all", ["one", "two", "three"])]


def test_region_markers_split_the_reference():
    text = "# region: quiet\none two\n# region: over\nthree\n"
    assert read_reference(text) == [("quiet", ["one", "two"]), ("over", ["three"])]


def test_a_perfect_transcript_has_no_errors():
    s = score([("all", "a b c d".split())], "a b c d".split())
    assert (s.correct, s.errors, s.wer) == (4, 0, 0.0)


def test_one_substitution_one_missing_and_one_extra_word_are_counted_separately():
    # reference: the cat sat on the mat  (6 words)
    # heard:     the cat sit on mat today -> sat->sit, "the" missing, "today" extra
    s = score([("all", "the cat sat on the mat".split())], "the cat sit on mat today".split())
    assert (s.correct, s.substituted, s.missing, s.extra) == (4, 1, 1, 1)
    assert s.wer == 3 / 6


def test_errors_are_charged_to_the_section_of_the_reference_word():
    ref = [("quiet", "one two three".split()), ("over", "four five six".split())]
    s = score(ref, "one two three four six".split())
    assert s.sections["quiet"]["missing"] == 0
    assert s.sections["over"]["missing"] == 1
    assert s.sections["over"]["reference_words"] == 3


def test_a_word_recognised_across_a_section_boundary_still_counts():
    ref = [("a", "one two".split()), ("b", "three four".split())]
    s = score(ref, "one two three four".split())
    assert s.errors == 0


def test_a_long_run_of_missing_words_is_reported_with_its_place():
    ref = [("a", "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9".split())]
    s = score(ref, "w0 w1 w7 w8 w9".split())
    assert s.missing_runs == [dict(section="a", first_word=2, words=5)]


def test_short_gaps_are_not_reported_as_runs():
    s = score([("a", "w0 w1 w2 w3 w4".split())], "w0 w3 w4".split())
    assert s.missing == 2 and s.missing_runs == []


def test_playback_words_in_the_transcript_are_counted_as_leaked():
    ref = [("all", "please open the door".split())]
    played = "the weather today is sunny with light wind".split()
    s = score(ref, "please open today is sunny the door".split(), playback=played)
    assert s.leaked == 3
    assert s.extra == 3


def test_reference_words_that_also_occur_in_the_playback_are_not_leaks():
    ref = [("all", "today is sunny outside".split())]
    s = score(ref, "today is sunny outside".split(), playback="today is sunny".split())
    assert s.leaked == 0


def test_two_stray_words_are_below_the_leak_threshold():
    assert leaked_words("is sunny".split(), "today is sunny".split()) == 0


def test_an_empty_transcript_misses_every_word():
    s = score([("all", "a b c".split())], [])
    assert (s.missing, s.wer) == (3, 1.0)


def test_alignment_covers_every_word_of_both_sides_once():
    ops = align("a b c".split(), "b c d".split())
    assert sorted(i for _, i, _ in ops if i is not None) == [0, 1, 2]
    assert sorted(j for _, _, j in ops if j is not None) == [0, 1, 2]
