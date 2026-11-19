//! Words, alignment and scores. No audio here.

use serde::Serialize;

/// Lower-case words. Hyphens and dashes split words; other punctuation is removed;
/// apostrophes stay.
pub fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() || c == '\'' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// One named part of a reference text.
pub type Section = (String, Vec<String>);

/// Split a reference text into sections.
///
/// A line of the form `# region: NAME` starts a section. Text before the first such line,
/// or a text with no such lines, belongs to the section named "all".
pub fn read_reference(text: &str) -> Vec<Section> {
    fn marker(line: &str) -> Option<&str> {
        let rest = line.trim_start().strip_prefix('#')?.trim_start().strip_prefix("region:")?;
        let name = rest.trim();
        (!name.is_empty()).then_some(name)
    }
    let mut sections = Vec::new();
    let (mut name, mut buf) = ("all".to_string(), String::new());
    for line in text.lines() {
        if let Some(n) = marker(line) {
            let w = words(&buf);
            if !w.is_empty() {
                sections.push((name, w));
            }
            name = n.to_string();
            buf.clear();
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    let w = words(&buf);
    if !w.is_empty() {
        sections.push((name, w));
    }
    sections
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// The same word on both sides.
    Same(usize, usize),
    /// A different word was heard.
    Substituted(usize, usize),
    /// A reference word with nothing heard for it.
    Missing(usize),
    /// A heard word with no reference word.
    Extra(usize),
}

/// Minimum-edit alignment of heard words against reference words.
///
/// Among alignments with the fewest edits, the one with the most exactly matching words
/// wins, so an error is reported as a missing word and an extra word instead of two
/// substitutions that happen to cost the same.
pub fn align(reference: &[String], heard: &[String]) -> Vec<Op> {
    let (n, m) = (reference.len(), heard.len());
    let big = (n + m + 1) as i64;
    let w = m + 1;
    let mut cost = vec![0i64; (n + 1) * w];
    for i in 0..=n {
        cost[i * w] = i as i64 * big;
    }
    for j in 0..=m {
        cost[j] = j as i64 * big;
    }
    let step = |i: usize, j: usize| if reference[i - 1] == heard[j - 1] { -1 } else { big };
    for i in 1..=n {
        for j in 1..=m {
            let diag = cost[(i - 1) * w + j - 1] + step(i, j);
            cost[i * w + j] = diag.min(cost[(i - 1) * w + j] + big).min(cost[i * w + j - 1] + big);
        }
    }
    let (mut ops, mut i, mut j) = (Vec::with_capacity(n.max(m)), n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && cost[i * w + j] == cost[(i - 1) * w + j - 1] + step(i, j) {
            ops.push(if reference[i - 1] == heard[j - 1] { Op::Same(i - 1, j - 1) } else { Op::Substituted(i - 1, j - 1) });
            i -= 1;
            j -= 1;
        } else if i > 0 && cost[i * w + j] == cost[(i - 1) * w + j] + big {
            ops.push(Op::Missing(i - 1));
            i -= 1;
        } else {
            ops.push(Op::Extra(j - 1));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Counts {
    pub reference_words: usize,
    pub correct: usize,
    pub substituted: usize,
    pub missing: usize,
    pub extra: usize,
}

impl Counts {
    pub fn errors(&self) -> usize {
        self.substituted + self.missing + self.extra
    }
    /// Word error rate: errors divided by reference words.
    pub fn wer(&self) -> f64 {
        self.errors() as f64 / self.reference_words as f64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SectionScore {
    pub name: String,
    #[serde(flatten)]
    pub counts: Counts,
    pub wer: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MissingRun {
    pub section: String,
    pub first_word: usize,
    pub words: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Score {
    #[serde(flatten)]
    pub counts: Counts,
    pub wer: f64,
    /// Extra or wrong words that form runs of playback text.
    pub leaked: usize,
    pub sections: Vec<SectionScore>,
    pub missing_runs: Vec<MissingRun>,
}

impl Score {
    pub fn section(&self, name: &str) -> Option<&SectionScore> {
        self.sections.iter().find(|s| s.name == name)
    }
}

/// How many of the words in `unmatched` lie in a run of at least `run` consecutive words
/// that also occurs, in the same order, somewhere in `playback`.
pub fn leaked_words(unmatched: &[String], playback: &[String], run: usize) -> usize {
    let (n, m) = (unmatched.len(), playback.len());
    if n == 0 || m == 0 || run == 0 {
        return 0;
    }
    // len[j]: length of the common run ending at unmatched[i], playback[j].
    let mut covered = vec![false; n];
    let mut prev = vec![0usize; m + 1];
    for i in 0..n {
        let mut cur = vec![0usize; m + 1];
        for j in 0..m {
            if unmatched[i] == playback[j] {
                cur[j + 1] = prev[j] + 1;
                if cur[j + 1] >= run {
                    for c in covered.iter_mut().take(i + 1).skip(i + 1 - cur[j + 1]) {
                        *c = true;
                    }
                }
            }
        }
        prev = cur;
    }
    covered.into_iter().filter(|c| *c).count()
}

pub const LEAK_RUN: usize = 3;
pub const MISSING_RUN: usize = 4;

/// Score heard words against a sectioned reference.
///
/// The alignment is over the whole reference, so a word that falls on the other side of a
/// section boundary still counts as correct. Each error is charged to the section of the
/// reference word it belongs to; an extra word is charged to the section of the reference
/// word that follows it (the last section at the very end).
pub fn score(reference: &[Section], heard: &[String], playback: &[String]) -> Score {
    let mut all = Vec::new();
    let mut owner = Vec::new();
    let mut sections: Vec<(String, Counts)> = Vec::new();
    for (name, sec) in reference {
        let k = match sections.iter().position(|(n, _)| n == name) {
            Some(k) => k,
            None => {
                sections.push((name.clone(), Counts::default()));
                sections.len() - 1
            }
        };
        sections[k].1.reference_words += sec.len();
        all.extend(sec.iter().cloned());
        owner.extend(std::iter::repeat(k).take(sec.len()));
    }
    let ops = align(&all, heard);
    // Section of each operation: extra words take the section of the next reference word.
    let mut op_section = vec![owner.last().copied(); ops.len()];
    let mut next = owner.last().copied();
    for (k, op) in ops.iter().enumerate().rev() {
        match *op {
            Op::Same(i, _) | Op::Substituted(i, _) | Op::Missing(i) => next = Some(owner[i]),
            Op::Extra(_) => {}
        }
        op_section[k] = next;
    }
    let mut total = Counts { reference_words: all.len(), ..Counts::default() };
    let mut unmatched = Vec::new();
    let mut missing_runs = Vec::new();
    let mut run: Option<(usize, usize)> = None;
    let close = |run: &mut Option<(usize, usize)>, runs: &mut Vec<MissingRun>| {
        if let Some((first, len)) = run.take() {
            if len >= MISSING_RUN {
                runs.push(MissingRun { section: sections_name(reference, &owner, first), first_word: first, words: len });
            }
        }
    };
    for (k, op) in ops.iter().enumerate() {
        let mut bump = |f: fn(&mut Counts) -> &mut usize| {
            *f(&mut total) += 1;
            if let Some(s) = op_section[k] {
                *f(&mut sections[s].1) += 1;
            }
        };
        match *op {
            Op::Same(..) => {
                bump(|c| &mut c.correct);
                close(&mut run, &mut missing_runs);
            }
            Op::Substituted(_, j) => {
                bump(|c| &mut c.substituted);
                unmatched.push(heard[j].clone());
                close(&mut run, &mut missing_runs);
            }
            Op::Missing(i) => {
                bump(|c| &mut c.missing);
                run = Some(run.map_or((i, 1), |(f, l)| (f, l + 1)));
            }
            Op::Extra(j) => {
                bump(|c| &mut c.extra);
                unmatched.push(heard[j].clone());
            }
        }
    }
    close(&mut run, &mut missing_runs);
    Score {
        wer: total.wer(),
        leaked: leaked_words(&unmatched, playback, LEAK_RUN),
        sections: sections.into_iter().map(|(name, counts)| SectionScore { name, wer: counts.wer(), counts }).collect(),
        missing_runs,
        counts: total,
    }
}

fn sections_name(reference: &[Section], owner: &[usize], word: usize) -> String {
    let mut names: Vec<&String> = Vec::new();
    for (name, _) in reference {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names[owner[word]].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }
    fn one(s: &str) -> Vec<Section> {
        vec![("all".to_string(), w(s))]
    }

    #[test]
    fn words_ignore_case_and_punctuation_and_split_hyphens() {
        assert_eq!(words("Clank-clank, they came; it's 3 o'clock\u{2014}now."), w("clank clank they came it's 3 o'clock now"));
    }

    #[test]
    fn a_reference_without_markers_is_one_section() {
        assert_eq!(read_reference("One two.\nThree."), one("one two three"));
    }

    #[test]
    fn region_markers_split_the_reference() {
        let r = read_reference("# region: quiet\none two\n# region: over\nthree\n");
        assert_eq!(r, vec![("quiet".to_string(), w("one two")), ("over".to_string(), w("three"))]);
    }

    #[test]
    fn a_perfect_transcript_has_no_errors() {
        let s = score(&one("a b c d"), &w("a b c d"), &[]);
        assert_eq!((s.counts.correct, s.counts.errors(), s.wer), (4, 0, 0.0));
    }

    #[test]
    fn one_substitution_one_missing_and_one_extra_word_are_counted_separately() {
        // reference: the cat sat on the mat (6 words)
        // heard:     the cat sit on mat today -> sat->sit, "the" missing, "today" extra
        let s = score(&one("the cat sat on the mat"), &w("the cat sit on mat today"), &[]);
        assert_eq!((s.counts.correct, s.counts.substituted, s.counts.missing, s.counts.extra), (4, 1, 1, 1));
        assert_eq!(s.wer, 0.5);
    }

    #[test]
    fn errors_are_charged_to_the_section_of_the_reference_word() {
        let r = vec![("quiet".to_string(), w("one two three")), ("over".to_string(), w("four five six"))];
        let s = score(&r, &w("one two three four six"), &[]);
        assert_eq!(s.section("quiet").unwrap().counts.missing, 0);
        assert_eq!(s.section("over").unwrap().counts.missing, 1);
        assert_eq!(s.section("over").unwrap().counts.reference_words, 3);
        assert_eq!(s.section("over").unwrap().wer, 1.0 / 3.0);
    }

    #[test]
    fn a_word_recognised_across_a_section_boundary_still_counts() {
        let r = vec![("a".to_string(), w("one two")), ("b".to_string(), w("three four"))];
        assert_eq!(score(&r, &w("one two three four"), &[]).counts.errors(), 0);
    }

    #[test]
    fn an_extra_word_is_charged_to_the_section_that_follows_it() {
        let r = vec![("a".to_string(), w("one two")), ("b".to_string(), w("three four"))];
        let s = score(&r, &w("one two um three four"), &[]);
        assert_eq!((s.section("a").unwrap().counts.extra, s.section("b").unwrap().counts.extra), (0, 1));
    }

    #[test]
    fn a_long_run_of_missing_words_is_reported_with_its_place() {
        let r = vec![("a".to_string(), w("w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"))];
        let s = score(&r, &w("w0 w1 w7 w8 w9"), &[]);
        assert_eq!(s.missing_runs, vec![MissingRun { section: "a".into(), first_word: 2, words: 5 }]);
    }

    #[test]
    fn short_gaps_are_not_reported_as_runs() {
        let s = score(&one("w0 w1 w2 w3 w4"), &w("w0 w3 w4"), &[]);
        assert_eq!(s.counts.missing, 2);
        assert!(s.missing_runs.is_empty());
    }

    #[test]
    fn playback_words_in_the_transcript_are_counted_as_leaked() {
        let played = w("the weather today is sunny with light wind");
        let s = score(&one("please open the door"), &w("please open today is sunny the door"), &played);
        assert_eq!((s.leaked, s.counts.extra), (3, 3));
    }

    #[test]
    fn reference_words_that_also_occur_in_the_playback_are_not_leaks() {
        let s = score(&one("today is sunny outside"), &w("today is sunny outside"), &w("today is sunny"));
        assert_eq!(s.leaked, 0);
    }

    #[test]
    fn two_stray_words_are_below_the_leak_threshold() {
        assert_eq!(leaked_words(&w("is sunny"), &w("today is sunny"), 3), 0);
    }

    #[test]
    fn overlapping_playback_runs_count_each_word_once() {
        // "a b c d" contains the runs "a b c" and "b c d"; four words leaked, not six.
        assert_eq!(leaked_words(&w("a b c d"), &w("a b c d"), 3), 4);
    }

    #[test]
    fn an_empty_transcript_misses_every_word() {
        let s = score(&one("a b c"), &[], &[]);
        assert_eq!((s.counts.missing, s.wer), (3, 1.0));
    }

    #[test]
    fn alignment_covers_every_word_of_both_sides_once() {
        let ops = align(&w("a b c"), &w("b c d"));
        let mut r = vec![];
        let mut h = vec![];
        for op in ops {
            match op {
                Op::Same(i, j) | Op::Substituted(i, j) => {
                    r.push(i);
                    h.push(j)
                }
                Op::Missing(i) => r.push(i),
                Op::Extra(j) => h.push(j),
            }
        }
        r.sort();
        h.sort();
        assert_eq!((r, h), (vec![0, 1, 2], vec![0, 1, 2]));
    }
}
