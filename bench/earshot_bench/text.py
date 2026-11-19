"""Words, alignment and scores. No audio here."""
import difflib
import re
from dataclasses import dataclass, field


def words(text):
    """Lower-case words. Hyphens split words; other punctuation is removed; apostrophes stay."""
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " ").replace("—", " "))


def read_reference(text):
    """Split a reference text into (section name, words) pairs.

    A line of the form `# region: NAME` starts a section. Text before the first such line,
    or a text with no such lines, belongs to the section named "all".
    """
    sections, name, buf = [], "all", []
    for line in text.splitlines():
        m = re.match(r"#\s*region:\s*(.+?)\s*$", line)
        if m:
            if words(" ".join(buf)):
                sections.append((name, words(" ".join(buf))))
            name, buf = m.group(1), []
        else:
            buf.append(line)
    if words(" ".join(buf)):
        sections.append((name, words(" ".join(buf))))
    return sections


def align(ref, hyp):
    """Minimum-edit alignment. Returns a list of (op, ref index or None, hyp index or None),
    op one of "=", "sub", "del" (reference word missing), "ins" (extra word)."""
    # Each cell holds edits * BIG - matches: among alignments with the fewest edits, the one
    # with the most exactly matching words wins, so errors are reported as a missing word and
    # an extra word instead of two substitutions that happen to cost the same.
    n, m = len(ref), len(hyp)
    BIG = n + m + 1
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i * BIG
    for j in range(m + 1):
        cost[0][j] = j * BIG
    for i in range(1, n + 1):
        row, up, r = cost[i], cost[i - 1], ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(up[j] + BIG, row[j - 1] + BIG, up[j - 1] + (BIG if r != hyp[j - 1] else -1))
    ops, i, j = [], n, m
    while i or j:
        same = i and j and ref[i - 1] == hyp[j - 1]
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (-1 if same else BIG):
            ops.append(("=" if same else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + BIG:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


@dataclass
class Score:
    reference_words: int = 0
    correct: int = 0
    substituted: int = 0
    missing: int = 0
    extra: int = 0
    leaked: int = 0
    sections: dict = field(default_factory=dict)
    missing_runs: list = field(default_factory=list)

    @property
    def errors(self):
        return self.substituted + self.missing + self.extra

    @property
    def wer(self):
        return self.errors / self.reference_words if self.reference_words else float("nan")

    def as_dict(self):
        d = {k: getattr(self, k) for k in ("reference_words", "correct", "substituted", "missing", "extra", "leaked")}
        d["wer"] = round(self.wer, 4)
        d["sections"] = {
            k: dict(v, wer=round((v["substituted"] + v["missing"] + v["extra"]) / v["reference_words"], 4))
            for k, v in self.sections.items()
        }
        d["missing_runs"] = self.missing_runs
        return d


def leaked_words(extra, playback, run=3):
    """How many of the words in `extra` (hypothesis words that did not match the reference,
    in order) form runs of at least `run` consecutive words that also occur in `playback`."""
    if not extra or not playback:
        return 0
    blocks = difflib.SequenceMatcher(None, extra, playback, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= run)


def score(reference_sections, hypothesis, playback=(), leak_run=3, missing_run=4):
    """Score hypothesis words against a sectioned reference.

    The alignment is over the whole reference, so a word that falls on the other side of a
    section boundary still counts as correct. Each error is charged to the section of the
    reference word it belongs to; an extra word is charged to the section of the reference
    word that follows it (the last section at the very end).
    """
    ref, owner = [], []
    for name, sec in reference_sections:
        ref += sec
        owner += [name] * len(sec)
    s = Score(reference_words=len(ref))
    for name, sec in reference_sections:
        t = s.sections.setdefault(name, dict(reference_words=0, correct=0, substituted=0, missing=0, extra=0))
        t["reference_words"] += len(sec)
    ops = align(ref, hypothesis)
    field_of = {"=": "correct", "sub": "substituted", "del": "missing", "ins": "extra"}
    unmatched, run_start, run_len = [], None, 0
    for k, (op, i, j) in enumerate(ops):
        if i is None:
            nxt = next((oi for _, oi, _ in ops[k + 1:] if oi is not None), None)
            sec = owner[nxt] if nxt is not None else (owner[-1] if owner else "all")
        else:
            sec = owner[i]
        setattr(s, field_of[op], getattr(s, field_of[op]) + 1)
        if sec in s.sections:
            s.sections[sec][field_of[op]] += 1
        if op in ("sub", "ins"):
            unmatched.append(hypothesis[j])
        if op == "del":
            run_start = i if run_len == 0 else run_start
            run_len += 1
        elif op != "ins":
            if run_len >= missing_run:
                s.missing_runs.append(dict(section=owner[run_start], first_word=run_start, words=run_len))
            run_len = 0
    if run_len >= missing_run:
        s.missing_runs.append(dict(section=owner[run_start], first_word=run_start, words=run_len))
    s.leaked = leaked_words(unmatched, list(playback), leak_run)
    return s
