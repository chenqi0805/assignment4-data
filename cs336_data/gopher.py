"""Gopher quality rules (problem 2.6).

Each predicate reports whether a document survives one of the heuristic
content filters from the Gopher paper (Rae et al. 2021, appendix A). The
handout mandates the four-rule subset (word count, mean word length,
ellipsis-ending lines, alphabetic-word fraction); the remaining paper rules
(symbol-to-word ratio, bullet-starting lines, stop words) are implemented for
parity with the paper. The paper defines words only implicitly, and standard
reimplementations (Dolma, RedPajama V2) split on whitespace, and so do we.
"""

import re

# "we remove any document that does not contain between 50 and 100,000 words"
MIN_WORDS = 50
MAX_WORDS = 100_000

# "whose mean word length is outside the range of 3 to 10 characters"
MIN_MEAN_WORD_LENGTH = 3
MAX_MEAN_WORD_LENGTH = 10

# "a symbol-to-word ratio greater than 0.1 for either the hash symbol or the
# ellipsis" — the paper counts the '#' and '…' characters.
SYMBOLS = ("#", "…")
MAX_SYMBOL_TO_WORD_RATIO = 0.1

# "more than 90% of lines starting with a bullet point"
_BULLET_PREFIXES = ("-", "*", "•", "·", "‣", "⁃")
MAX_BULLET_LINE_FRACTION = 0.9

# "more than 30% [of lines] ending with an ellipsis" — the handout's ellipsis
# is the ASCII "..." the fixtures use, not the single '…' character.
_ELLIPSIS = "..."
MAX_ELLIPSIS_LINE_FRACTION = 0.3

# "we ... require that 80% of words in a document contain at least one
# alphabetic character"
MIN_ALPHABETIC_WORD_FRACTION = 0.8

# "a 'stop word' filter, to remove documents that do not contain at least two
# of the following English words: the, be, to, of, and, that, have, with"
STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})
MIN_STOP_WORDS = 2

# RedPajama V2 extends the paper with a sentence-count rule ("contains fewer
# than 3 sentences"). It is NOT part of all_gopher_rules_pass: the provided
# suite's passing fixtures contain documents with no terminal punctuation at
# all, so a positive punctuation requirement contradicts the ground-truth
# tests. The predicate is exported for pipeline experiments (handout 2.6 says
# "at least" the four-rule subset).
_SENTENCE_END_RE = re.compile(r"[.!?]+")


def passes_word_count(text: str) -> bool:
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS


def passes_mean_word_length(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    mean_length = sum(len(word) for word in words) / len(words)
    return MIN_MEAN_WORD_LENGTH <= mean_length <= MAX_MEAN_WORD_LENGTH


def passes_symbol_to_word_ratio(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    return all(text.count(symbol) / len(words) <= MAX_SYMBOL_TO_WORD_RATIO for symbol in SYMBOLS)


def passes_bullet_lines(text: str) -> bool:
    lines = text.split("\n")
    bullets = sum(1 for line in lines if line.lstrip().startswith(_BULLET_PREFIXES))
    return bullets / len(lines) <= MAX_BULLET_LINE_FRACTION


def passes_ellipsis_lines(text: str) -> bool:
    lines = text.split("\n")
    ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith(_ELLIPSIS))
    return ellipsis_lines / len(lines) <= MAX_ELLIPSIS_LINE_FRACTION


def passes_alphabetic_words(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    alphabetic = sum(1 for word in words if any(c.isalpha() for c in word))
    return alphabetic / len(words) >= MIN_ALPHABETIC_WORD_FRACTION


def passes_stop_words(text: str) -> bool:
    distinct_words = {word.lower() for word in text.split()}
    return len(distinct_words & STOP_WORDS) >= MIN_STOP_WORDS


def passes_punctuation(text: str) -> bool:
    """RedPajama V2's sentence rule: at least three terminal punctuation marks.

    Excluded from all_gopher_rules_pass — see the module docstring.
    """
    return len(_SENTENCE_END_RE.findall(text)) >= 3


# The conjunction, in the paper's order. passes_punctuation is deliberately
# absent (see above); the other seven rules all hold on every passing fixture
# in the provided suite.
_RULES = (
    passes_word_count,
    passes_mean_word_length,
    passes_symbol_to_word_ratio,
    passes_bullet_lines,
    passes_ellipsis_lines,
    passes_alphabetic_words,
    passes_stop_words,
)


def all_gopher_rules_pass(text: str) -> bool:
    return all(rule(text) for rule in _RULES)
