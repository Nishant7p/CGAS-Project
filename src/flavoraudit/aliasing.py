"""Ingredient aliasing: map raw recipe lines to CulinaryDB entities (Singh & Bagler 2018, Sec. IV-A).

Protocol (reconstructed and validated against CulinaryDB's 160k expert-aliased lines; analysis/aliasing/):

1. lower-case; strip punctuation, digits and special characters;
2. singularize every word with ``inflect`` (CulinaryDB did the same: its gold labels contain 'asparagu', 'molass');
3. drop English and culinary stop-words;
4. match dictionary entries built from entity names and synonyms, LONGEST entry first. CulinaryDB's synonym
   notation: '-' separates parts that may appear in any order ('pepper-bell' matches "green bell pepper");
   '=' joins words that may be written apart or together ('corn=starch' matches "corn starch" and "cornstarch");
   synonyms wrapped in '#..#' are disabled (CulinaryDB never assigns those entities; a same-name group entity is
   used instead);
5. if nothing matches fully, accept the best PARTIAL match of a multi-part synonym (CulinaryDB's curators accepted
   these: "teriyaki sauce" -> Soybean Sauce via 'soy-sauce', "grated parmesan" -> Parmesan Cheese);
6. one entity per line. Ties between equally good matches are broken by preferences learned from expert-aliased
   lines (e.g. "garlic cloves" -> Garlic, "jalapeno pepper" -> Jalapeno), then by the rightmost match;
7. optional mined synonyms: frequent n-grams of unmatched lines that point to one entity with high precision
   (the paper's n-gram curation step, done automatically on training data only).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import inflect

_P = inflect.engine()
_WORD = re.compile(r"[a-z]+")

# NLTK's English stop-word list (static copy so the library needs no corpus download).
ENGLISH_STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
    it its itself they them their theirs themselves what which who whom this that these those am is are was were be
    been being have has had having do does did doing a an the and but if or because as until while of at by for with
    about against between into through during before after above below to from up down in out on off over under again
    further then once here there when where why how all any both each few more most other some such no nor not only own
    same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven
    isn ma mightn mustn needn shan shouldn wasn weren won wouldn""".split()  # noqa: SIM905 (readable list)
)

# Units, sizes and preparation words; an optional extra list (the evaluation compares with/without it).
CULINARY_STOPWORDS = frozenset(
    """cup tablespoon tbsp tbs tbl teaspoon tsp ounce oz pound lb gram g kg kilogram ml millilitre milliliter l litre
    liter quart qt pint gallon pinch dash drop handful can tin package pkg packet jar bottle box bag carton container
    bunch sprig stalk slice piece inch cm strip sheet envelope stick cube wedge scoop splash
    large small medium big jumbo extra thin thick fine finely coarse coarsely roughly thinly thickly lightly heavy
    fresh freshly dried dry frozen thawed canned cooked uncooked raw ripe firm soft softened melted room temperature
    chopped minced diced sliced grated shredded crushed ground mashed pureed julienned cubed halved quartered
    peeled seeded pitted cored trimmed rinsed drained divided beaten whisked sifted packed loosely
    optional taste needed desired garnish serving plus additional approximately approx about cut into
    whole boneless skinless deveined shelled washed cleaned torn crumbled toasted roasted
    hot cold warm lukewarm boiling""".split()  # noqa: SIM905 (readable list)
)


@lru_cache(maxsize=200_000)
def singular(word: str) -> str:
    s = _P.singular_noun(word)
    return s if s else word


def tokens(text: str, stopwords: frozenset[str] = ENGLISH_STOPWORDS) -> list[str]:
    """Normalized, singularized tokens of a raw line, without stop-words."""
    out = []
    for w in _WORD.findall(str(text).lower()):
        s = singular(w)
        if s not in stopwords and w not in stopwords:
            out.append(s)
    return out


@dataclass(frozen=True)
class Entry:
    entity_id: int
    parts: tuple[tuple[str, ...], ...]  # each part: words that must be adjacent (or written joined)
    label: str

    @property
    def n_words(self) -> int:
        return sum(len(p) for p in self.parts)


@dataclass
class Match:
    entity_id: int | None
    label: str | None
    how: str  # "full" | "partial" | "override" | "none"
    candidates: tuple[int, ...] = ()


def decode(syn: str, split_parts: bool = True) -> tuple[tuple[str, ...], ...]:
    s = syn.strip().strip("#").strip().lower()
    raw_parts = s.split("-") if split_parts else [s.replace("-", " ")]
    parts = []
    for raw in raw_parts:
        words = tuple(singular(w) for w in _WORD.findall(raw.replace("=", " ")))
        if words:
            parts.append(words)
    return tuple(parts)


@dataclass
class Aliaser:
    entries: list[Entry]
    stopwords: frozenset[str] = ENGLISH_STOPWORDS
    partial: bool = True
    tie_break: str = "rightmost"
    preferences: dict[tuple[int, int], int] = field(default_factory=dict)
    overrides: dict[str, int] = field(default_factory=dict)  # normalized token string -> entity id

    def __post_init__(self) -> None:
        self.entries = sorted(self.entries, key=lambda e: -e.n_words)
        self._by_word: dict[str, list[int]] = defaultdict(list)
        for k, e in enumerate(self.entries):
            for w in {w for p in e.parts for w in p}:
                self._by_word[w].append(k)
        self._joined = {"".join(p): p for e in self.entries for p in e.parts if len(p) > 1}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_culinarydb(cls, cdb, allowed_entities: set[int] | None = None, **kw) -> Aliaser:
        """Dictionary from CulinaryDB names + synonyms. allowed_entities defaults to entities CulinaryDB uses."""
        if allowed_entities is None:
            allowed_entities = set(int(e) for e in cdb.aliases["entity_id"].unique())
        entries = []
        for e, name, syns in zip(
            cdb.ingredients["entity_id"], cdb.ingredients["name"], cdb.ingredients["synonyms"], strict=True
        ):
            e = int(e)
            if e not in allowed_entities:
                continue
            if syns and all(s.strip().startswith("#") for s in syns):
                continue  # disabled entity (its same-name group entity is used instead)
            seen = set()
            forms = [(s, True) for s in syns if not s.strip().startswith("#")] + [(name, False)]
            for s, split in forms:
                parts = decode(s, split_parts=split)
                if parts and parts not in seen:
                    seen.add(parts)
                    entries.append(Entry(e, parts, s))
        return cls(entries, **kw)

    # ------------------------------------------------------------------ matching
    def _expand(self, toks: list[str]) -> list[str]:
        out: list[str] = []
        for t in toks:
            out.extend(self._joined.get(t, (t,)))
        return out

    @staticmethod
    def _find(part: tuple[str, ...], toks: list[str]) -> int:
        n = len(part)
        for i in range(len(toks) - n, -1, -1):
            if tuple(toks[i : i + n]) == part:
                return i
        return -1

    def candidates(self, toks: list[str]) -> tuple[dict[int, int], str]:
        """{entity_id: rightmost position} of the best-scoring matches, and how they matched."""
        tokset = set(toks)
        cand = sorted({k for w in tokset for k in self._by_word.get(w, ())})
        full: dict[int, int] = {}
        best_len = 0
        for k in cand:  # entries are sorted longest first
            e = self.entries[k]
            if e.n_words < best_len:
                break
            if not all(w in tokset for p in e.parts for w in p):
                continue
            pos = [self._find(p, toks) for p in e.parts]
            if min(pos) < 0:
                continue
            best_len = e.n_words
            full[e.entity_id] = max(full.get(e.entity_id, -1), max(pos))
        if full:
            return full, "full"
        if self.partial:
            best_score, part = 0, {}
            for k in cand:
                e = self.entries[k]
                if len(e.parts) < 2:
                    continue
                found = [(p, self._find(p, toks)) for p in e.parts]
                hit = [(p, i) for p, i in found if i >= 0]
                if not hit:
                    continue
                score = sum(len(p) for p, _ in hit)
                pos = max(i for _, i in hit)
                if score > best_score:
                    best_score, part = score, {e.entity_id: pos}
                elif score == best_score:
                    part[e.entity_id] = max(part.get(e.entity_id, -1), pos)
            if part:
                return part, "partial"
        return {}, "none"

    def _choose(self, cands: dict[int, int]) -> int:
        def key(a: int):
            pref = sum(self.preferences.get((a, b), 0) - self.preferences.get((b, a), 0) for b in cands if b != a)
            pos = cands[a] if self.tie_break == "rightmost" else -cands[a]
            return (pref, pos)

        return max(cands, key=key)

    def alias(self, line: str) -> Match:
        toks = tokens(line, self.stopwords)
        key = " ".join(toks)
        if key in self.overrides:
            return Match(self.overrides[key], key, "override")
        cands, how = self.candidates(self._expand(toks))
        if not cands:
            return Match(None, None, "none")
        e = self._choose(cands)
        return Match(e, key, how, tuple(sorted(cands)))

    # ------------------------------------------------------------------ learning from expert-aliased lines
    def fit_preferences(self, lines, gold) -> int:
        """Learn pairwise tie-break preferences: when the expert picked A among tied candidates {A, B, ...}."""
        wins: Counter = Counter()
        for line, g in zip(lines, gold, strict=True):
            cands, _ = self.candidates(self._expand(tokens(line, self.stopwords)))
            if g in cands and len(cands) > 1:
                for o in cands:
                    if o != g:
                        wins[(int(g), int(o))] += 1
        self.preferences = dict(wins)
        return len(wins)

    def mine_synonyms(
        self, lines, gold, min_support: int = 5, min_precision: float = 0.9, max_n: int = 3
    ) -> list[Entry]:
        """The paper's n-gram curation, automated: n-grams of UNMATCHED lines that reliably indicate one entity."""
        counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for line, g in zip(lines, gold, strict=True):
            toks = self._expand(tokens(line, self.stopwords))
            if self.candidates(toks)[1] != "none":
                continue
            grams = {tuple(toks[i : i + n]) for n in range(1, max_n + 1) for i in range(len(toks) - n + 1)}
            for gram in grams:
                counts[gram][int(g)] += 1
        new = []
        for gram, c in counts.items():
            ent, k = c.most_common(1)[0]
            total = sum(c.values())
            if total >= min_support and k / total >= min_precision:
                new.append(Entry(ent, (gram,), "mined:" + " ".join(gram)))
        return new

    @classmethod
    def reference(cls, cdb) -> Aliaser:
        """The production aliaser: CulinaryDB vocabulary + tie-break preferences learned from ALL its expert lines.

        Held-out accuracy of this configuration (fit on 80% of lines, tested on 20%): 97.9% of unique lines,
        99.0% of line occurrences (analysis/aliasing/evaluate_aliaser.py).
        """
        al = cls.from_culinarydb(cdb)
        lines = cdb.aliases["original"].astype(str).str.strip().str.lower()
        gold = (
            cdb.aliases.assign(phrase=lines)
            .groupby("phrase")["entity_id"]
            .agg(lambda s: Counter(s).most_common(1)[0][0])
        )
        al.fit_preferences(list(gold.index), [int(x) for x in gold.to_numpy()])
        return al

    def with_entries(self, extra: list[Entry]) -> Aliaser:
        return Aliaser(
            self.entries + extra,
            self.stopwords,
            self.partial,
            self.tie_break,
            dict(self.preferences),
            dict(self.overrides),
        )
