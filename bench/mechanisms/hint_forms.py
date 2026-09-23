#!/usr/bin/env python
"""One recipe, injected many ways: which variable was the null result about?

WHY THIS EXISTS

`recipe_oracle.py` asks "does a minimal hint change the generated code?" and
answers it with ONE configuration: the recipe text appended to the end of the
user turn, imperative phrasing, one recipe at a time. That is a perfectly good
ceiling test for that configuration and it is about to be read as a ceiling
test for the IDEA.

It is not. "Inject a performance recipe" is at least three independent
variables stacked into a single cell:

  WHERE the hint goes   - a system message, a user prefix, a user suffix, or
                          the opening of an assistant turn the model continues
  HOW it is phrased     - an order, a fact, a checklist, a line of code, a
                          question, a condition
  HOW MUCH is injected  - one recipe, three, all five
  WHETHER IT IS RIGHT   - the matching recipe, or a misrouted one

A null result at one cell of that space says the cell is null. It does not say
the hypothesis is dead, and a corpus project should not be cancelled on it --
nor started on a single positive cell. This harness turns the one cell into a
matrix so a null can be attributed to a VARIABLE, and so a positive can be
attributed to something other than "we added words to the prompt".

WHAT IS HELD FIXED, AND WHY THAT MATTERS MORE THAN WHAT VARIES

The recipe CONTENT is identical across every form. The six phrasings are
rendered from one set of per-recipe facts (FACETS below), not hand-written six
times, because six independently written texts would vary in quality as well
as in form and the arm labelled "checklist" would really be measuring "the
checklist I happened to write better". Only the imperative form is taken
verbatim -- it is imported from recipe_oracle.RECIPES, so the baseline cell of
this matrix is byte-identical to the arm already measured there and the two
runs are comparable.

THE PLACEBO IS NOT OPTIONAL, AND ONE OF THEM IS NOT ENOUGH

recipe_oracle carries one placebo: same shape, same length, no performance
content. It has to, or a measured gain is indistinguishable from the effect of
adding text. This harness needs the same control at every count, because
"3 recipes beat 1 recipe" is exactly the claim that a longer neutral prompt
would also produce. So the placebo is a FAMILY of five neutral items, each
length-matched to the real recipe it stands in for, and the placebo arm at
count k uses the k placebos matched to the k real recipes in the real arm.
Character cost is reported per arm for the same reason: the stated design
constraint is "minimal hint -- do not balloon context", so cost sits in the
same table as benefit and neither can be quoted without the other.

THE HINT IS A PRIMER, NOT A MANDATE, AND THAT IS A TESTABLE DIFFERENCE

The hint's job is to get the model thinking in the family of recipes we care
about. It has to leave the model free to decide the recipe does not apply here.
That is not a stylistic preference, it is forced by the selector: a decision
tree three levels deep at 90% per node is 0.9^3 = 73% end to end, so roughly
one hint in four will arrive WRONG. A design that only works when routing is
correct is not viable at that error rate -- and nothing in the original
single-cell experiment measured what a wrong hint costs. The number does not
exist yet, and it is the number the architecture stands or falls on.

So this matrix carries two things the original could not:

  the CONDITIONAL register, which states its trigger before its advice, so a
  model can check the trigger, find it false, and drop the hint -- an escape an
  imperative sentence does not offer; and

  MISROUTED arms, which inject a deliberately mismatched recipe and measure the
  damage directly, in both registers, on the same problems.

HOW THE MISROUTED RESULT READS, WHICH IS NOT HOW THE OTHERS READ

A null on the right-hint arms is disappointing. A null on the WRONG-hint arms
is a GOOD outcome: it means a bad hint is free, and routing accuracy stops
being the binding constraint on the architecture. If instead wrong_conditional
costs significantly less than wrong_imperative, the conditional register is
buying robustness to routing error, and that is precisely what makes a lossy
selector acceptable. "No significant difference" is not uniformly bad news in
this table, and a write-up that reports it as though it were has misread the
experiment.

THE THINKING MODEL'S TRUNCATION IS AN OUTCOME, NOT A WRONG ANSWER

This model reasons before it writes and returns empty content with
finish_reason "length" when the budget runs out. Scoring that as a failed
solution would make any arm that costs more prompt tokens look worse at
correctness for a reason that has nothing to do with correctness -- and the
longer arms in this matrix are exactly the ones at risk. Every record carries
an explicit outcome: "scored", "budget_exhausted" or "no_code", and
budget_exhausted rows are counted and excluded rather than folded into pass@1
(PROTOCOL rule 3: never let a failure of the harness become a wrong answer).

NO GPU IS NEEDED TO FALSIFY THE CONSTRUCTION

Every arm's message array is built and checked against a mock generator in
`--dry-run`, which is the default. Message construction is where this harness
can silently be wrong -- a hint that lands in the wrong role, a prefill that is
not a prefill, a placebo that is half the length of what it controls for -- and
all of that is decidable without a single token of generation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

import livecodebench as lcb            # noqa: E402
import recipe_oracle as ro             # noqa: E402

# The recipes themselves are NOT redefined here. They are the ones already
# measured, so this matrix extends that experiment instead of starting a new
# one that cannot be compared to it.
RECIPES = ro.RECIPES
BY_ID = {r["id"]: r for r in RECIPES}
REAL_IDS = [r["id"] for r in RECIPES if r["id"] != "placebo"]

INJECTION_POINTS = ("system", "user_prefix", "user_suffix", "assistant_prefill")
FORMS = ("imperative", "declarative", "checklist", "fragment", "question",
         "conditional")
COUNTS = (1, 3, 5)

# Which recipes a given count selects. Fixed rather than random so an arm means
# the same thing on every problem and across re-runs: the count variable must
# not smuggle in a "which recipe" variable.
#   1 -> io          the largest, most mechanical Python effect; the one the
#                    oracle run already has a number for
#   3 -> io, structures, complexity   one from each family (I/O, data
#                    structure, algorithmic bound) rather than three
#                    neighbours, so count=3 is not "io said three ways"
COUNT_IDS = {1: ["io"],
             3: ["io", "structures", "complexity"],
             5: list(REAL_IDS)}

# ---------------------------------------------------------------------------
# Content, factored so that FORM varies and CONTENT does not.
#
# Each point carries the same fact in five registers. "imp" is an order, "dec"
# states it as true of the world, "code" shows it happening, "q" asks whether
# the situation applies without giving the answer -- that asymmetry is the
# point of the question arm and is why the question form is deliberately not
# content-matched to the checklist form.
#
# "con" states the TRIGGER first and the advice second, which is the register
# the whole misrouting question turns on: a model can read the trigger, decide
# it does not hold, and drop the advice. An imperative sentence gives it
# nowhere to put that judgement.
#
# This register is not an invention for the experiment. It is the voice the
# collected corpus already speaks in when it is at its best -- verbatim from
# bench/recipes/systems.jsonl line 22 (Agner Fog, Optimizing software in C++
# 10.4, tagged confidence "contested"):
#
#     "If the loop reads a[i] and b[i] together, one array of structs beats
#      two arrays. SoA only wins when the loop touches a subset of fields.
#      Know which case you have."
#
# Trigger, advice, and the counter-case under which the advice reverses. Across
# the three corpus files, 63 of 242 collected recipes (26%) already carry a
# trigger word -- a minority, so this arm is testing whether the corpus should
# be rewritten toward its own best examples, not assuming it.
# ---------------------------------------------------------------------------
FACETS = {
    "io": [
        {"imp": "read all input at once with sys.stdin.buffer.read().split(), "
                "never input() in a loop",
         "dec": "large inputs are read in one call with "
                "sys.stdin.buffer.read().split(); input() per line is the "
                "bottleneck",
         "con": "If the input runs to many thousands of lines, read it in one "
                "call with sys.stdin.buffer.read().split() rather than input() "
                "per line",
         "code": "data = sys.stdin.buffer.read().split()",
         "q": "is the input large enough that calling input() once per line "
              "will dominate the runtime?"},
        {"imp": "write one joined string to sys.stdout rather than print per "
                "line",
         "dec": "output goes out once, as a single joined string written to "
                "sys.stdout",
         "con": "If the answer is one line per query, collect the lines and "
                "write them joined to sys.stdout once, rather than printing "
                "inside the loop",
         "code": "sys.stdout.write('\\n'.join(out))",
         "q": "is there enough output that one print per line will cost more "
              "than the work does?"},
        {"imp": "use sys.stdin.readline where the input has to be read line "
                "by line",
         "dec": "where input must be taken line by line, sys.stdin.readline "
                "stands in for input()",
         "con": "If the input has to be taken line by line, sys.stdin.readline "
                "does it; input() is the slow way to say the same thing",
         "code": "line = sys.stdin.readline()",
         "q": "does anything here read line by line where sys.stdin.readline "
              "would do?"},
    ],
    "structures": [
        {"imp": "use collections.deque for queue pops, not list.pop(0)",
         "dec": "queue pops come from a collections.deque; list.pop(0) is "
                "linear in the length of the list",
         "con": "If anything pops from the front of a sequence, use "
                "collections.deque; list.pop(0) is linear every time",
         "code": "q = deque(start)",
         "q": "does anything pop from the front of a list where a deque "
              "belongs?"},
        {"imp": "use a set or dict for membership, never a list",
         "dec": "membership is tested against a set or dict; a list scan is "
                "linear",
         "con": "If a value is tested for membership more than once, keep it "
                "in a set or dict; a list scan is linear",
         "code": "seen = set(values)",
         "q": "is there a membership test against a list that a set would "
              "answer in constant time?"},
        {"imp": "preallocate with [0]*n where the size is known rather than "
                "appending repeatedly",
         "dec": "where the size is known up front, [0]*n replaces repeated "
                "append",
         "con": "If the final size is known before the loop starts, allocate "
                "[0]*n up front rather than appending",
         "code": "dp = [0] * n",
         "q": "is the size known up front, so the list can be preallocated?"},
    ],
    "strings": [
        {"imp": "build strings with ''.join(parts), never += in a loop",
         "dec": "strings are built with ''.join(parts); += in a loop copies "
                "the whole string on every pass",
         "con": "If a string is being grown inside a loop, collect the parts "
                "and ''.join(parts) at the end; += copies the whole string "
                "each pass",
         "code": "s = ''.join(parts)",
         "q": "is a string being grown with += inside a loop?"},
        {"imp": "slice rather than concatenate",
         "dec": "a slice takes the place of a concatenation where only part "
                "of the string is wanted",
         "con": "If only part of a string is wanted, slice it rather than "
                "rebuilding it by concatenation",
         "code": "head = s[:k]",
         "q": "is a substring being rebuilt where a slice would do?"},
        {"imp": "compare lengths before comparing contents",
         "dec": "lengths are compared before contents, so unequal strings are "
                "rejected in constant time",
         "con": "If two strings are compared often, check len() first; "
                "unequal lengths settle it in constant time",
         "code": "if len(a) != len(b): return False",
         "q": "could a length check reject this pair before the contents are "
              "touched?"},
    ],
    "loops": [
        {"imp": "hoist attribute and global lookups out of hot loops into "
                "locals",
         "dec": "attribute and global lookups are hoisted out of hot loops "
                "into locals",
         "con": "If an attribute or a global is read on every iteration of a "
                "hot loop, hoist it into a local first",
         "code": "append = out.append",
         "q": "is an attribute or a global being looked up on every "
              "iteration?"},
        {"imp": "prefer comprehensions and built-ins (sum, max, any) over "
                "manual Python-level loops",
         "dec": "built-ins (sum, max, any) and comprehensions run the loop in "
                "C rather than in Python",
         "con": "If a loop only sums, maxes or tests values, a built-in or a "
                "comprehension runs it in C instead of in Python",
         "code": "total = sum(values)",
         "q": "could a built-in run this loop in C instead of in Python?"},
        {"imp": "avoid recomputing inside the loop condition",
         "dec": "anything used by the loop condition is computed once, before "
                "the loop",
         "con": "If the loop condition recomputes a value that cannot change, "
                "compute it once before the loop",
         "code": "n = len(values)",
         "q": "is the loop condition recomputing something that does not "
              "change?"},
    ],
    "complexity": [
        {"imp": "check the input bound first: above 10^5 an O(n^2) scan will "
                "not finish",
         "dec": "the input bound decides the algorithm: above 10^5 an O(n^2) "
                "scan does not finish in time",
         "con": "If the input bound is above 10^5, an O(n^2) scan will not "
                "finish; read the bound before choosing the approach",
         "code": "# n up to 2*10^5 -> O(n log n) at worst",
         "q": "what is the input bound, and does it already rule out an "
              "O(n^2) scan?"},
        {"imp": "reach for sorting, prefix sums, a heap or a hash map before "
                "writing nested loops",
         "dec": "sorting, prefix sums, a heap or a hash map take the place of "
                "nested loops",
         "con": "If nested loops look like the answer, check first whether "
                "sorting, prefix sums, a heap or a hash map removes the inner "
                "one",
         "code": "pre = list(accumulate(values))",
         "q": "would sorting, a prefix sum, a heap or a hash map remove the "
              "inner loop?"},
        {"imp": "count the work in the inner loop against the stated limit",
         "dec": "the work in the inner loop is counted against the stated "
                "limit before it is written",
         "con": "If the inner loop is unavoidable, count its work against the "
                "stated limit before writing it",
         "code": "# 2*10^5 * log n ~= 3.6e6 steps",
         "q": "how many steps does the inner loop cost at the stated limit?"},
    ],
}

# The placebo family. Same shape, same register, same approximate length as the
# recipe each one stands in for, and no performance content whatsoever -- these
# are things a model would do anyway. "placebo" is recipe_oracle's control,
# kept under its original id so the control arm is also comparable across runs;
# the other four exist only so that count can be varied with the length
# confound still controlled.
PLACEBO_FACETS = {
    "placebo": [
        {"imp": "write the solution in Python and give the program a sensible "
                "structure",
         "dec": "the solution is a Python program with a sensible structure",
         "con": "If the problem asks for a program, write it in Python and "
                "give it a sensible structure",
         "code": "# solution.py",
         "q": "is the program laid out in a structure a reader would "
              "follow?"},
        {"imp": "use clear variable names",
         "dec": "variable names say what they hold",
         "con": "If a variable holds something specific, give it a name that "
                "says so",
         "code": "total_count = 0",
         "q": "do the variable names say what they hold?"},
        {"imp": "handle the input format described in the problem statement",
         "dec": "the input format is the one described in the problem "
                "statement",
         "con": "If the statement describes an input format, handle that "
                "format",
         "code": "# input format: as described in the statement",
         "q": "does the reader match the input format in the statement?"},
    ],
    "placebo2": [
        {"imp": "read the whole problem statement before you start writing",
         "dec": "the whole problem statement is read before any code is "
                "written",
         "con": "If the statement is long, read all of it before writing any "
                "code",
         "code": "# statement read in full",
         "q": "has the whole statement been read before writing starts?"},
        {"imp": "name the pieces of the solution after what they represent",
         "dec": "the pieces of the solution are named after what they "
                "represent",
         "con": "If a piece of the solution represents something, name it "
                "after that",
         "code": "grid_rows = []",
         "q": "is each piece named after what it represents?"},
        {"imp": "keep the program in a single file that runs top to bottom",
         "dec": "the program is a single file that runs from top to bottom",
         "con": "If the program is small, keep it in a single file that runs "
                "top to bottom",
         "code": "if __name__ == '__main__': main()",
         "q": "does the program run top to bottom from one file?"},
    ],
    "placebo3": [
        {"imp": "write the program in ordinary Python, nothing exotic",
         "dec": "the program is ordinary Python, using nothing exotic",
         "con": "If ordinary Python expresses the solution, use ordinary "
                "Python and nothing exotic",
         "code": "def main():",
         "q": "is the program written in ordinary Python?"},
        {"imp": "keep the code readable",
         "dec": "the code is readable",
         "con": "If a line is hard to read, write it so that it reads "
                "clearly",
         "code": "value = parts[0]",
         "q": "is the code readable as written?"},
        {"imp": "print the answer in the format the statement asks for",
         "dec": "the answer is printed in the format the statement asks for",
         "con": "If the statement asks for a particular output format, print "
                "the answer in it",
         "code": "print(answer)",
         "q": "is the answer printed in the format the statement asks for?"},
    ],
    "placebo4": [
        {"imp": "put the logic of the solution in a function rather than at "
                "module level",
         "dec": "the logic of the solution sits in a function rather than at "
                "module level",
         "con": "If the logic runs to more than a couple of lines, put it in "
                "a function rather than at module level",
         "code": "def solve(data):",
         "q": "does the logic sit in a function rather than at module "
              "level?"},
        {"imp": "use the standard library rather than writing helpers from "
                "scratch",
         "dec": "the standard library is used rather than helpers written "
                "from scratch",
         "con": "If the standard library already does it, use the standard "
                "library rather than writing a helper",
         "code": "import math",
         "q": "is the standard library doing the work a helper would?"},
        {"imp": "return the result instead of printing from inside the logic",
         "dec": "the result is returned rather than printed from inside the "
                "logic",
         "con": "If a function computes the answer, return it rather than "
                "printing from inside",
         "code": "return answer",
         "q": "is the result returned rather than printed from inside?"},
    ],
    "placebo5": [
        {"imp": "work through the sample case by hand before you write the "
                "code",
         "dec": "the sample case is worked through by hand before the code is "
                "written",
         "con": "If the statement includes a sample case, work through it by "
                "hand before writing the code",
         "code": "# sample: 3 -> 6",
         "q": "has the sample case been worked through by hand first?"},
        {"imp": "make sure the program handles the smallest input in the "
                "stated range",
         "dec": "the program handles the smallest input in the stated range",
         "con": "If the stated range has a smallest input, make sure the "
                "program handles it",
         "code": "# n == 1 is legal here",
         "q": "does the program handle the smallest input in the range?"},
        {"imp": "check the output against the sample before finishing",
         "dec": "the output is checked against the sample before finishing",
         "con": "If there is a sample output, check the program against it "
                "before finishing",
         "code": "# matches sample output",
         "q": "has the output been checked against the sample?"},
    ],
}

# Which placebo stands in for which recipe. Index-matched, so a placebo arm at
# any count is length-matched item for item, not just in total.
PLACEBO_FOR = dict(zip(REAL_IDS, ["placebo", "placebo2", "placebo3",
                                  "placebo4", "placebo5"]))

# A token that must survive into every WIDE form of a recipe (imperative,
# declarative, checklist). Checked in the dry run: if a rewrite ever drops the
# actual mechanism from one form, that arm stops measuring form and starts
# measuring content, and the difference is invisible by eye.
#
# The token also has to be UNIQUE to its recipe, because the count assertion
# counts tokens to decide how many recipes were injected. The first version
# used "join" for strings and the count=1 arm reported two recipes: the io
# recipe says "write one joined string". A token that is merely present is not
# a token that identifies.
TOPIC_TOKEN = {"io": "stdin", "structures": "deque", "strings": "''.join",
               "loops": "hoist", "complexity": "o(n^2)"}

# Wide forms carry every point of every selected recipe; narrow forms carry the
# lead point only. That is what makes a fragment a fragment and a question a
# question, and it means form is partly confounded with length BY DESIGN --
# which is why arm_cost() reports the length of every arm next to its result.
#
# conditional is a wide form: it says everything the imperative says and then
# the trigger on top, so it is the most expensive register in the matrix. That
# cost is the price of the escape it offers, and the two have to be read
# together or the arm looks like a free win.
WIDE_FORMS = ("imperative", "declarative", "checklist", "conditional")
NARROW_FORMS = ("fragment", "question")

# Words a conditional may open with. Checked in the dry run: a "conditional"
# that does not state a trigger first is just a longer imperative, and the arm
# would be measuring length.
TRIGGER_WORDS = ("if", "when", "where", "unless", "above", "below", "once")

LEAD = "Keep this in mind while you write:"

# The assistant prefill has to read as the model's own words or it is not a
# prefill, so it cannot reuse LEAD. It deliberately does NOT open a ```python
# fence: an open fence would change what lcb.extract_code() sees, and the
# injection-point variable would then be confounded with a parsing change.
PREFILL_HEAD = "Before I write the program, the things that decide the "\
               "runtime here:\n"
PREFILL_TAIL = "\n\nNow the program."


def _points(recipe_id: str) -> list:
    return FACETS.get(recipe_id) or PLACEBO_FACETS[recipe_id]


def render_hint(ids: list, form: str) -> str:
    """The hint text for a set of recipes in one form. No positioning yet.

    Multi-recipe rendering takes the LEAD point of each recipe rather than all
    of them. Five recipes x three bullets is fifteen bullets, which is no
    longer a checklist in any sense a reader would recognise, and the count arm
    would then be varying form as well as count.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    single = len(ids) == 1

    if form == "imperative":
        # Verbatim from the measured corpus. This is the only form that is not
        # rendered, and it is why the default cell here reproduces the oracle.
        return "\n".join(BY_ID[i]["text"] if i in BY_ID
                         else _imperative_text(i) for i in ids)

    if form == "declarative":
        out = []
        for i in ids:
            pts = _points(i) if single else _points(i)[:1]
            out.append(". ".join(p["dec"][0].upper() + p["dec"][1:]
                                 for p in pts) + ".")
        return "\n".join(out)

    if form == "conditional":
        # Trigger first, advice second, one sentence per point. The full stop
        # is added here rather than stored, so the stored text stays a clause
        # and the registers keep sharing one source of content.
        out = []
        for i in ids:
            pts = _points(i) if single else _points(i)[:1]
            out.append(" ".join(p["con"].rstrip(".") + "." for p in pts))
        return "\n".join(out)

    if form == "checklist":
        bullets = []
        for i in ids:
            pts = _points(i) if single else _points(i)[:1]
            bullets += ["- " + p["imp"][0].upper() + p["imp"][1:] for p in pts]
        return "\n".join(bullets)

    if form == "fragment":
        return "\n".join(_points(i)[0]["code"] for i in ids)

    # question
    return "\n".join(_points(i)[0]["q"][0].upper() + _points(i)[0]["q"][1:]
                     for i in ids)


def _imperative_text(placebo_id: str) -> str:
    """Imperative text for a placebo that recipe_oracle does not carry.

    Only placebo2..placebo5 land here; "placebo" itself comes from the imported
    RECIPES, so the control arm keeps its original wording.
    """
    pts = PLACEBO_FACETS[placebo_id]
    return " ".join(p["imp"][0].upper() + p["imp"][1:] + "." for p in pts)


def select_ids(count: int, content: str, recipe_ids: list | None = None) -> list:
    """The recipe ids an arm injects.

    An explicit recipe_ids wins, which is what makes the generator a generator
    rather than the fixed list this harness exists to replace.
    """
    if recipe_ids is not None:
        ids = list(recipe_ids)
    else:
        if count not in COUNT_IDS:
            raise ValueError(f"count must be one of {sorted(COUNT_IDS)}")
        ids = list(COUNT_IDS[count])
    if content == "placebo":
        # Index-matched substitution, so the control is matched item by item.
        ids = [PLACEBO_FOR[i] if i in PLACEBO_FOR else i for i in ids]
    elif content != "recipe":
        raise ValueError(f"content must be 'recipe' or 'placebo', not {content!r}")
    return ids


# ---------------------------------------------------------------------------
# Misrouting. What does a WRONG hint cost?
#
# The selector is a decision tree, its per-node errors compound, and at a
# plausible 90% per node over three levels roughly a quarter of hints arrive
# mismatched. That is not an edge case to be handled later, it is the ordinary
# operating condition, and the experiment has to price it.
#
# "Furthest from the problem" is operationalised as: a recipe with no topical
# cue anywhere in the statement, chosen from the pool by a rotation keyed on a
# SHA-1 of the question id.
#
#   - SHA-1, not hash(). Python salts str hashing per process, so a hash()-keyed
#     rotation would pick a different wrong recipe on every run and the arm
#     would not be reproducible. The dry run checks determinism in a FRESH
#     interpreter with a different PYTHONHASHSEED for exactly this reason.
#   - The pool excludes "io", because that is the recipe the right-hint arms
#     inject; a misroute arm that happened to pick it would be a right-hint arm
#     wearing a different label.
#   - The pool also excludes "loops", which has no reliable textual cue and
#     whose advice (hoist lookups, prefer built-ins) is close to universally
#     applicable. Injecting it is not a genuine misroute, and counting it as
#     one would make misrouting look CHEAPER than it is -- the dangerous
#     direction for a viability decision.
#   - The cue lists are crude keyword matches and are not trying to be a
#     selector. Their only job is conservative exclusion: anything that might
#     be relevant is disqualified from being called wrong.
# ---------------------------------------------------------------------------
# Measured over all 175 problems in data/test6.jsonl: the rule picks
# structures 72, strings 66, complexity 37, and has to relax the cue filter on
# 3 rows (1.7%, flagged per record). Three pairings rather than one, so this
# arm measures "a wrong hint" and not "the one wrong hint I happened to pick";
# complexity comes up least because its cues (10^5 and friends) appear in most
# statements, which is the filter being conservative in the intended direction.
MISROUTE_CUES = {
    "io": ("10^5", "10^6", "queries", "lines"),
    "structures": ("queue", "stack", "graph", "adjacent", "bfs", "tree",
                   "edge", "vertex", "vertices"),
    "strings": ("string", "character", "substring", "letter", "palindrome"),
    "loops": (),
    "complexity": ("10^5", "10^6", "10^9", "100000", "200000", "300000"),
}
MISROUTE_EXCLUDED = set(COUNT_IDS[1]) | {"loops"}


def wrong_recipe_for(row: dict) -> tuple:
    """(recipe_id, degraded) -- a recipe deterministically wrong for this row.

    `degraded` is true when every candidate had a topical cue in the statement
    and the filter had to be dropped to return anything at all. Those rows
    still get a hint, but they are flagged, counted and reported, because a
    fallback whose rate is invisible is the zero-vector bug again: the analysis
    must be able to drop them rather than quietly average a possibly-relevant
    hint into the cost of being wrong (PROTOCOL rule 2).
    """
    q = (row.get("question_content") or "").lower()
    pool = [i for i in REAL_IDS if i not in MISROUTE_EXCLUDED
            and not any(cue in q for cue in MISROUTE_CUES[i])]
    degraded = not pool
    if degraded:
        pool = [i for i in REAL_IDS if i not in MISROUTE_EXCLUDED]
    h = int(hashlib.sha1(str(row.get("question_id")).encode()).hexdigest(), 16)
    return pool[h % len(pool)], degraded


def base_prompt(row: dict) -> str:
    """The unhinted prompt, from livecodebench's own templates.

    Identical to recipe_oracle.prompt_for(row, None) by construction: if these
    two ever diverge, the no-hint arm here is not the no-hint arm there and the
    runs stop being comparable.
    """
    functional = bool(row.get("starter_code", "").strip())
    if functional:
        return lcb.PROMPT_FUNCTIONAL.format(question=row["question_content"],
                                            starter=row["starter_code"])
    return lcb.PROMPT_STDIN.format(question=row["question_content"])


def base_messages(row: dict) -> list:
    return [{"role": "user", "content": base_prompt(row)}]


def build_messages(row: dict, injection_point: str = "user_suffix",
                   form: str = "imperative", count: int = 1,
                   recipe_ids: list | None = None,
                   content: str = "recipe") -> list:
    """One cell of the matrix, as an OpenAI messages array.

    The hint BLOCK is byte-identical across system, user_prefix and
    user_suffix -- only where it is placed changes. That is the whole reason
    those three arms can be compared: they differ in position and in nothing
    else, including length.
    """
    if injection_point not in INJECTION_POINTS:
        raise ValueError(f"unknown injection point {injection_point!r}")
    msgs = base_messages(row)                      # fresh objects every call
    if injection_point is None:
        return msgs
    ids = select_ids(count, content, recipe_ids)
    hint = render_hint(ids, form)
    block = f"{LEAD}\n{hint}"

    if injection_point == "system":
        return [{"role": "system", "content": block}] + msgs
    if injection_point == "user_prefix":
        # The trailing newline is not decoration. recipe_oracle's suffix block
        # ends with one, and without a matching one here the prefix arm would
        # be one character cheaper than the suffix arm -- which is nothing in
        # itself, but it means the two arms are no longer a pure position
        # contrast, and "pure position contrast" is the only claim this arm
        # makes. The dry run asserts the two lengths are equal.
        return [{"role": "user",
                 "content": f"{block}\n\n{msgs[0]['content']}\n"}]
    if injection_point == "user_suffix":
        # Reproduces recipe_oracle.HINT_BLOCK exactly; asserted in the dry run.
        return [{"role": "user",
                 "content": msgs[0]["content"] + f"\n\n{block}\n"}]
    return msgs + [{"role": "assistant",
                    "content": PREFILL_HEAD + hint + PREFILL_TAIL}]


def arm_ids(row: dict, arm: dict) -> list:
    """The recipe ids an arm actually injects into THIS row.

    Row-dependent only for the misrouted arms. Returned separately from the
    messages so the live loop can record it: an effect on a wrong-hint arm is
    useless unless it can be traced to the specific pairing that produced it,
    and "some wrong recipe" is not a pairing.
    """
    if arm.get("point") is None:
        return []
    if arm.get("misroute"):
        return [wrong_recipe_for(row)[0]]
    return select_ids(arm["count"], arm.get("content", "recipe"),
                      arm.get("ids"))


def messages_for_arm(row: dict, arm: dict) -> list:
    if arm.get("point") is None:
        return base_messages(row)
    return build_messages(row, arm["point"], arm["form"], arm["count"],
                          arm_ids(row, arm), arm.get("content", "recipe"))


# ---------------------------------------------------------------------------
# The arms.
#
# WHY NOT THE FULL PRODUCT. 4 points x 5 forms x 3 counts x 2 contents is 120
# cells. At recipe_oracle's n=18 problems that is 2,160 generations, which on
# one GPU is days, and it would still be the wrong design:
#
#   - PROTOCOL rule 4. Every cell gets compared to the baseline, so 120 cells
#     is 120 tests against one control. Corrected for multiplicity at n=18
#     paired problems, nothing short of an enormous effect survives, and a
#     harness that cannot detect the effect it was built for is the reranker
#     mistake again with a bigger table.
#   - PROTOCOL rule 5. Most of the product is uninteresting. The question is
#     whether EACH VARIABLE moves anything, which is a main-effects question,
#     and main effects are recoverable from a one-variable-at-a-time design at
#     full n per contrast.
#
# So: a fractional design around one default cell. The default is
# (user_suffix, imperative, count=1, io), because that is precisely the cell
# recipe_oracle already ran -- this matrix extends a measured point rather than
# floating next to it. Each other arm moves exactly ONE variable off that
# default, so any difference is attributable to that variable and to nothing
# else. Three placebo arms sit at the default and at each count, because the
# length confound grows with count and must be controlled where it grows. One
# placebo sits at the system point, because "text in the system role" is a
# plausible effect all by itself and would otherwise be credited to the recipe.
# One combination arm exists because a pure main-effects design cannot see an
# interaction at all, and if every main effect is null the most plausible
# interaction still deserves its single shot before the branch is called dead.
# ---------------------------------------------------------------------------
DEFAULT_ARMS = [
    {"name": "none", "point": None, "form": None, "count": 0,
     "why": "baseline: the unhinted prompt, the thing everything is measured "
            "against"},

    {"name": "base_suffix_imp1", "point": "user_suffix", "form": "imperative",
     "count": 1,
     "why": "the cell recipe_oracle already ran; the origin of this matrix"},
    {"name": "base_placebo1", "point": "user_suffix", "form": "imperative",
     "count": 1, "content": "placebo", "controls": "base_suffix_imp1",
     "why": "control for the default cell: same shape and length, no perf "
            "content"},

    # INJECTION POINT, everything else at the default.
    {"name": "point_system", "point": "system", "form": "imperative",
     "count": 1,
     "why": "same bytes as the default, in a system message"},
    {"name": "point_system_placebo", "point": "system", "form": "imperative",
     "count": 1, "content": "placebo", "controls": "point_system",
     "why": "separates 'a recipe in the system role' from 'any text in the "
            "system role'"},
    {"name": "point_prefix", "point": "user_prefix", "form": "imperative",
     "count": 1,
     "why": "same bytes as the default, before the problem instead of after; "
            "tests recency against primacy at zero cost difference"},
    {"name": "point_prefill", "point": "assistant_prefill",
     "form": "imperative", "count": 1,
     "why": "the model continues its own sentence rather than being told; the "
            "only arm where the hint is not an instruction at all"},

    # FORM, at the default point and count.
    {"name": "form_declarative", "point": "user_suffix", "form": "declarative",
     "count": 1,
     "why": "the same facts stated as true of the world rather than ordered"},
    {"name": "form_checklist", "point": "user_suffix", "form": "checklist",
     "count": 1,
     "why": "the same orders as a list; tests whether structure alone helps"},
    {"name": "form_fragment", "point": "user_suffix", "form": "fragment",
     "count": 1,
     "why": "one line of code, no prose; the cheapest arm in the matrix and "
            "the one most likely to be copied verbatim"},
    {"name": "form_question", "point": "user_suffix", "form": "question",
     "count": 1,
     "why": "asks without answering; if this works the corpus needs prompts, "
            "not recipes, which is a much cheaper thing to build"},
    {"name": "form_conditional", "point": "user_suffix", "form": "conditional",
     "count": 1,
     "why": "trigger before advice, so the model can check the trigger and "
            "drop the hint; its control is base_suffix_imp1 at the same cell, "
            "not a placebo -- the contrast is register, and the placebo at "
            "that cell already prices the act of adding text"},

    # COUNT, at the default point and form, each with its own control.
    {"name": "count_3", "point": "user_suffix", "form": "imperative",
     "count": 3,
     "why": "three recipes from three families; tests whether one was simply "
            "the wrong one"},
    {"name": "count_3_placebo", "point": "user_suffix", "form": "imperative",
     "count": 3, "content": "placebo", "controls": "count_3",
     "why": "the length confound grows with count, so the control grows with "
            "it"},
    {"name": "count_5", "point": "user_suffix", "form": "imperative",
     "count": 5,
     "why": "everything at once; also the arm most likely to exhaust the "
            "thinking budget, which is why that outcome is recorded apart"},
    {"name": "count_5_placebo", "point": "user_suffix", "form": "imperative",
     "count": 5, "content": "placebo", "controls": "count_5",
     "why": "control at the largest context cost in the matrix"},

    # MISROUTING. The pair that decides whether a lossy selector is viable.
    # Same problem, same position, same count, same WRONG recipe -- the two
    # arms differ in register and in nothing else, so a difference between them
    # is the escape the conditional register offers and cannot be anything
    # else. Read them against "none", not against the default cell: the
    # question is what a wrong hint costs compared to no hint at all.
    {"name": "wrong_imperative", "point": "user_suffix", "form": "imperative",
     "count": 1, "misroute": True,
     "why": "a mismatched recipe as an order, with no way for the model to "
            "decline it; the cost of a routing error in the register the "
            "corpus mostly uses today"},
    {"name": "wrong_conditional", "point": "user_suffix", "form": "conditional",
     "count": 1, "misroute": True,
     "why": "the SAME mismatched recipe with its trigger stated, so the model "
            "can observe the trigger is false and ignore it; if this costs "
            "less than wrong_imperative, the register buys robustness to "
            "routing error and a 73%-accurate tree becomes affordable"},

    # One interaction, chosen in advance rather than after seeing the results.
    {"name": "combo_system_checklist_3", "point": "system", "form": "checklist",
     "count": 3,
     "why": "the a-priori best guess (standing instructions, structured, "
            "covering three families); a main-effects design is blind to "
            "interactions and this is the one worth a single cell"},
]

ARM_BY_NAME = {a["name"]: a for a in DEFAULT_ARMS}


# ---------------------------------------------------------------------------
# Cost. "Minimal hint, do not balloon context" is a design constraint, and a
# constraint that is not measured is a hope.
# ---------------------------------------------------------------------------
def arm_cost(arm: dict, row: dict) -> dict:
    """Characters each arm adds over the unhinted prompt, split by role.

    Characters, not tokens, and deliberately so: pulling a tokenizer in here
    would add a dependency that must match the served model's vocabulary to be
    worth anything, and if it silently did not match, the numbers would look
    exactly as plausible as correct ones. est_tokens is chars/4, labelled as
    the estimate it is; the comparison BETWEEN arms is what this table is for
    and that ratio is stable across arms of similar prose.
    """
    base = base_messages(row)
    msgs = messages_for_arm(row, arm)
    def chars(ms, role): return sum(len(m["content"]) for m in ms
                                    if m["role"] == role)
    added = sum(len(m["content"]) for m in msgs) - \
        sum(len(m["content"]) for m in base)
    return {"arm": arm["name"], "added_chars": added,
            "est_tokens": round(added / 4),
            "system_chars": chars(msgs, "system"),
            "user_added": chars(msgs, "user") - chars(base, "user"),
            "assistant_chars": chars(msgs, "assistant"),
            "n_messages": len(msgs)}


def cost_table(row: dict, arms: list | None = None) -> list:
    return [arm_cost(a, row) for a in (arms or DEFAULT_ARMS)]


# ---------------------------------------------------------------------------
# Generation. One transport for every arm.
# ---------------------------------------------------------------------------
def generate_messages(cond: dict, messages: list, max_tokens: int,
                      timeout: int) -> tuple:
    """lcb.generate, but taking a messages array instead of one user string.

    lcb.generate hardcodes `messages=[{"role": "user", ...}]`, which cannot
    express three of the four injection points. Routing only SOME arms through
    it and the rest through a second path would put a transport difference
    inside the variable being measured, so every arm goes through this one.
    Body, headers, key and returned metadata are kept identical to
    lcb.generate's on purpose -- the dry run asserts the single-user-message
    case produces the same request body that lcb.generate would have sent.
    """
    body = {"model": cond["model"], "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.2}
    t0 = time.time()
    headers = {"Content-Type": "application/json"}
    if lcb._KEY:
        headers["Authorization"] = f"Bearer {lcb._KEY}"
    req = urllib.request.Request(f"{cond['url']}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    return text, {"ms": round((time.time() - t0) * 1000),
                  "finish": (d["choices"][0].get("finish_reason") or ""),
                  "empty_content": not (msg.get("content") or "").strip(),
                  "chars": len(text)}


def request_body(cond: dict, messages: list, max_tokens: int) -> dict:
    """The body generate_messages would send. Exists so the dry run can check
    it against lcb.generate's without a socket."""
    return {"model": cond["model"], "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.2}


def classify(meta: dict, code: str) -> str:
    """What actually happened, before any question of right or wrong.

    'budget_exhausted' is the documented failure of this thinking model: it
    spends the budget on reasoning and returns empty content with
    finish_reason 'length'. Note the test is on empty_content, not on the text
    -- lcb.generate substitutes reasoning_content into the text when content is
    empty, so a truncated reply is never an empty string by the time it gets
    here, and a check on the text would miss every one of them.
    """
    if meta.get("empty_content") and meta.get("finish") == "length":
        return "budget_exhausted"
    if not (code or "").strip():
        return "no_code"
    return "scored"


def make_record(row: dict, arm: dict, outcome: str, passed: bool, why: str,
                perf: dict, meta: dict) -> dict:
    """The row written to jsonl. Factored out so the DRY RUN can assert on it.

    A field that only exists on the live path is a field nobody has checked.
    injected_ids in particular: the misrouted arms are worthless without it,
    because "a wrong hint hurt" is not a finding until you can say which wrong
    hint, on which problem.
    """
    ids = arm_ids(row, arm)
    rec = {"question_id": row.get("question_id"), "arm": arm["name"],
           "point": arm["point"], "form": arm["form"], "count": arm["count"],
           "content": arm.get("content", "recipe"),
           "misroute": bool(arm.get("misroute")),
           "injected_ids": ids,
           "difficulty": row.get("difficulty"),
           "outcome": outcome, "passed": passed, "why": why,
           "added_chars": arm_cost(arm, row)["added_chars"], **perf, **meta}
    if arm.get("misroute"):
        # Whether the cue filter had to be relaxed for this row. Kept per row
        # rather than as a global count so the analysis can drop exactly the
        # affected pairs instead of the whole arm.
        rec["misroute_degraded"] = wrong_recipe_for(row)[1]
    return rec


# ---------------------------------------------------------------------------
# Mock generator: the dry run's model.
# ---------------------------------------------------------------------------
MOCK_CODE = ("Here is the program.\n\n```python\n"
             "import sys\n"
             "def main():\n"
             "    data = sys.stdin.buffer.read().split()\n"
             "    sys.stdout.write(str(len(data)))\n"
             "main()\n"
             "```\n")


def mock_generate(cond: dict, messages: list, max_tokens: int,
                  timeout: int) -> tuple:
    """A model that answers instantly and can be made to fail on purpose.

    It is rigged so the dry run exercises all three outcomes rather than only
    the happy one: the largest arm truncates the way the real model does, and
    one arm replies in prose with no code. A dry run in which everything
    succeeds proves only that the happy path exists.
    """
    # count_5 is the longest prompt in the matrix, and the real model's
    # truncation risk scales with prompt length, so that is where the mock
    # truncates.
    if _mock_marker(messages, "count_5"):
        return ("...reasoning that never reached an answer...",
                {"ms": 1, "finish": "length", "empty_content": True,
                 "chars": 44})
    if _mock_marker(messages, "no_code"):
        return ("I would approach this by considering the constraints.",
                {"ms": 1, "finish": "stop", "empty_content": False,
                 "chars": 53})
    return MOCK_CODE, {"ms": 1, "finish": "stop", "empty_content": False,
                       "chars": len(MOCK_CODE)}


_MOCK_FLAGS: dict = {}


def mock_key(messages: list) -> str:
    """Identity of a messages array, for rigging the mock at one exact arm.

    A hash of the WHOLE array. The first attempt keyed on the first 200
    characters, which every user-suffix arm shares because the problem comes
    first -- so one rigged arm silently rigged nine, and the dry run reported
    four scored outcomes where it should have reported fourteen. Exactly the
    class of bug this dry run exists to catch, caught on itself.
    """
    return hashlib.sha1(json.dumps(messages).encode()).hexdigest()


def _mock_marker(messages: list, flag: str) -> bool:
    return _MOCK_FLAGS.get(flag) == mock_key(messages)


# ---------------------------------------------------------------------------
# Dry run. Everything below runs without a GPU, a server or the dataset.
# ---------------------------------------------------------------------------
SYNTH_STDIN_ROW = {
    "question_id": "synthetic_stdin",
    "difficulty": "easy",
    "starter_code": "",
    "question_content": "Given N and then N integers, print their sum.",
    "public_test_cases": "[]", "private_test_cases": "", "metadata": {},
}
SYNTH_FUNC_ROW = {
    "question_id": "synthetic_functional",
    "difficulty": "medium",
    "starter_code": "class Solution:\n    def f(self, a: list) -> int:\n",
    "question_content": "Return the sum of the list.",
    "public_test_cases": "[]", "private_test_cases": "",
    "metadata": {"func_name": "f"},
}


def fixture_rows() -> list:
    """Real rows where the dataset is present, synthetic ones otherwise.

    PROTOCOL rule 7: a builder validated only against prompts written by the
    person who wrote the builder has not been validated. The synthetic rows are
    the fallback and are labelled as such in the output, never quietly
    substituted.
    """
    path = os.path.join(BENCH, "data", "test6.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:                                # noqa: BLE001
                    continue
    real_stdin = next((r for r in rows
                       if not (r.get("starter_code") or "").strip()), None)
    real_func = next((r for r in rows
                      if (r.get("starter_code") or "").strip()), None)
    out = []
    out.append((real_stdin or SYNTH_STDIN_ROW,
                "real stdin row" if real_stdin else "SYNTHETIC stdin row"))
    out.append((real_func or SYNTH_FUNC_ROW,
                "real functional row" if real_func else
                "SYNTHETIC functional row"))
    return out


def _all_rows() -> list:
    """Every problem in the dataset, or [] when it is not present.

    The misrouting checks run over all of them rather than a sample: the
    routing rule is cheap, the dataset is local, and "it worked on the two rows
    I looked at" is how a rule that degenerates on the third gets shipped
    (PROTOCOL rule 5 -- ask the question at the layer where n is free).
    """
    path = os.path.join(BENCH, "data", "test6.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:                                    # noqa: BLE001
                continue
    return rows


def emit_routing(limit: int) -> None:
    """Print the misrouting decision for the first `limit` rows, as JSON.

    Exists only so the dry run can re-run this in a FRESH interpreter under a
    different PYTHONHASHSEED and compare. Determinism checked inside one
    process is not determinism: str hashing is salted per process, so a rule
    keyed on hash() would look perfectly stable within a run and silently
    re-randomise on the next one -- and every misrouted result already written
    would belong to a pairing that no longer exists.
    """
    rows = _all_rows() or [SYNTH_STDIN_ROW, SYNTH_FUNC_ROW]
    print(json.dumps({r.get("question_id"): wrong_recipe_for(r)[0]
                      for r in rows[:limit]}))


def _routing_in_fresh_process(limit: int = 40) -> tuple:
    """(mapping, error). Runs THIS file with --emit-routing under a new salt.

    A subprocess, not an import: the point is a new interpreter with a
    different PYTHONHASHSEED, which is not reachable any other way. It
    generates nothing and touches no endpoint.
    """
    import subprocess
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = ("12345" if env.get("PYTHONHASHSEED") != "12345"
                             else "54321")
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--emit-routing", "--routing-limit", str(limit)],
                           capture_output=True, text=True, env=env,
                           timeout=120)
    except Exception as e:                                       # noqa: BLE001
        return {}, f"child process failed: {type(e).__name__}: {e}"
    if p.returncode != 0:
        return {}, f"child exited {p.returncode}: {p.stderr.strip()[-200:]}"
    try:
        return json.loads(p.stdout.strip().splitlines()[-1]), ""
    except Exception as e:                                       # noqa: BLE001
        return {}, f"unreadable child output: {type(e).__name__}: {e}"


class Checks:
    """Assertion bookkeeping that keeps going after the first failure.

    Stopping at the first failure hides how much else is broken, and this is
    the only place the construction gets checked at all.
    """

    def __init__(self) -> None:
        self.rows = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        return bool(ok)

    def report(self) -> int:
        print(f"\n{'=' * 78}")
        print(f"  DRY RUN  --  {len(self.rows)} assertions")
        print(f"{'=' * 78}")
        bad = 0
        for name, ok, detail in self.rows:
            bad += 0 if ok else 1
            mark = "ok  " if ok else "FAIL"
            print(f"  [{mark}] {name}")
            if detail:
                print(f"         {detail}")
        print(f"\n  {len(self.rows) - bad} passed, {bad} failed")
        return bad


def valid_messages(msgs) -> tuple:
    """Is this an OpenAI chat messages array a server would accept?"""
    if not isinstance(msgs, list) or not msgs:
        return False, "not a non-empty list"
    for m in msgs:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            return False, f"bad keys: {m if not isinstance(m, dict) else set(m)}"
        if m["role"] not in ("system", "user", "assistant"):
            return False, f"bad role {m['role']!r}"
        if not isinstance(m["content"], str) or not m["content"].strip():
            return False, f"empty content in {m['role']}"
    roles = [m["role"] for m in msgs]
    if roles.count("system") > 1:
        return False, "more than one system message"
    if "system" in roles and roles[0] != "system":
        return False, "system message is not first"
    if roles.count("user") != 1:
        return False, f"expected exactly one user message, got {roles.count('user')}"
    if "assistant" in roles and roles[-1] != "assistant":
        return False, "assistant message is not last"
    return True, ""


def dry_run() -> int:                                            # noqa: C901
    c = Checks()
    fixtures = fixture_rows()
    row, row_label = fixtures[0]
    func_row, func_label = fixtures[1]
    print(f"  fixtures: {row_label} ({row.get('question_id')}), "
          f"{func_label} ({func_row.get('question_id')})")

    # Snapshot everything shared, to prove later that nothing was mutated.
    row_snapshot = copy.deepcopy(row)
    recipes_snapshot = copy.deepcopy(RECIPES)
    base_snapshot = base_prompt(row)

    built = {}
    for arm in DEFAULT_ARMS:
        built[arm["name"]] = messages_for_arm(row, arm)

    # 1. every arm is a valid messages array, on both row shapes
    bad = []
    for arm in DEFAULT_ARMS:
        for r, lbl in ((row, "stdin"), (func_row, "functional")):
            ok, why = valid_messages(messages_for_arm(r, arm))
            if not ok:
                bad.append(f"{arm['name']}/{lbl}: {why}")
    c.check("1. every arm builds a valid OpenAI messages array "
            "(both stdin and functional rows)", not bad, "; ".join(bad))

    # 2. the system arm puts the hint in the system role and NOT in the user turn
    sysm = built["point_system"]
    hint_1 = render_hint(["io"], "imperative")
    c.check("2. system arm: hint is in a system message and absent from the "
            "user turn",
            sysm[0]["role"] == "system" and hint_1 in sysm[0]["content"]
            and hint_1 not in sysm[1]["content"]
            and sysm[1]["content"] == base_snapshot,
            f"roles={[m['role'] for m in sysm]}, "
            f"user turn identical to base: "
            f"{sysm[1]['content'] == base_snapshot}")

    # 3. assistant prefill produces a trailing assistant message
    pre = built["point_prefill"]
    c.check("3. prefill arm: last message is an assistant turn carrying the "
            "hint, and no system message appears",
            pre[-1]["role"] == "assistant" and hint_1 in pre[-1]["content"]
            and all(m["role"] != "system" for m in pre)
            and "```" not in pre[-1]["content"],
            f"roles={[m['role'] for m in pre]}, "
            f"tail={pre[-1]['content'][-24:]!r}, no open code fence")

    # 4. prefix/suffix put the same bytes on opposite sides of the problem
    pfx = built["point_prefix"][0]["content"]
    sfx = built["base_suffix_imp1"][0]["content"]
    c.check("4. prefix and suffix place an identical hint block on opposite "
            "sides of the problem, at identical character cost",
            pfx.index(hint_1) < pfx.index("### Question")
            and sfx.index(hint_1) > sfx.index("### Question")
            and len(pfx) == len(sfx),
            f"prefix len={len(pfx)}, suffix len={len(sfx)}")

    # 5. the default cell reproduces recipe_oracle byte for byte
    oracle_prompt = ro.prompt_for(row, BY_ID["io"])
    c.check("5. default cell is byte-identical to recipe_oracle's 'io' arm "
            "(the two experiments are comparable)",
            sfx == oracle_prompt,
            f"lengths {len(sfx)} vs {len(oracle_prompt)}")
    oracle_none = ro.prompt_for(row, None)
    c.check("6. the 'none' arm is byte-identical to the unhinted "
            "livecodebench prompt",
            built["none"][0]["content"] == oracle_none
            and len(built["none"]) == 1, "")

    # 7. placebo arms are within 25% of the arm they control for
    rows = []
    for arm in DEFAULT_ARMS:
        if not arm.get("controls"):
            continue
        real = arm_cost(ARM_BY_NAME[arm["controls"]], row)["added_chars"]
        plac = arm_cost(arm, row)["added_chars"]
        ratio = plac / real if real else 0.0
        rows.append((arm["name"], arm["controls"], real, plac, ratio))
    off = [f"{n} vs {ct}: {p} vs {r} chars ({ratio:.2f}x)"
           for n, ct, r, p, ratio in rows if not 0.75 <= ratio <= 1.25]
    c.check("7. every placebo arm's character cost is within 25% of the arm "
            "it controls for",
            not off,
            "; ".join(off) or "; ".join(
                f"{n} {ratio:.2f}x" for n, _ct, _r, _p, ratio in rows))

    # 8/9. count really injects that many recipes
    def injected(text: str, ids: list) -> int:
        return sum(1 for i in ids if TOPIC_TOKEN[i] in text.lower())
    t3 = built["count_3"][0]["content"]
    t5 = built["count_5"][0]["content"]
    t1 = sfx
    c.check("8. count=3 injects exactly 3 distinct recipes (and count=1 "
            "exactly 1)",
            injected(t3, COUNT_IDS[3]) == 3 and injected(t1, REAL_IDS) == 1,
            f"count_3 matched {injected(t3, COUNT_IDS[3])}/3, "
            f"count_1 matched {injected(t1, REAL_IDS)}/1")
    c.check("9. count=5 injects all 5 recipes",
            injected(t5, REAL_IDS) == 5,
            f"matched {injected(t5, REAL_IDS)}/5")

    # 10. nothing is mutated: not the row, not RECIPES, not a returned array
    first = build_messages(row, "user_suffix", "imperative", 1)
    first[0]["content"] = "CLOBBERED"
    first.append({"role": "user", "content": "CLOBBERED"})
    second = build_messages(row, "user_suffix", "imperative", 1)
    c.check("10. no arm mutates the shared base prompt, the row, or RECIPES "
            "(a clobbered return value does not reach the next build)",
            second[0]["content"] == oracle_prompt
            and base_prompt(row) == base_snapshot
            and row == row_snapshot and RECIPES == recipes_snapshot, "")

    # 11. content parity: every wide form of every recipe keeps the mechanism
    lost = []
    for rid in REAL_IDS:
        for form in WIDE_FORMS:
            if TOPIC_TOKEN[rid] not in render_hint([rid], form).lower():
                lost.append(f"{rid}/{form}")
    c.check("11. every wide form of every recipe still names the mechanism "
            "(form varies, content does not)", not lost, "; ".join(lost))

    # 12. the narrow forms are narrow, and the question form asks
    q = render_hint(["io"], "question")
    frag = render_hint(["io"], "fragment")
    c.check("12. narrow forms are shorter than wide ones and the question "
            "form asks rather than tells",
            len(frag) < len(render_hint(["io"], "checklist"))
            and len(q) < len(render_hint(["io"], "imperative"))
            and q.rstrip().endswith("?") and "\n" not in frag,
            f"fragment {len(frag)}ch (1 line), question {len(q)}ch")

    # 13. arm names are unique and every variable level is covered
    names = [a["name"] for a in DEFAULT_ARMS]
    pts = {a["point"] for a in DEFAULT_ARMS if a["point"]}
    fms = {a["form"] for a in DEFAULT_ARMS if a["form"]}
    cts = {a["count"] for a in DEFAULT_ARMS if a["count"]}
    c.check("13. arm names unique, and every level of every variable appears "
            "at least once",
            len(set(names)) == len(names)
            and pts == set(INJECTION_POINTS) and fms == set(FORMS)
            and cts == set(COUNTS),
            f"points {sorted(pts)}; forms {sorted(fms)}; counts {sorted(cts)}")

    # 14. the transport matches livecodebench's for the single-message case
    cond = lcb.CONDITIONS["direct"]
    mine = request_body(cond, built["none"], 8000)
    theirs = {"model": cond["model"],
              "messages": [{"role": "user", "content": oracle_none}],
              "max_tokens": 8000, "temperature": 0.2}
    c.check("14. generate_messages sends the same request body lcb.generate "
            "would for a plain single-user-message arm",
            mine == theirs, "")

    # 15. mock end-to-end: every arm produces a recorded outcome, and the
    #     truncation and no-code cases are recorded apart from a wrong answer
    _MOCK_FLAGS.clear()
    _MOCK_FLAGS["count_5"] = mock_key(built["count_5"])
    _MOCK_FLAGS["no_code"] = mock_key(built["form_question"])
    outcomes = {}
    for arm in DEFAULT_ARMS:
        msgs = messages_for_arm(row, arm)
        text, meta = mock_generate(cond, msgs, 8000, 60)
        outcomes[arm["name"]] = classify(meta, lcb.extract_code(text))
    _MOCK_FLAGS.clear()
    c.check("15. mock dry run: every arm yields an explicit outcome, with "
            "budget_exhausted and no_code recorded apart from a wrong answer",
            set(outcomes.values()) == {"scored", "budget_exhausted", "no_code"}
            and outcomes["count_5"] == "budget_exhausted"
            and outcomes["form_question"] == "no_code"
            and all(v == "scored" for k, v in outcomes.items()
                    if k not in ("count_5", "form_question")),
            f"{sum(1 for v in outcomes.values() if v == 'scored')} scored, "
            f"1 budget_exhausted, 1 no_code")

    # 16. the conditional register renders for every recipe AND every placebo
    missing = []
    for rid in list(FACETS) + list(PLACEBO_FACETS):
        txt = render_hint([rid], "conditional")
        if not txt.strip() or txt == render_hint([rid], "imperative"):
            missing.append(rid)
    c.check("16. conditional renders for all 5 recipes and all 5 placebos, "
            "and is not just the imperative text",
            not missing and len(list(FACETS) + list(PLACEBO_FACETS)) == 10,
            "; ".join(missing))

    # 17. conditional keeps the mechanism (it is a wide form, so check 11
    #     already covers it; named separately because it is a new register and
    #     a silent content drift here would invalidate the misrouting result)
    drift = [rid for rid in REAL_IDS
             if TOPIC_TOKEN[rid] not in render_hint([rid], "conditional").lower()]
    c.check("17. conditional says the same thing the other wide forms say "
            "(content parity across registers)", not drift, "; ".join(drift))

    # 18. it states a trigger BEFORE the advice, or it is just a long order
    bad_trig = []
    for rid in list(FACETS) + list(PLACEBO_FACETS):
        for sent in render_hint([rid], "conditional").split(". "):
            s = sent.strip()
            if not s:
                continue
            head = s.split()[0].lower().strip(",")
            if head not in TRIGGER_WORDS:
                bad_trig.append(f"{rid}: {s[:40]!r}")
            elif "," not in s and ";" not in s:
                bad_trig.append(f"{rid}: no clause break in {s[:40]!r}")
    c.check("18. every conditional sentence opens with a trigger word and "
            "separates the trigger from the advice",
            not bad_trig, "; ".join(bad_trig[:3]))

    # 19. a misrouted arm never injects the recipe that fits the problem
    pool_rows = _all_rows() or [row, func_row]
    wrong_bad, degraded = [], 0
    for r in pool_rows:
        wid, deg = wrong_recipe_for(r)
        degraded += 1 if deg else 0
        q = (r.get("question_content") or "").lower()
        if wid in COUNT_IDS[1] or wid in MISROUTE_EXCLUDED:
            wrong_bad.append(f"{r.get('question_id')}: picked {wid}")
        elif not deg and any(cue in q for cue in MISROUTE_CUES[wid]):
            wrong_bad.append(f"{r.get('question_id')}: {wid} is cued in the "
                             f"statement")
    c.check("19. over every problem in the dataset, the misrouted arm never "
            "injects the right-hint recipe nor a recipe cued by the statement",
            not wrong_bad,
            f"{len(pool_rows)} rows checked, {degraded} degraded "
            f"({degraded / max(len(pool_rows), 1):.0%} had a cue for every "
            f"candidate and are flagged in the record)")

    # 20. the mismatch is reproducible -- within the process AND in a fresh
    #     interpreter with a different hash seed
    once = [wrong_recipe_for(r)[0] for r in pool_rows[:40]]
    random.seed(1)
    twice = [wrong_recipe_for(r)[0] for r in pool_rows[:40]]
    random.seed(999)
    thrice = [wrong_recipe_for(r)[0] for r in pool_rows[:40]]
    child, child_err = _routing_in_fresh_process()
    mine = {r.get("question_id"): wrong_recipe_for(r)[0]
            for r in pool_rows[:40]}
    c.check("20. misrouting is deterministic: same within the process under "
            "different RNG seeds, and same in a fresh interpreter with "
            "PYTHONHASHSEED changed",
            once == twice == thrice and child == mine and not child_err,
            child_err or f"{len(mine)} rows agreed across processes")

    # 21. the injected wrong recipe id reaches the result record
    wrec = make_record(row, ARM_BY_NAME["wrong_imperative"], "scored", True,
                       "", {"max_s": 0.1, "total_s": 0.1, "peak_kb": 1},
                       {"ms": 1, "finish": "stop", "empty_content": False,
                        "chars": 10})
    c.check("21. the result record names the wrong recipe that was injected, "
            "and flags whether the cue filter was relaxed",
            wrec["injected_ids"] == [wrong_recipe_for(row)[0]]
            and wrec["misroute"] is True
            and "misroute_degraded" in wrec
            and wrec["injected_ids"][0] not in COUNT_IDS[1],
            f"injected_ids={wrec['injected_ids']}, "
            f"degraded={wrec['misroute_degraded']}")

    # 22. the two misrouted arms differ in REGISTER and in nothing else
    wi = arm_ids(row, ARM_BY_NAME["wrong_imperative"])
    wc = arm_ids(row, ARM_BY_NAME["wrong_conditional"])
    a_i, a_c = ARM_BY_NAME["wrong_imperative"], ARM_BY_NAME["wrong_conditional"]
    c.check("22. wrong_imperative and wrong_conditional inject the SAME wrong "
            "recipe at the same point and count (a pure register contrast)",
            wi == wc and a_i["point"] == a_c["point"]
            and a_i["count"] == a_c["count"]
            and a_i["form"] != a_c["form"],
            f"both inject {wi}")

    # 23. cost is bounded: the constraint is a minimal hint.
    #
    # The conditional register gets a higher ceiling, NAMED rather than waived.
    # It costs more because it says more -- trigger as well as advice -- and
    # that extra clause is the whole product being tested. The first run of
    # this check failed at 423 chars against a flat 400 ceiling, which is the
    # check doing its job: the number is now stated, bounded, and in the table
    # next to whatever benefit the register turns out to buy. At count=1 the
    # conditional register costs 2.1x the imperative one; a corpus rewritten
    # this way at count=5 would be roughly 1,800 characters of hint, and that
    # is the figure to weigh against "do not balloon context".
    ceiling = {"conditional": 460}
    costs = {r["arm"]: r["added_chars"] for r in cost_table(row)}
    over = []
    for k, v in costs.items():
        a = ARM_BY_NAME[k]
        cap = ceiling.get(a["form"], 400)
        if (a["count"] == 1 and v > cap) or v > 1400:
            over.append(f"{k}={v} (cap {cap})")
    imp1 = costs["base_suffix_imp1"]
    con1 = costs["form_conditional"]
    c.check("23. 'minimal hint' is enforced, not hoped for: count=1 arms add "
            "<400 chars (<460 for the conditional register, which states a "
            "trigger as well), no arm adds >1400",
            not over, "; ".join(over) or
            f"max {max(costs.values())} chars "
            f"(~{max(costs.values()) // 4} tokens); conditional costs "
            f"{con1 / imp1:.1f}x the imperative register at count=1")

    print(f"\n  {'arm':<26}{'point':<19}{'form':<13}{'n':<4}"
          f"{'added':<8}{'~tok':<7}{'where':<10}")
    print("  " + "-" * 88)
    for r in cost_table(row):
        a = ARM_BY_NAME[r["arm"]]
        where = ("sys" if r["system_chars"] else
                 "asst" if r["assistant_chars"] else
                 "user" if r["added_chars"] else "-")
        print(f"  {r['arm']:<26}{str(a['point'] or '-'):<19}"
              f"{str(a['form'] or '-'):<13}{a['count']:<4}"
              f"{r['added_chars']:<8}{r['est_tokens']:<7}{where:<10}")

    print("\n  placebo matching (added chars, placebo / real)")
    for n, ct, r, p, ratio in rows:
        print(f"    {n:<26} {p:>5} / {r:<5} = {ratio:.2f}x")

    wid, wdeg = wrong_recipe_for(row)
    print(f"\n  the two misrouted arms' costs above are for THIS row only: "
          f"they inject\n  {wid!r} here (degraded={wdeg}) and a different "
          f"recipe elsewhere, so their cost\n  varies by problem while every "
          f"other arm's is fixed. Same recipe in both,\n  so the register "
          f"contrast between them is still clean.")

    return c.report()


# ---------------------------------------------------------------------------
# Live run. Not reachable without --live, and it checks the endpoint is alive
# before spending anything (PROTOCOL rule 1).
# ---------------------------------------------------------------------------
def run_live(args) -> None:
    lcb._KEY = lcb.api_key()
    cond = lcb.CONDITIONS[args.endpoint]
    try:
        _t, meta = generate_messages(
            cond, [{"role": "user", "content": "Reply with the word ready."}],
            64, 120)
        print(f"  {args.endpoint} alive, {meta['ms']}ms")
    except Exception as e:                                       # noqa: BLE001
        raise SystemExit(f"  {args.endpoint} ({cond['url']}) not answering: "
                         f"{type(e).__name__}: {e}\n"
                         f"  Refusing to run: a dead endpoint scores 0% on "
                         f"every arm and reads as a result.")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rnd = random.Random(args.seed)
    picked = []
    for diff in ("easy", "medium", "hard"):
        g = [r for r in rows if r.get("difficulty") == diff]
        rnd.shuffle(g)
        picked += g[:max(1, args.n // 3)]
    rnd.shuffle(picked)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["question_id"], r["arm"]))
            except Exception:                                    # noqa: BLE001
                continue

    print(f"  {len(picked)} problems x {len(DEFAULT_ARMS)} arms = "
          f"{len(picked) * len(DEFAULT_ARMS)} generations, "
          f"{len(done)} already done")

    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(picked):
            qid = row["question_id"]
            cases = lcb.all_cases(row, args.max_cases)
            for arm in DEFAULT_ARMS:
                if (qid, arm["name"]) in done:
                    continue
                msgs = messages_for_arm(row, arm)
                try:
                    text, meta = generate_messages(cond, msgs, args.max_tokens,
                                                   args.gen_timeout)
                except Exception as e:                           # noqa: BLE001
                    out.write(json.dumps({
                        "question_id": qid, "arm": arm["name"],
                        "outcome": "error",
                        "error": f"{type(e).__name__}: {e}"}) + "\n")
                    out.flush()
                    continue
                code = lcb.extract_code(text)
                outcome = classify(meta, code)
                if outcome == "scored":
                    passed, why, perf = lcb.run_tests(code, row, cases,
                                                      args.test_timeout)
                else:
                    # Not a wrong answer. Recorded, counted, and kept out of
                    # pass@1 rather than scored as a failure.
                    passed, why, perf = False, outcome, {
                        "max_s": 0.0, "total_s": 0.0, "peak_kb": 0}
                rec = make_record(row, arm, outcome, passed, why, perf, meta)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"  [{i + 1}/{len(picked)}] {qid:<12} "
                      f"{arm['name']:<26} {outcome:<17} "
                      f"{'PASS' if passed else 'fail'} {why[:28]}", flush=True)

    report(args.out)


def report(path: str) -> None:
    """pass@1 per arm, truncation counted apart, and each contrast paired.

    Every comparison is against the DEFAULT cell or against the arm's own
    placebo, never arm-against-arm across two variables at once: two arms that
    differ in two variables cannot attribute a difference to either.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:                                    # noqa: BLE001
                continue
    by: dict = {}
    for r in rows:
        by.setdefault(r["question_id"], {})[r["arm"]] = r
    names = [a["name"] for a in DEFAULT_ARMS]
    complete = {q: v for q, v in by.items()
                if all(n in v for n in names)
                and all(v[n].get("outcome") == "scored" for n in names)}
    n = len(complete)

    print(f"\n{'=' * 82}")
    print(f"  HINT FORM MATRIX  --  {n} problems scored on all "
          f"{len(names)} arms")
    print(f"{'=' * 82}")
    kinds: dict = {}
    for r in rows:
        if r.get("outcome") != "scored":
            kinds[(r["arm"], r.get("outcome"))] = \
                kinds.get((r["arm"], r.get("outcome")), 0) + 1
    if kinds:
        print("  non-scored generations, excluded from pass@1 rather than "
              "counted as wrong:")
        for (a, k), v in sorted(kinds.items(), key=lambda x: -x[1]):
            print(f"    {a:<26} {str(k):<18} {v}")
    if not n:
        print("  nothing complete yet")
        return

    print(f"\n  {'arm':<26}{'pass@1':<24}{'median slowest':<17}"
          f"{'added chars':<12}")
    print("  " + "-" * 80)
    for nm in names:
        k = sum(1 for v in complete.values() if v[nm]["passed"])
        lo, hi = lcb.wilson(k, n)
        ps = [v[nm]["max_s"] for v in complete.values() if v[nm]["passed"]]
        ms = statistics.median(ps) if ps else 0.0
        add = next(iter(complete.values()))[nm].get("added_chars", 0)
        print(f"  {nm:<26}{k}/{n} = {k / n:5.1%} [{lo:.0%}-{hi:.0%}]".ljust(52)
              + f"{ms:<17.3f}{add:<12}")

    def contrast(a: str, b: str, label: str) -> None:
        bb = sum(1 for v in complete.values()
                 if v[b]["passed"] and not v[a]["passed"])
        cc = sum(1 for v in complete.values()
                 if v[a]["passed"] and not v[b]["passed"])
        p = lcb.mcnemar(bb, cc)
        print(f"    {label:<44} +{bb} -{cc}  p={p:.4f}"
              + ("" if bb + cc >= 10 else
                 f"   [{bb + cc} discordant: underpowered]"))

    print("\n  against the default cell (one variable moved, nothing else)")
    for nm in names:
        a = ARM_BY_NAME[nm]
        # Misrouted arms are excluded here and get their own block:
        # wrong_conditional differs from the default cell in TWO variables
        # (register and routing), and a contrast across two variables cannot
        # be attributed to either.
        if nm in ("none", "base_suffix_imp1") or a.get("controls") \
                or a.get("misroute"):
            continue
        contrast("base_suffix_imp1", nm, nm)
    print("\n  against no hint at all")
    contrast("none", "base_suffix_imp1", "default cell vs none")
    print("\n  against the matched placebo -- the only comparison that "
          "separates\n  a recipe from the act of adding text")
    for a in DEFAULT_ARMS:
        if a.get("controls"):
            contrast(a["name"], a["controls"],
                     f"{a['controls']} vs its placebo")

    # The misrouting block, read against NO HINT rather than against the
    # default cell: the question is what a wrong hint costs compared to not
    # having one, which is the cost the selector's error rate multiplies.
    print("\n  MISROUTING -- the cost of a wrong hint, and whether the "
          "conditional\n  register refunds it")
    contrast("base_suffix_imp1", "wrong_imperative",
             "right recipe vs wrong recipe (both imperative)")
    contrast("none", "wrong_imperative", "none vs wrong_imperative")
    contrast("none", "wrong_conditional", "none vs wrong_conditional")
    contrast("wrong_imperative", "wrong_conditional",
             "wrong_imperative vs wrong_conditional")
    deg = sum(1 for r in rows if r.get("misroute_degraded"))
    if deg:
        print(f"    {deg} misrouted records had the cue filter relaxed and "
              f"may have received a\n    relevant recipe; drop them before "
              f"quoting the misrouting numbers")
    print("    READ THESE THE OTHER WAY ROUND. A null here is a GOOD result:")
    print("    it means a wrong hint is close to free, so the selector's "
          "accuracy is not\n    the binding constraint. Only a significant "
          "LOSS against 'none' threatens the\n    design, and a loss on "
          "wrong_imperative that wrong_conditional does not share\n    is the "
          "case for rewriting the corpus in the conditional register.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Injection-form matrix for minimal performance hints. "
                    "Dry run by default; --live is the only thing that "
                    "touches a GPU.")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--live", action="store_true",
                    help="actually generate (hours of GPU); off by default")
    ap.add_argument("--show", metavar="ARM",
                    help="print one arm's messages array and exit")
    ap.add_argument("--emit-routing", action="store_true",
                    help="print the misrouting decisions as JSON and exit "
                         "(used by the dry run's cross-process determinism "
                         "check)")
    ap.add_argument("--routing-limit", type=int, default=40)
    ap.add_argument("--data", default=os.path.join(BENCH, "data", "test6.jsonl"))
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--endpoint", default="direct",
                    choices=list(lcb.CONDITIONS))
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--gen-timeout", type=int, default=900)
    ap.add_argument("--test-timeout", type=float, default=12.0)
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=os.path.join(HERE, "hint_form_results.jsonl"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.emit_routing:
        emit_routing(args.routing_limit)
        return
    if args.show:
        row = fixture_rows()[0][0]
        print(json.dumps(messages_for_arm(row, ARM_BY_NAME[args.show]),
                         indent=2))
        return
    if args.report_only:
        report(args.out)
        return
    if args.live:
        run_live(args)
        return
    raise SystemExit(dry_run())


if __name__ == "__main__":
    main()
