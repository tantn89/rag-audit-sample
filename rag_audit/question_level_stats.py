"""Question-level robustness of a before/after gain: a sign test and a bootstrap CI.

regressgate's permutation test shuffles trials, so it rules out run-to-run (sampling) noise for
these 67 questions. These two statistics treat the question as the unit instead:

- sign test: of the questions whose pass rate changed, how many went up; the exact one-sided
  binomial p-value for at least that many under "up and down are equally likely";
- bootstrap: resample the 67 questions with replacement (seeded, so the interval is the same on
  every run) and take the 2.5th and 97.5th percentiles of the mean per-question change.

Both still describe this question set; neither makes it a sample of real user questions.
"""

from __future__ import annotations

import random
import statistics
from math import comb

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20251210  # fixed: the published interval must be reproducible


def sign_test(deltas: list[float]) -> dict:
    """Exact one-sided sign test that changes are more often up than down (ties dropped)."""
    up = sum(d > 0 for d in deltas)
    down = sum(d < 0 for d in deltas)
    n = up + down
    p = sum(comb(n, k) for k in range(up, n + 1)) / 2 ** n if n else 1.0
    return {"up": up, "down": down, "unchanged": len(deltas) - n, "p_one_sided": p}


def bootstrap_ci(deltas: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED, level: float = 0.95) -> dict:
    """Percentile bootstrap CI of the mean per-question change."""
    if not deltas:
        raise ValueError("bootstrap_ci needs at least one question")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples))
    tail = (1 - level) / 2
    lo = means[int(tail * resamples)]
    hi = means[min(resamples - 1, int((1 - tail) * resamples))]
    return {"mean": statistics.fmean(deltas), "low": lo, "high": hi, "level": level,
            "resamples": resamples, "seed": seed}


def per_question_deltas(base: dict[str, list[bool]], head: dict[str, list[bool]]) -> dict[str, float]:
    """Change in pass rate per question present in both runs (trial outcomes by question id)."""
    return {cid: statistics.fmean(head[cid]) - statistics.fmean(base[cid])
            for cid in sorted(base) if cid in head and base[cid] and head[cid]}


def question_level(base: dict[str, list[bool]], head: dict[str, list[bool]]) -> dict:
    deltas = per_question_deltas(base, head)
    values = list(deltas.values())
    return {"questions": len(values), "sign_test": sign_test(values),
            "bootstrap_ci_of_gain": bootstrap_ci(values),
            "dropped": sorted(cid for cid, d in deltas.items() if d < 0)}
