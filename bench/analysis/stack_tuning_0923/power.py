"""Exact McNemar power for "stack on for class X", paired, one sample per arm
per question. Model: each question is discordant with probability d; a
discordant question favours the stack with probability pi. Power = P(exact
two-sided McNemar p < alpha). Pure arithmetic, no data."""
from math import comb


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def power(n, d, pi, alpha):
    tot = 0.0
    for D in range(n + 1):
        pD = comb(n, D) * d ** D * (1 - d) ** (n - D)
        if pD < 1e-12:
            continue
        for b in range(D + 1):
            if mcnemar_p(b, D - b) < alpha:
                tot += pD * comb(D, b) * pi ** b * (1 - pi) ** (D - b)
    return tot


def n_for(d, pi, alpha, target=0.8, nmax=600):
    for n in range(5, nmax):
        if power(n, d, pi, alpha) >= target:
            return n
    return None


if __name__ == "__main__":
    print("min discordant pairs, all one way: alpha .05 ->",
          next(k for k in range(1, 20) if mcnemar_p(k, 0) < .05),
          "; alpha .025 ->", next(k for k in range(1, 20) if mcnemar_p(k, 0) < .025))
    print("observed 3-0:", mcnemar_p(3, 0), " 5 vs 1:", round(mcnemar_p(5, 1), 3))
    print("\nn per class for 80% power")
    print(f"{'d':>5} {'pi':>5} {'net gain':>9} {'a=.05':>7} {'a=.025':>7}")
    for d in (0.30, 0.38):
        for pi in (0.65, 0.75, 0.85, 0.95):
            print(f"{d:>5} {pi:>5} {d*(2*pi-1):>9.3f} {str(n_for(d, pi, .05)):>7} {str(n_for(d, pi, .025)):>7}")
    print("\npower at the clean held-out sizes (completion 37, generation 60), alpha .025")
    for n in (37, 60, 97):
        print(n, " ".join(f"d={d},pi={pi}: {power(n, d, pi, .025):.2f}" for d in (0.30, 0.38) for pi in (0.75, 0.85)))
