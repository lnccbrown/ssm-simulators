"""Experiment 1 group-average parameter estimates digitised from the paper.

Figures 7 (DK = 1 - d/b and B = b - A per accumulator) and 8 (match /
mismatch mean rates). Values read off bar heights, resolution ~0.005 (DK) and
~0.02 (B, v). Panel *titles* are used for the error-cost level; the Fig. 7
caption swaps (a)/(b) relative to the titles, and the titles agree with the
text (DK larger under high cost; low-cost DK-left larger for speed).
A, the matching-accumulator sv, and t0 are only in the supplement (not
available); sv of the mismatching accumulator is fixed at 1 by the paper.
"""

DK = {  # (cost, emphasis) -> (left, right)
    ("high", "accuracy"): (0.222, 0.122),
    ("high", "speed"): (0.224, 0.111),
    ("low", "accuracy"): (0.122, 0.109),
    ("low", "speed"): (0.172, 0.055),
}
B = {
    ("high", "accuracy"): (2.90, 3.05),
    ("high", "speed"): (2.25, 2.38),
    ("low", "accuracy"): (3.43, 3.72),
    ("low", "speed"): (2.81, 2.84),
}
V = {  # (cost, emphasis, similarity) -> (match, mismatch)
    ("high", "accuracy", "lower"): (3.55, 2.99),
    ("high", "accuracy", "higher"): (3.58, 2.93),
    ("high", "speed", "lower"): (3.33, 2.80),
    ("high", "speed", "higher"): (3.27, 2.80),
    ("low", "accuracy", "lower"): (3.31, 2.63),
    ("low", "accuracy", "higher"): (3.26, 2.63),
    ("low", "speed", "lower"): (3.17, 2.64),
    ("low", "speed", "higher"): (3.17, 2.69),
}
SV_MISMATCH = 1.0
