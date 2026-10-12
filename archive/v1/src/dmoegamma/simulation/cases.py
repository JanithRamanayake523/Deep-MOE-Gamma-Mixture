"""The base paper's exact simulation designs (Tables 1 and 2 therein).

Vaidyanathan & Vani Lakshmi (2017), Sri Lankan J. Appl. Stat. 17(3).
Cases 1-3: two-component bivariate mixtures with high / low / medium overlap
(equal mean vectors / equal component-wise scales / equal variances,
respectively). Case 4: three-component bivariate mixture. The paper simulates
exactly n_i observations from component i (fixed counts) for 500 runs.

Published misclassification rates (Table 5 therein):
    Case 1: 0.4988, Case 2: 0.1155, Case 3: 0.1584, Case 4: 0.0550
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..params import Theta


@dataclass(frozen=True)
class PaperCase:
    name: str
    theta: Theta
    counts: tuple[int, ...]          # fixed per-component sample sizes
    published_misclassification: float
    description: str

    @property
    def n(self) -> int:
        return int(sum(self.counts))


PAPER_CASES: dict[str, PaperCase] = {
    # Case 1: high overlap; components have the same mean vector
    #   means: (10*3.5, 2.1*2) = (35, 4.2) and (8.75*4, 3.5*1.2) = (35, 4.2)
    "case1": PaperCase(
        name="case1",
        theta=Theta(
            pi=[0.50, 0.50],
            alpha=[[10.00, 2.10], [8.75, 3.50]],
            beta=[[3.50, 2.00], [4.00, 1.20]],
        ),
        counts=(50, 50),
        published_misclassification=0.4988,
        description="Two-component bivariate, high overlap, equal mean vectors",
    ),
    # Case 2: low overlap; component-wise scale parameters equal
    "case2": PaperCase(
        name="case2",
        theta=Theta(
            pi=[0.50, 0.50],
            alpha=[[3.20, 6.70], [3.60, 1.50]],
            beta=[[2.30, 2.30], [1.90, 1.90]],
        ),
        counts=(25, 25),
        published_misclassification=0.1155,
        description="Two-component bivariate, low overlap, equal scales",
    ),
    # Case 3: medium overlap; equal variance-covariance
    #   variances: (9*1, 4*1) = (9, 4) and (1*9, 1*4) = (9, 4)
    "case3": PaperCase(
        name="case3",
        theta=Theta(
            pi=[0.80, 0.20],
            alpha=[[9.00, 4.00], [1.00, 1.00]],
            beta=[[1.00, 1.00], [3.00, 2.00]],
        ),
        counts=(80, 20),
        published_misclassification=0.1584,
        description="Two-component bivariate, medium overlap, equal variances",
    ),
    # Case 4: three-component bivariate
    "case4": PaperCase(
        name="case4",
        theta=Theta(
            pi=[0.15, 0.15, 0.70],
            alpha=[[4.20, 1.50], [1.70, 6.90], [9.80, 2.50]],
            beta=[[2.00, 2.00], [5.00, 5.00], [3.60, 1.00]],
        ),
        counts=(30, 30, 140),
        published_misclassification=0.0550,
        description="Three-component bivariate",
    ),
}
