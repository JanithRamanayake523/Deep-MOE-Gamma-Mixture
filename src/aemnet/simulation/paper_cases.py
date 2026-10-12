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

from ..mixture_params import MixtureParams


@dataclass(frozen=True)
class PaperCase:
    name: str
    params: MixtureParams
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
        params=MixtureParams([0.50, 0.50], {"alpha": np.array([[10.00, 2.10], [8.75, 3.50]]), "beta": np.array([[3.50, 2.00], [4.00, 1.20]])}),
        counts=(50, 50),
        published_misclassification=0.4988,
        description="Two-component bivariate, high overlap, equal mean vectors",
    ),
    # Case 2: low overlap; component-wise scale parameters equal
    "case2": PaperCase(
        name="case2",
        params=MixtureParams([0.50, 0.50], {"alpha": np.array([[3.20, 6.70], [3.60, 1.50]]), "beta": np.array([[2.30, 2.30], [1.90, 1.90]])}),
        counts=(25, 25),
        published_misclassification=0.1155,
        description="Two-component bivariate, low overlap, equal scales",
    ),
    # Case 3: medium overlap; equal variance-covariance
    #   variances: (9*1, 4*1) = (9, 4) and (1*9, 1*4) = (9, 4)
    "case3": PaperCase(
        name="case3",
        params=MixtureParams([0.80, 0.20], {"alpha": np.array([[9.00, 4.00], [1.00, 1.00]]), "beta": np.array([[1.00, 1.00], [3.00, 2.00]])}),
        counts=(80, 20),
        published_misclassification=0.1584,
        description="Two-component bivariate, medium overlap, equal variances",
    ),
    # Case 4: three-component bivariate
    "case4": PaperCase(
        name="case4",
        params=MixtureParams([0.15, 0.15, 0.70], {"alpha": np.array([[4.20, 1.50], [1.70, 6.90], [9.80, 2.50]]), "beta": np.array([[2.00, 2.00], [5.00, 5.00], [3.60, 1.00]])}),
        counts=(30, 30, 140),
        published_misclassification=0.0550,
        description="Three-component bivariate",
    ),
}


def sample_case(case: PaperCase, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The paper's design: exactly counts[i] observations from component i.
    Returns X (n, 2), events (all ones), labels (n,)."""
    from .families import get_family

    fam = get_family("gamma", k=case.params.comps["alpha"].shape[1])
    rng = np.random.default_rng(rng)
    labels = np.repeat(np.arange(case.params.g), case.counts)
    X = np.concatenate([fam.sample_component(rng, case.params.component(i), c) for i, c in enumerate(case.counts)])
    return X, np.ones_like(X), labels
