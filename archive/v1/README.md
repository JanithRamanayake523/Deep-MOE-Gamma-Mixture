# Archive: v1 ("Deep MoE for Gamma Mixture Parameter Estimation")

This folder keeps the **superseded first version** of the project for the record.
It is **not maintained, not imported by the current package, and not runnable
as-is**: the package `__init__` files and shared modules it relied on have since
been rewritten for v2 (`src/aemnet`).

| Path | What it was |
|---|---|
| `RESEARCH_METHODOLOGY_v1.md` | the v1 research design (an AI-assisted draft; see the integrity note in the main methodology) |
| `src/dmoegamma/` | v1 modules: `Theta` parameter container, DeepSets encoder + gate + per-slot expert heads (`models/moe.py`), composite loss, v1 trainer, gamma-only sampler / prior, v1 baselines (ECM, MLE, MoM, random-walk MCMC), v1 benchmark runner |
| `configs/`, `scripts/`, `tests/` | v1 training / Study 1-2 configs, CLIs and tests |
| `paper_raw_extract.txt` | raw text extraction of the base paper (the clean copy is in `references/`) |

Why v1 was replaced (details: main `RESEARCH_METHODOLOGY.md`, section 0):

1. Its "mixture of experts" was hollow: experts saw only a global embedding,
   responsibilities never reached them, and the heads were not equivariant to
   component order.
2. Amortisation was unmotivated for a tractable likelihood, and the main
   comparator (WH-MCLUST) was weak.

What survived into v2:

- the WH-MCLUST baseline, now `src/aemnet/baselines/wh_mclust.py`;
- the base-paper cases, now `src/aemnet/simulation/paper_cases.py` and E9;
- the weighted gamma MLE, now in `src/aemnet/simulation/families/gamma.py`;
- the v1 architecture idea, as the ablation `configs/train/baselines/v1style_gamma_k2.yaml`.

Two v1 bugs were fixed before archiving: `torch.nanmin` (which does not exist)
in the encoder, and a wrong `expand` in the composite loss.
