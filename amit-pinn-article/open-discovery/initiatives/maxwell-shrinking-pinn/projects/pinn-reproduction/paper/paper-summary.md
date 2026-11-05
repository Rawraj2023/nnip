# Paper summary — provenance record

**Title:** Insight of thermal transport in Maxwell fluid flow on a porous
contracting flat surface inside a porous material with non-uniform heat
source/sink: Multiple solutions and stability treatment
**Authors:** Amit Kumar Pandey, Prakash Goswami, Sohita Rajput, Krishnendu
Bhattacharyya, Ali J. Chamkha, Dhananjay Yadav
**Journal:** Forces in Mechanics, article 100352 (uncorrected proof at time of
reading), DOI 10.1016/j.finmec.2026.100352, 12 pages.
**Local copy:** `pagination_FINMEC_100352-1.pdf` in this folder (sha256
`005acd8e27c6533be8e9e4e996e7567bd9eb4d7671f31472bbc6d7d3fe4f5e3`), read in
full (all 12 pages) on 2026-11-05.

This file is a durable transcription for reproduction purposes. Where a
number below is used as an automated gate in `../evaluator.py`, it is marked
**[gate]** and was cross-checked against the exact inline comments already
present in the original `pinn_maxwell_shrinking.py` script (which the paper's
own co-author wrote against the paper). Tables transcribed purely for
reference (not used as automated gates) are marked **[reference only —
verify against PDF before quoting]**, since they were read visually off
rendered table images rather than extracted as text.

## Physical problem

2D steady boundary-layer flow and heat transfer of a Maxwell (viscoelastic)
fluid driven by a continuously *contracting* sheet (`U_w = c x`, `c < 0`
contraction constant convention per the paper) embedded in a Darcy porous
medium, with wall mass transfer (suction/injection) and a heat source/sink
that depends on **both** temperature and spatial position (`A*`, `B*` terms),
not temperature alone.

## Governing similarity ODEs (paper Eqs. 13–15)

```
(1 - β f²) f''' + 2β f f' f'' + f f'' - f'² - K f'  = 0        momentum
θ'' + Pr f θ' + A* θ + B* e^{-η}                      = 0        energy

f(0) = S,  f'(0) = -1,  f'(∞) = 0
θ(0) = 1,  θ(∞) = 0
```

- `β` = Deborah number / Maxwell parameter (`β=0` recovers Newtonian flow).
- `K` = porous-medium permeability parameter.
- `S` = wall mass-transfer parameter (`S>0` suction, `S<0` injection).
- `Pr` = Prandtl number.
- `A*>0` heat source / `A*<0` heat sink (temperature-dependent).
- `B*>0` heat source / `B*<0` heat sink (space-dependent, decays as `e^{-η}`).
- `η_∞ = 10` truncation (paper's own choice; also used by `reference_bvp.py`
  and `ref_eig_cheb.py`'s base-flow solve).

## Outputs

```
Cf Re_x^{1/2}  = (1 + β) f''(0)         local skin friction
Nu Re_x^{-1/2} = -θ'(0)                  local Nusselt number
```

## Solution method used by the paper itself

Shooting method converted to an IVP, **4th-order Runge–Kutta** with step
`Δη = 0.01`, unknown wall values `f''(0)`, `θ'(0)` iterated by the **Secant
method** to `10⁻⁵` convergence against the far-field BCs. This is *not* what
either companion script uses: `reference_bvp.py` uses `scipy.solve_bvp`
(collocation) and the PINN uses a neural trial function with hard-constrained
BCs — both independent re-derivations of the same claim by a different
numerical route, which is the right design for cross-validation.

## Dual solutions and the critical suction parameter

For the baseline case (`β=0.1, K=0.1, Pr=2, A*=0.1, B*=0.1`) the paper reports
dual (upper/lower branch) similarity solutions exist only for `S ≥ S_c`, with
**[gate] `S_c = 1.8391`** (paper text, Section 4.1, and Fig. 2 caption). No
similarity solution exists for `S < S_c`. Increasing `β` lowers `S_c`
(elasticity delays boundary-layer separation, reducing the suction needed);
e.g. for `β=0.2`, dual solutions persist down to `S ≥ 1.76205`.

## Table 1 — `Cf Re_x^{1/2}` (page 7)

Baseline row used as an automated accuracy gate:

| S | K | β | upper | lower |
|---|---|---|---|---|
| 1.95 | 0.1 | 0.1 | **[gate] 1.93523180** | **[gate] 0.75244081** |

Remaining rows, **[reference only — verify against PDF before quoting]**:

| S | K | β | upper | lower |
|---|---|---|---|---|
| 1.85 | 0.1 | 0.1 | 1.38936949 | 1.02155828 |
| 1.90 | 0.1 | 0.1 | 1.69646847 | 0.84821391 |
| 2.00 | 0.1 | 0.1 | 2.16110373 | 0.67993534 |
| 2.05 | 0.1 | 0.1 | 2.38746357 | 0.61844420 |
| 2.10 | 0.1 | 0.1 | 2.62119412 | 0.56244284 |
| 2.15 | 0.1 | 0.1 | 2.86722875 | 0.50851816 |
| 2.20 | 0.1 | 0.1 | 3.12997127 | 0.45402074 |
| 2.30 | 0.1 | 0.1 | 3.72404718 | 0.33319199 |
| 2.40 | 0.1 | 0.1 | 4.44764614 | 0.17611793 |
| 1.95 | 0   | 0.1 | 1.20001829 | 0.74850744 |
| 1.95 | 0.1 | 0.15 | 2.77435660 | 0.71841216 |
| 1.95 | 0.1 | 0.17 | 3.36044216 | 0.68611372 |
| 1.95 | 0.1 | 0.20 | 4.93494320 | 0.58865333 |
| 1.95 | 0.13 | 0.1 | 1.98493433 | 0.69930309 |
| 1.95 | 0.16 | 0.1 | 2.03093195 | 0.64972967 |
| 1.95 | 0.20 | 0.1 | 2.08774567 | 0.58799291 |

## Table 2 — `Nu Re_x^{-1/2}` (page 7)

Baseline row used as an automated accuracy gate:

| S | K | β | Pr | A* | B* | upper | lower |
|---|---|---|---|---|---|---|---|
| 1.95 | 0.1 | 0.1 | 2 | 0.1 | 0.1 | **[gate] -3.35807824** | **[gate] -3.14427924** |

Remaining rows, **[reference only — verify against PDF before quoting]**
(Pr, A*, B* varied; S, K varied — see PDF page 7 Table 2 for the complete
grid, it is denser than Table 1).

## Table 3 — smallest stability eigenvalue `α₁` (page 10)

Stability base case: `β=0.1, K=0.1, A*=0.1, Pr=2`, `B*=0` (eigenvalue problem
is linear/homogeneous and decoupled from the particular energy forcing term).

| S | upper branch α₁ | lower branch α₁ |
|---|---|---|
| 2.0 | **[gate] 0.8810518** | **[gate] -0.0376726** |
| 2.1 | 1.1498579 | -0.1290175 |
| 2.2 | 1.4609727 | -0.2006274 |
| 2.3 | 1.8396721 | -0.2601093 |
| 2.4 | 2.3116364 | -0.3115262 |

(rows beyond `S=2.0` are **[reference only — verify against PDF before
quoting]**; `S=2.0` is the row already hardcoded as a comparison target inside
`pinn_maxwell_shrinking.py`'s own driver, cross-checked against the PDF image
here.)

**Sign convention / physical meaning:** positive leading eigenvalue ⇒
disturbances decay ⇒ branch is stable. The paper finds the **upper branch
stable** (`α₁ > 0`) and the **lower branch unstable** (`α₁ < 0`) at every
tabulated `S`, consistent with classical boundary-layer dual-solution theory
(Weidman-style stability treatment, their Eqs. 28–33).

## Key qualitative conclusions (Section 6, for context when judging candidate
PINN changes — do not let a "faster" candidate silently break these signs)

- Higher Deborah number `β`: expands the dual-solution existence range
  (lowers `S_c`), raises upper-branch skin friction, lowers lower-branch skin
  friction, raises upper-branch Nusselt number.
- Higher permeability parameter `K`: enlarges the existence range; raises
  upper-branch `Cf` and `Nu`; lowers lower-branch `Cf` and `Nu`.
- Higher Prandtl number `Pr`: raises cooling rate (`Nu`) on both branches.
- Temperature-dependent source (`A*>0`) / sink (`A*<0`): reduces / enhances
  `Nu` on both branches.
- Space-dependent source (`B*>0`) / sink (`B*<0`): reduces / enhances `Nu`,
  more strongly than the `A*` effect.
- Upper branch is always the physically stable, realizable one.
