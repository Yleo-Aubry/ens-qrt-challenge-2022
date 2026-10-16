# Linear factors for stock return prediction

Model submitted to [ENS Challenge Data no. 72](https://challengedata.ens.fr/challenges/72/), set by
Qube Research & Technologies (QRT, 2022): learn a linear factor model that predicts the next-day
return of each stock from its last 250 daily returns.

Public score: **0.0855**, against 0.0354 for the official benchmark with random factors and 0.0823
for an unweighted ridge regression.

## Problem

### Data

We observe the daily returns $r_{i,t}$ of 50 stocks over 754 dates. For each stock $i$ and each
date $t \in \{250, \dots, 753\}$ (that is, $T = 504$ dates), let

$$x_{i,t} = (r_{i,t-1}, r_{i,t-2}, \dots, r_{i,t-250}) \in \mathbb{R}^{250}$$

be the vector of the 250 past returns and $y_{i,t} = r_{i,t}$ the target. The hidden test set
contains 50 other stocks over the same period: the model must generalise from one stock to another,
not from one period to another.

### Prescribed model

The prediction is a linear model with $F = 10$ factors:

$$\hat y_{i,t} = \beta^\top A^\top x_{i,t}, \qquad A \in \mathbb{R}^{250 \times 10}, \quad \beta \in \mathbb{R}^{10},$$

where the matrix $A$ must lie on the Stiefel manifold:

$$A^\top A = I_{10}.$$

A submission that violates this constraint (to within $10^{-6}$ on any entry) receives a score
of $-1$.

### Metric

For each date, the vector of predictions $\hat y_t = (\hat y_{i,t})_i$ is compared with the vector of
realised returns $y_t = (y_{i,t})_i$ by cosine similarity, and the result is averaged over dates:

$$\mathrm{score} = \frac{1}{T} \sum_{t} \frac{\langle \hat y_t, y_t \rangle}{\lVert \hat y_t \rVert \, \lVert y_t \rVert}.$$

The metric depends only on the direction of the predictions at each date, not on their scale.

## Reduction of the problem

The prediction depends on $(A, \beta)$ only through the vector

$$w = A\beta \in \mathbb{R}^{250}, \qquad \hat y_{i,t} = w^\top x_{i,t}.$$

Conversely, any $w \neq 0$ can be written as $A\beta$ with $A$ on the Stiefel manifold: complete
$w / \lVert w \rVert$ with 9 orthonormal vectors (QR decomposition of $[\,w, Z\,]$, where $Z$ is a
$250 \times 9$ Gaussian matrix drawn with a fixed seed), then set $\beta = A^\top w$. Since $w$
lies in the column space of $A$, this gives $A\beta = w$.

The Stiefel constraint therefore does not restrict the set of attainable predictors: the problem
reduces to estimating a linear filter $w$ of 250 coefficients, shared by all stocks. The 9
additional columns of $A$ have no effect on the prediction.

## Model

The final model is the sum of two normalised estimators of $w$.

### 1. Date-weighted ridge regression

$$\hat w_R = \arg\min_w \sum_t \omega_t \sum_i \left(y_{i,t} - w^\top x_{i,t}\right)^2 + \lambda \lVert w \rVert^2,
\qquad \omega_t \propto \frac{1}{\lVert y_t \rVert},$$

with weights $\omega_t$ normalised to a mean of 1 and $\lambda = 10^4$. The solution is in closed
form:

$$\hat w_R = \left(X^\top \Omega X + \lambda I\right)^{-1} X^\top \Omega y.$$

The weighting corrects a mismatch between least squares and the metric. The metric is a cosine per
date, so it is insensitive to the magnitude of the returns on that date. Without weighting, a few
highly volatile sessions dominate the sum of squares. With $\omega_t \propto 1/\lVert y_t \rVert$,
each date contributes to the estimate in proportion to its magnitude rather than its square.

The parameter $\lambda$ has little effect over the range $[1, 10^4]$: with 25,200 observations for
250 coefficients, the problem is well conditioned.

### 2. Direct maximisation of the cosine with smoothing

$$\hat w_C = \arg\max_{\lVert u \rVert = 1} \;
\frac{1}{T} \sum_t \frac{\langle X_t u, y_t \rangle}{\lVert X_t u \rVert \, \lVert y_t \rVert}
\;-\; \lambda_s \, d \sum_{k=1}^{d-1} (u_{k+1} - u_k)^2,$$

where $X_t$ is the matrix of the $x_{i,t}$ at date $t$, $d = 250$ and $\lambda_s = 3 \cdot 10^{-5}$.

The first term is exactly the challenge metric on the training sample. The second penalises
changes in the filter from one lag to the next and limits overfitting. The normalisation
$\lVert u \rVert = 1$ is necessary: since the cosine is invariant to rescaling $u$, a penalty applied
to an unconstrained vector could be cancelled by letting its norm tend to 0. For the same reason,
an $L^2$ penalty would be constant here and have no effect.

The optimisation is carried out over $v \in \mathbb{R}^{250}$ with $u = v / \lVert v \rVert$, using
L-BFGS with an analytical gradient, starting from $\hat w_R / \lVert \hat w_R \rVert$. Without the
penalty, random initialisations all converge to the same solution, with an in-sample score of
0.1491: this is the maximum of the training metric over all linear filters.

### 3. Blend

$$w = \frac{\hat w_R}{\lVert \hat w_R \rVert} + \hat w_C.$$

The two components have equal weight, and this weight is not tuned on the data. The two filters are
close (cosine of 0.94 between them) but not identical. Averaging them reduces estimation variance:
it is the only change tested that significantly improves both the in-sample score and the
cross-validated score.

The submission is obtained by writing $w = A\beta$ with the construction described above.

## Validation

The test set consists of other stocks over the same dates. Cross-validation therefore splits the
**stocks**, never the dates: 5 folds of 10 stocks, repeated over 5 random splits (seeds 42 to 46),
giving 25 folds. Everything learnt from the data (date weights, ridge solution, cosine
optimisation) is recomputed on the training stocks of each fold only.

Models are compared fold by fold on the same splits (paired gain). The 25 folds reuse the same 50
stocks and are not independent: the standard error divides the standard deviation of the folds by
$\sqrt{5}$ rather than $\sqrt{25}$. A gain is regarded as genuine only if it exceeds two standard
errors and does not lower the in-sample score.

| Model | In-sample | Cross-validation | Paired gain | Public |
|---|---|---|---|---|
| Unweighted ridge ($\lambda = 10^4$) | 0.1298 | 0.0445 | -0.0075 ± 0.0028 | 0.0823 |
| Date-weighted ridge | 0.1407 | 0.0521 | reference | 0.0851 |
| Smoothed cosine alone | 0.1478 | 0.0572 | +0.0051 ± 0.0026 | 0.0834 |
| **Blend (final model)** | **0.1462** | **0.0557** | **+0.0037 ± 0.0013** | **0.0855** |

The blend is the only model whose gain exceeds two standard errors (2.8). With 50 test stocks,
public differences below about 0.003 are not significant; the choice rests on cross-validation,
not on the public leaderboard.

## Approaches not retained

The following were evaluated with the same protocol and not retained. This repository contains
only the final model; the code for these experiments is not included.

| Approach | Outcome |
|---|---|
| Fixed bases of 10 factors (lag blocks, DCT, exponentials) | cross-validation 0.015 to 0.023 |
| Principal component analysis (10 factors) | public 0.0380 |
| Autoregressive model AR(10) | public 0.0128 |
| Clipping returns at 2 standard deviations and Huber loss | best cross-validation (0.066), but in-sample 0.119 and public 0.0724 |
| Cosine maximisation without smoothing | gain not significant, public 0.0838 |
| $L^1$ penalty on the filter | no effect, the optimal filter is not sparse |
| Bagging over stocks or blocks of dates | negative gain |
| Synthetic stocks (combinations of correlated pairs) | equivalent to a reweighting, no gain |
| Reweighting stocks by correlation group | equivalent to inverse-volatility weighting, lower in-sample score |
| Filter estimated on the cross-sectional mean (market timing) | the market component carries only 5% of the target variance, negative gain |

Two findings limit the scope for further gains. The in-sample score is bounded by 0.1491, the
maximum of the metric over all linear filters, and the date-weighted ridge already reaches 95% of
it. Moreover, out-of-sample performance comes almost entirely from test stocks that have a highly
correlated counterpart in the training sample.

## Repository structure

```
data/                         challenge training data (source: challengedata.ens.fr)
qrt_model/
    data.py                   data loading and lag matrix
    metrics.py                challenge metric, Stiefel constraint check
    model.py                  weighted ridge, smoothed cosine, blend, construction of (A, beta)
    validation.py             cross-validation across stocks
    submission.py             writing and reading submission files
notebooks/evaluation.ipynb    in-sample scores, cross-validation, estimated filters
scripts/make_submission.py
submissions/
    final_submission.csv      final submission (public score 0.0855)
    example/                  format expected by the platform
tests/                        unit tests
```

## Usage

Python 3.11 or later.

```
pip install -r requirements.txt
python -m pytest -q tests
python scripts/make_submission.py submissions/new_submission.csv
```

The script fits the model on the 50 stocks, checks the Stiefel constraint and writes a file of
2,510 rows: the 2,500 coefficients of $A$ in row-major order, followed by the 10 coefficients of
$\beta$. It refuses to overwrite an existing file. Among other things, the tests check that the
model reproduces `submissions/final_submission.csv` exactly.
