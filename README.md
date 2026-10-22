# Pricing motor insurance: frequency × severity on French MTPL data

A motor insurer must charge each policy its **pure premium**: the expected cost of its claims per
year. This repo estimates it on `freMTPL2`, a public French motor third-party liability (MTPL)
portfolio of 678,013 policies and 26,639 claims, with the standard actuarial decomposition:
**how often** a policy claims (frequency) times **how much** a claim costs (severity).

Everything is in one notebook, [`Motor_Insurance_Pricing.ipynb`](Motor_Insurance_Pricing.ipynb):
data cleaning, a Poisson GLM against gradient boosting (LightGBM) for the frequency, a Gamma GLM
for the severity, and the comparison of the resulting premiums on a test set. This README explains
the theory behind each step and summarises the results.

## Headline

| Test set (135,777 policies) | Frequency: Poisson deviance ↓ | Frequency: Gini ↑ |   Pure premium: Gini ↑ |
|------------------------------|------------------------------:|------------------:|-----------------------:|
| same premium for everybody   |                        0.4755 |                 0 |                      0 |
| GLM                          |                        0.4530 |            0.2984 |     0.3062 (GLM × GLM) |
| gradient boosting            |                    **0.4441** |        **0.3341** | **0.3084** (GBM × GLM) |

- **The raw claim counts are wrong for a block of 9,139 policies.** Rebuilding the claim counts from the claims table
  reproduces exactly the counts published by Wüthrich & Merz (2023): 678,007 policies, 26,383 claims.
- **Frequency: gradient boosting beats the GLM**, and where the two disagree the boosting is right
  (right panel below).
- **Severity: there is almost nothing to learn.** In cross-validation neither a GLM nor a GBM removes
  more than a fraction of a percent of the deviance of the plain mean: the cost of a claim depends on
  the accident, which the data do not describe.

![Frequency on the test set: lift charts and double lift](figures/frequency_test.png)

## The problem

For a policy with characteristics $x$ observed for $e$ years (the **exposure**), let $N$ be its
number of claims and $Y_1, \dots, Y_N$ their costs. The total cost is $S = \sum_{k=1}^{N} Y_k$ and the
pure premium is the expected cost per year:

$$
\pi(x) = \frac{\mathbb{E}[S \mid x]}{e} = \frac{\mathbb{E}[\mathbb{E}[S \mid N,x] \mid x]}{e}=\frac{\mathbb{E}[N\mathbb{E}[Y \mid x] \mid x]}{e}=\underbrace{\frac{\mathbb{E}[N \mid x]}{e}}_{\text{frequency } \lambda(x)} \times \underbrace{\mathbb{E}[Y \mid x]}_{\text{severity } \mu(x)} .
$$

The factorisation holds if, given $x$, the claim costs are identically distributed and independent of
the number of claims. It lets us fit two simpler models, on two different datasets:
the frequency on all policies, the severity on the claims only.

## Frequency: Poisson with an exposure offset

The number of claims is modelled as $N \sim \text{Poisson}(\lambda(x)\, e)$: a policy observed for half a
year is expected to have half the claims of the same policy observed for a year. The Poisson model can be motivated as a limit of Binomial models. Fix a policy
profile $x$ and an exposure period of $e$ years. Divide this period into
$M$ short intervals, and suppose that each interval independently produces
one claim with probability $p_M$, and no claim otherwise. The total number
of claims then satisfies

$$ N_M \sim \text{Bi}(M,p_M). $$

Let $\lambda(x)$ denote the annual claim frequency. If, as $M\to\infty$,
the probabilities satisfy $Mp_M\to e\lambda(x)$, then, for every
non-negative integer $n$,

$$ \Pr(N_M=n) = \binom{M}{n}p_M^n(1-p_M)^{M-n} \longrightarrow \exp\!\bigl(-e\lambda(x)\bigr) \frac{\bigl(e\lambda(x)\bigr)^n}{n!}.$$

Thus the limiting claim count is Poisson with mean $e\lambda(x)$.
Here $\lambda(x)$ is a rate, measured in claims per year, whereas
$e\lambda(x)$ is the expected number of claims over the exposure period.
This motivates the assumption that expected claim counts $N$ scale linearly
with exposure.
With a log link,

$$
\log \mathbb{E}[N \mid x] = \log e + \log \lambda(x), \text{ where }\lambda(x) \text{ is estimated with a linear model, i.e.} \log \lambda(x) = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k.
$$

The term $\log e$ has no coefficient: it is an **offset**, a known part of the linear predictor.

**Estimating the coefficients $\beta_i$** Coefficients are estimated by maximizing the log-likelihood of the Poisson distribution assuming $\{x^i\}$ observations, which we recall to be

$$ \ell( \beta_1, \dots, \beta_k|x^i) = \sum_{i} N_i \log(\mu_i) - \mu_i- \log(N_i !), \text{ with } \mu_i = e_i \lambda_i = e_i \exp(\beta_0 + \beta_1 x_1 ^i + \dots + \beta_k x_k ^i)$$

**The log link makes the tariff multiplicative.** Each coefficient becomes a **relativity**
$e^{\beta_j}$, the factor by which a characteristic multiplies the frequency with respect to a
reference level. 

**The Poisson GLM is balanced.** With the log link (the canonical link of the Poisson family) the
maximum-likelihood equations (i.e. $\nabla_ \beta \ell =0$) are $X^\top (N - \hat\mu) = 0$; the row of the intercept says that the
total predicted claims equal the observed total on the training data, exactly. The notebook checks it. We also point out that if $X$ (i.e. the matrix whose columns are the values of the features $x_i$) has 
full rank, the hessian of the log-likelihood is negative definite, i.e. $\ell$ is strictly concave, so any finite maximizer is unique.  
The reason we discard one of the levels for each categorical variable after the one-hot encoding is precisely to maintain this property, i.e. to have a unique MLE $\hat{\beta}$. 

**The deviance measures the fit.** For counts, the squared error is a poor measure: the variance grows
with the mean, and 96% of the policies have no claim. The Poisson deviance

$$
D = \frac{2}{\sum_i e_i} \sum_i  \left[ N_i \log \frac{N_i}{\hat\mu_i} - (N_i - \hat\mu_i) \right], \qquad \hat\mu_i = \hat\lambda(x_i)\, e_i  \quad \text{ where } \quad \hat{\lambda}(x) =  \exp(\hat{\beta_0} + \hat{\beta_1} x_1 ^i + \dots + \hat{\beta_k} x_k ^i)
$$

is the loss the Poisson model minimises (lower is better). Even a correctly specified conditional-mean model generally 
has positive out-of-sample deviance, because individual claim counts remain random. That is why the improvements look small
(a few percent): they are compared with that randomness. 

## Gradient boosting

A gradient boosting machine (here LightGBM) adds hundreds of small decision trees, each one fitted to
correct the errors of the previous ones, maximising the same likelihood. Unlike a GLM it finds by
itself non-linear effects and interactions, and it uses the raw covariates.

The exposure is passed as the starting point of each policy (`init_score`), set to
$\log e + \log \hat\lambda_0$, where $\hat\lambda_0$ is the average frequency: the trees learn a
multiplicative correction to the average. Two practical points:
- LightGBM does **not** store `init_score` in the model: at prediction time the trees return only
  their correction, and the base frequency has to be put back by hand.
- The number of trees is chosen by early stopping on a validation part of the training set, never on
  the test set.

## Severity: Gamma with a log link, and large claims

Claim costs are positive and skewed, and their spread grows with their mean. The Gamma distribution
has exactly this property, $\operatorname{Var}(Y) = \phi\, \mu^2$ (a constant coefficient of
variation), and with a log link the severity tariff is multiplicative too.

**Large claims.** This is one of the crucial aspects of the problem. The largest 1% of claims carry 38% of the total cost, and the largest is 4 million. A
model fitted on raw amounts would mostly describe a handful of claims. Claims are therefore capped at
$u = 50{,}000$, the severity model is fitted on the capped amounts, and the cost above the cap is added
back to every claim as a **loading**:

$$
\text{expected cost per claim} = \hat\mu_{\text{capped}}(x) \times (1 + L), \qquad
L = \frac{\sum_k (Y_k - u)_+}{\sum_k \min(Y_k, u)} \quad \text{(on the training claims)} .
$$

The loading assumes that large claims are proportional to the "normal" severity of each profile: with
68 training claims above the cap, how they depend on 8 covariates cannot be estimated by any model.

**Cross-validation.** With about 21,000 noisy claims, one validation split is not enough to tell
whether a flexible model helps. The notebook compares the mean, the GLM and a GBM with 5-fold
cross-validation, measured by $D^2 = 1 - D_{\text{model}} / D_{\text{mean}}$, the share of the Gamma
deviance of the mean that a model removes. The deviance $D$ can be expressed in terms of the log likelihood as

$$ D = 2 \phi ( \max_{\mu} \ell(\mu) - \ell (\hat{\beta})),$$

i.e. $2 \phi$ times the difference between the maximum possible value of $\ell$ (with $\mu_i = N_i$) and the value it attains 
considering the maximum likelihood estimator of the coefficients $\hat{\beta}$. 

## Pure premium, balance and how the models are compared

Each frequency model is multiplied by the severity GLM, applied to **every** policy (with or without
claims). Each resulting premium is then rescaled by a constant so that its total equals the total
observed cost **on the training set** (balance correction); the test set is never used to fit or
calibrate anything.

On the test set the premiums are compared with these tools:
- **Gini index.** Sort the policies from the lowest to the highest predicted premium and draw the
  share of observed cost against the share of exposure (the Lorenz curve). A model that ranks risks
  well puts little cost on the policies it rates lowest; the Gini index is $1 - 2 \times$ (area under the
  curve): 0 for no ranking, higher is better. It measures ranking only, not the level of the premium.
- **Balance**: total predicted / total observed cost.
- **Lift and double lift charts.** Cut the policies into 10 groups of equal exposure by predicted
  value (lift) or by the ratio of two models' predictions (double lift), and compare observed and
  predicted values in each group. In a double lift chart, where the two models disagree, the one whose
  line follows the observed one is right.


## Why the train/test split is grouped

38% of the policies share all their characteristics with at least one other record: the same car
renewed during the year, fleets of identical cars. If two records of the same car land one in train and
one in test, the model has already seen that car, and the test score is optimistic (more for a GBM,
which can memorise exact combinations of covariates). The split therefore sends all policies with
identical covariates to the same side.

## The data, and what is wrong with them

The data come from the CASdatasets R package: the original 2017 release, with the same row counts as
the OpenML version used by the scikit-learn tutorial, downloaded by the notebook from a fixed commit of
the [GitHub repository](https://github.com/dutangc/CASdatasets). Three problems, three decisions:

| Problem | Evidence | Decision |
|---|---|---|
| The policy table counts 36,102 claims, the claims table 26,639 | All 9,139 policies with `IDpol` ≤ 24,500 have a claim in the policy table, almost none in the claims table | Claim counts rebuilt from the claims table; 195 claims without a policy dropped |
| A few policies have 6 to 16 claims | Mostly one car split over records of a few weeks | Policies with more than 5 claims dropped |
| 1,224 policies are exposed for more than one year | The portfolio is observed over one year | Exposure capped at 1 |

The result, **678,007 policies and 26,383 claims**, matches exactly the counts reported by
Wüthrich & Merz (2023, §13.1) after the same cleaning. `Area` is left out of the models: it is just
`Density` cut into six classes.

## Results

All on the test set, which the models never saw.

**Frequency** (claims per year):

| model | Poisson deviance | Gini | balance |
|---|---:|---:|---:|
| same frequency for everybody | 0.4767 | 0 | 1.003 |
| Poisson GLM | 0.4545 | 0.295 | 1.001 |
| LightGBM | **0.4457** | **0.331** | 1.001 |

**Severity** (capped cost per claim): $D^2$ of the Gamma GLM in 5-fold cross-validation 0.2%, of a small GBM 0.05%; on the test claims the GLM reaches -0.20%. 

**Pure premium** (cost per year):
| model | Poisson deviance | Gini | balance |
|---|---:|---:|---:|
| same frequency for everybody | 0.4755 | 0 | 1.0001 |
| Poisson GLM | 0.4530 | 0.2984 | 0.9974 |
| LightGBM | 0.4441 | 0.3341 | 0.9982 |

![Pure premium on the test set: Lorenz curves and double lift](figures/pure_premium_test.png)

### Do the models work?

Not "which one is better", but whether they work at all: on the test set, cut into 10 groups of equal
exposure from the lowest to the highest prediction (notebook §9).

| test set | riskiest 10% / safest 10% (observed) | largest error in a group | total predicted / observed |
|---|-------------------------------------:|-------------------------:|---------------------------:|
| frequency, GLM |                                 5.6× |                      13% |                      0.997 |
| frequency, GBM |                                 7.7× |                       9% |                      0.998 |
| pure premium, GLM × GLM, cost below 50,000 per claim |                                 6.8× |                      29% |                      0.966 |
| pure premium, GBM × GLM, cost below 50,000 per claim |                                 9.4× |                      36% |                      0.967 |

The frequency models separate the risks and get their level right. The severity model barely moves
away from the average cost per claim. Indeed very large claims are predictable by
no model: they are covered on average by the loading, and they make one year's total cost uncertain.

**Every premium over-predicts the test cost by 4%**, the flat one included. The test set has a smaller share of cost above the cap than the training set
(about 18% against 24%). This is the large-claim loading meeting one year of heavy-tailed data, not a defect of the models. the test set happens to
contain fewer very large claims than the training set (15% of its cost above the cap against 24%). This
is the large-claim loading meeting one year of heavy-tailed data, not a defect of the models.

## Limitations

- **One test year.** With large claims this volatile, the difference between the two premiums (Gini
  0.308 against 0.306) is within what another sample could reverse; a bootstrap of the test set or more
  years of data would be needed before choosing one for a real tariff.
- **No time dimension.** The data have no dates, so the models are tested on a random sample of the same
  year, not on the following one.
- **The loading is estimated on 68 claims**, and the frequency × severity split assumes that the cost
  of a claim does not depend on how many claims the policy had.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook Motor_Insurance_Pricing.ipynb
```

The first run downloads the data (about 4 MB) into `data/` and converts them; the whole notebook then
runs in about a minute. Tested with Python 3.13, pandas 3.0, scikit-learn 1.9, statsmodels 0.15,
LightGBM 4.7.

## Data and licence

The code in this repo is MIT (see `LICENSE`). The data are **not** included: the notebook downloads
them from the CASdatasets package by C. Dutang and A. Charpentier, distributed under the GNU GPL
(version 2 or later). See `NOTICE.txt`.

This project was developed with the help of an AI assistant (Claude).

## References

- M. V. Wüthrich, M. Merz (2023). *Statistical Foundations of Actuarial Learning and its Applications*.
  Springer (open access). Section 13.1: the freMTPL2 data and their cleaning.

