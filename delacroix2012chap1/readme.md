# Child quantity-quality tradeoff

## Model

Main functional equations in page 12 and 17 of [de la Croix, 2012](https://www.cambridge.org/core/books/fertility-education-growth-and-sustainability/02F5F1FC987715CC5F1C4AA2E8B6C4E9)

if $w^i_t > \theta/\eta\phi$:

$$
e^i_t = \frac{\eta \phi w^i_t - \theta}{1-\eta},
$$

$$
n^i_t = \frac{(1-\eta)\gamma w^i_t}{(\phi w^i_t - \theta)(1+\gamma)}.
$$

otherwise,

$$
e^i_t = 0,
$$

$$
n^i_t = \frac{\gamma}{\phi(1+\gamma)}.
$$

with

$$
w^i_t = y^i_t = v (1-\phi n).
$$

## Data

Data on observable variables, such as $n, e + \theta, y$:
- $n$: [Fertility rate, total (births per woman)] / 2 $\times$ (1 - [Mortality rate, infant (per 1,000 live births)] /1000)
- $\theta + e$: [Adjusted savings: education expenditure (% of GNI)] $\times$ [GNI per capita, PPP (current international $)]
- $y$: [GNI per capita, PPP (current international $)]
The unobserved, $w=v$ can be inferred from the equation

$$
  y = v(1-\phi n)
$$

All data are taken from WDI, from 1998 to 2002, for all available countries.

## Structural Estimation

The method is Full Information Maximum Likelihood, which is implemented in `mle.py`.

## Results

eta: 0.5463, phi: 0.0350, theta: 49.9935, gamma: 0.0858

R^2 on $n$ equation: 0.5666

R^2 on $e$ equation: 0.8249

From the chapter

eta: 0.572, phi: 0.039, theta: 51.61, gamma: 0.103

![](https://raw.githubusercontent.com/thanhqtran/replications/main/delacroix2012chap1/plot.png)













