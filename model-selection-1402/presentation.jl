### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 21c3983f-3cb4-40e4-906f-f32b30c3af55
using Pkg; Pkg.activate("/Users/pavanchaggar/ResearchDocs/Presentations/model-selection-1402")

# ╔═╡ 1353a419-ce1d-48c4-975e-51789d6cc53b
include("functions.jl");

# ╔═╡ d212b0bb-6faa-4a3a-98b9-dafe7bee364a
html"""<style>
main {
max-width: 900px;
}"""

# ╔═╡ 6964a783-3e38-43e5-81f0-6163ac9a2bf2
html"<button onclick='present()'>present</button>"

# ╔═╡ 8be16947-1600-446e-8ef7-90783ab2e41e
md" 
# Understanding Predictive Information Critera for Bayesian Models
\
**Pavanjit Chaggar, December 2021** \
pavanjit.chaggar@maths.ox.ac.uk \
@ChaggarPavan on Twitter \
DPhil student at the Mathematical Institute.
Supervised by Alain Goriely and Saad Jbabdi, with support from Stefano Magon and Gregory Klein at Roche.
"

# ╔═╡ 57a6649d-9003-4edf-bb3c-9d1b18a72806
md" 
# Outline 

1) Log Predictive Density as a Measure of Predictive Accuracy
2) Information Criteria
3) Example
" 

# ╔═╡ fe795025-76a5-412d-9ded-a24fcabeee68
md" 
# Some definitions
"

# ╔═╡ d649bd5f-c46b-4a3a-8ba0-a69be091e5d9
md" 
# Expectation 
The expected value of a random variable, $\mathbf{X}$, is given by: 
```math 
\mathbb{E}[\mathbf{X}] = \int x f(x) dx
```
The conditional expectation of a random variable, $\mathbf{X}$, w.r.t. some other random variable, $\mathbf{Y}$, is: 
```math
\mathbb{E}[\mathbf{X} \mid \mathbf{Y} = y] = \int x f(x \mid y) dx
```
"

# ╔═╡ bded7dd1-c1dc-4f7d-bda6-099aa8077f83
md"
# Entropy

Given a perfect encoding scheme, the **expected** number of bits needed to encode a random variable, $\mathbf{X}$, is given by: 
```math
\mathbf{H}[\mathbf{X}] = \int x \log f(x) dx
```

The cross entropy between two distributions, $f(x)$ and $g(x)$ for $x \in \mathbf{X}$: 

```math
\begin{align}
\mathbf{H}[f, g] &= -E[f \mid g] \\
&= \int \log g(x) f(x) dx
\end{align}
```
It can also be reformulated as: 
```math
\mathbf{H}[f, g] = \mathbf{H}[f] + \mathbf{D}_{kl}[f \mid \mid g]
```
"

# ╔═╡ fc4027c4-2dc1-47ed-98a5-f6ca1c307bf7
md" 
# Measure of Predictive Accuracy

On what basis do we compare models?

" 

# ╔═╡ a281a7c0-9fe4-46b0-bc0e-2b76b0cc0602
md" 
## Log Predictive Density

* Fancier name for log-likelihood, $\log p(y, \theta)$
* For model comparison, we are interested in how well the model describes the data and gneeralises. Therefore, we are not interetsed in the impact of the prior and can use the log-likelihood as opposed to the log-posterior.
* The prior is still useful for finding good maps between parameters and data.
"

# ╔═╡ 02e97f11-36d9-48dd-bd1e-eccb5d27c7e6
md" 
## Predictive Density

The posterior predictive density given posterior distribution $p( \theta \mid y )$ and new data point $\hat{y}_i$ is: 
```math
p(\hat{y_i} \mid y) = \int p(\hat{y_i} \mid \theta) p(\theta \mid y) d\theta
```

The log predictive densisty is simply the logarithm of this and is refered to as the log predictive density (lpd). 

Since future data are unknown, we should define an expectation over $y_i$, called the expected log predictive densisty (elpd). 

```math
\mathbb{E}[\log p(\hat{y_i} \mid y) \mid f(\hat{y_i})] = \int \log p(\hat{y_i} \mid y) f(\hat{y_i}) d\hat{y_i}
```
Where $f(\cdot)$ is the *true* generative data distribution. 
"

# ╔═╡ a097328f-0a66-41ad-8fda-dff92f0481d6
md" 
## Predictive Density
```math
- \mathbb{E}[\log p(\hat{y_i} \mid y) \mid f(\hat{y_i})] = -\int \log p(\hat{y_i} \mid y) f(\hat{y_i}) d\hat{y_i}
```
The negative elpd has the same form as a cross entropy:
```math
\mathbf{H}[f, g] = \int \log g(x) fx dx.
```
and can therefore be interpreted as the cross-entropy between the true data generating process $f(\hat{y_i})$ and the predictive model $p(\hat{y_i} \mid y)$. Or how much information the model captures about the true generative process. 

Similarly, it may be interpreted as the KL divergence between the model and the true generative process, since, 
```math
\mathbf{H}[f, g] = \mathbf{H}[f] + \mathbf{D}_{kl}[f \mid \mid g]
```

and $\mathbf{H}[f]$ is an unknown constant. 

"

# ╔═╡ e75be44f-1433-49dd-b486-ef173b8c556f
md" 
## Predictive Density

For a new dataset, $\hat{y} = \{y_1, y_2, \ldots, y_n\}$, we can define the **pointwise) elpd, or the expected log pointwise predictive density (ellpd) as simply the sum over the elpd's 

```math
\mathbb{E}[\log p(\hat{y} \mid y) \mid f(\hat{y})] = \sum_{i = 1}^{n} \int \log p(\hat{y_i} \mid y) f(\hat{y_i}) d\hat{y_i}
```
"

# ╔═╡ 66899832-3de0-4789-a477-dfb3ace19301
md" 
## Predictive Density 

Since we do not know what the true data generating process is, we leave it out and estimate the ellpd up to a constant and so we no longer have an expectation, just a summation over the log pointwise predictive density (lppd) 

```math
\mathbb{E}[\log p(\hat{y} \mid y) \mid f(\hat{y})] \approx \sum_{i = 1}^{n} \int \log p(\hat{y_i} \mid \theta) p (\theta \mid y)d\theta
```

and in practice, this is computed using the posterior samples: 

```math
\sum_{i = 1}^{n} log \bigg( \frac{1}{S} \sum_{s=1}^{S} p(y_i \mid \theta_s) \bigg)
```
"

# ╔═╡ 2490c06a-1994-453c-9299-ae24d6f3e6cb
md"
## Predictive Density

* We can assess predictive accuracy as the cross entropy between the true data generating process and the predictive distribution. 
* We can estimate this using the log pointwise predictive density
* This is typically a biased estimate for the predictive accuracy against future observations and thus are need of bias correction. This is usually what information criteria attempt to address.
"

# ╔═╡ f677be1a-733f-4e16-983a-6757ddb37e18
md" # Information Criteria
* Helps us assess predictive accuracy 
* Helps us choose a model
"

# ╔═╡ 01fa40b0-c545-4f97-9e36-99fc1a62790c
md" # What can we do?
- Within-sample predictive accuracy 
  - Summary of predictive accuracy on the training data. Definitely biased...
- Adjusted within-sample predictive accuracy 
  - Try to unbias the predictive accuracy by penalising model complexity (over-fitting).
  - AIC, BIC, DIC, WAIC.
- Out-of-sample predictive accuracy (cross validation)
  - We usually don't have out-of-sample data. validation.
  - Leave-p-out cross validation.
"

# ╔═╡ 2bd8770e-b6f7-4fde-9f9f-f61ab9385977
md" 
# AIC
The goldren retriever. Simple and a bit goofy.
"

# ╔═╡ 7e7568c2-819b-4480-b4cb-b883d1d6174b
md" 
## AIC

Recall that the elpd is the negative cross entropy between the true data generating process $f(\cdot)$ and the estimated model $g(\cdot) = p(\Theta \mid \mathbf{y})$. 

>$elpd = - \mathbf{H}[f, g] = -\mathbf{H}[f] - \mathbf{D}_{kl}[f \mid \mid g]$

Since we don't know $f(\cdot)$, we can't estimate $H[f]$. However, this is just an unknown constant. So, to maximimise the elpd we can just minimise the KL-divergence bewteen $f(\cdot)$ and $g(\cdot)$!
"

# ╔═╡ de85f919-aa9b-4856-b4bd-202404bc217b
md"
## AIC

We can approximate the minimum KL-divergence by using the maximum likelihood estimate. 
>$\hat{elpd}_{AIC} \approx min(\mathbf{D}_{kl}[f \mid \mid g]) = log(\mathbf{y} \mid \Theta_{MLE})$

The AIC is the negative of this with a penalty for the number of parameters, multiplied by two for good measure.

>$AIC = 2k - 2log(\mathbf{y} \mid \Theta_{MLE})$

For some family of models, we would want to choose the model with the lowest AIC score.s"

# ╔═╡ cfa6888d-d01d-4838-8230-f897598ffd92
md" 
# BIC
A slightly more well groomed golden retriever. But still simple and goofy.
"

# ╔═╡ 4726ebb2-b0d6-4d19-a2ab-6ba4be2c8c2f
md" 
## BIC

Almost identical to the AIC with one *significant* change:

>$BIC = k\log(n)-2log(\mathbf{y} \mid \Theta_{MLE})$

The penalty term is scaled by $\log(n)$, where n is the number of data points.
"

# ╔═╡ 2c6bfd56-0293-4a4d-a327-75273d70ab93
md" 
# DIC 
A poodle like information criteria. Generally liked, a bit of a pain sometimes.
"

# ╔═╡ 83ada7b6-77c4-490b-b139-e08a85193205
md" 
## DIC 

A more Bayesian version of the same principle as AIC, but with some modifications. 

* First, instead of using the MLE estimate, we use the posterior mean, $\Theta_B = \mathbb{E}[\Theta \mid \mathbf{y}]$. 
* Second, there's a longer penalty term for the effective number of parameters.

>```math
>\begin{align}
>DIC &= 2p_{DIC} - 2\log p(\mathbf{y} \mid \bar{\Theta}) \\
>p_{DIC} &= 2\big[\log p(\mathbf{y} \mid \bar{\Theta}) - \frac{1}{S}\sum_s \log p(y \mid \Theta_s)\big] \\
> &=2\text{var}[\log p(\mathbf{y}, \Theta) \mid p(\Theta \mid \mathbf{y})]
>\end{align}
>```
This is especially useful when the number of parameters is not obvious, e.g. when there are lots of covarying parameters.
"

# ╔═╡ d84e587c-59dd-42b6-acfc-531796982187
md" 
# WAIC"

# ╔═╡ 878db965-8345-45e6-95e5-5d4ab972e807
md" 
## WAIC

Unlike the AIC, BIC and DIC, the WAIC uses the full lppd as a measure of predictive accuracy, as oppoesed to some optimised log-likelihood. 

There are two penalty terms, to be consistent with DIC and LOO-CV, the variance based measure is preferred. 
>$p_{waic} = \sum_{i=1}^n \text{V}_{s=1}^{S}[log(y_i \mid \Theta_s)]$

This is the summed pointwise posterior variance of the log predictive density.

"

# ╔═╡ 81c1127a-3402-439b-ae7a-4124dbda003c
md" 
## WAIC

Then, the WAIC is defined as: 

>```math
>\begin{align}
>\text{WAIC} &= lppd - p_{waic} \\
>&= 2 \bigg[\sum_{i = 1}^{n} log \bigg( \frac{1}{S} \sum_{s=1}^{S} p(y_i \mid \theta_s) \bigg)  \\
> &- \sum_{i=1}^n \text{V}_{s=1}^{S}[log(y_i \mid \Theta_s)]\bigg]
>\end{align} 
>```
"

# ╔═╡ Cell order:
# ╠═21c3983f-3cb4-40e4-906f-f32b30c3af55
# ╠═1353a419-ce1d-48c4-975e-51789d6cc53b
# ╠═d212b0bb-6faa-4a3a-98b9-dafe7bee364a
# ╟─6964a783-3e38-43e5-81f0-6163ac9a2bf2
# ╟─8be16947-1600-446e-8ef7-90783ab2e41e
# ╟─57a6649d-9003-4edf-bb3c-9d1b18a72806
# ╟─fe795025-76a5-412d-9ded-a24fcabeee68
# ╟─d649bd5f-c46b-4a3a-8ba0-a69be091e5d9
# ╟─bded7dd1-c1dc-4f7d-bda6-099aa8077f83
# ╟─fc4027c4-2dc1-47ed-98a5-f6ca1c307bf7
# ╟─a281a7c0-9fe4-46b0-bc0e-2b76b0cc0602
# ╟─02e97f11-36d9-48dd-bd1e-eccb5d27c7e6
# ╟─a097328f-0a66-41ad-8fda-dff92f0481d6
# ╟─e75be44f-1433-49dd-b486-ef173b8c556f
# ╟─66899832-3de0-4789-a477-dfb3ace19301
# ╟─2490c06a-1994-453c-9299-ae24d6f3e6cb
# ╟─f677be1a-733f-4e16-983a-6757ddb37e18
# ╟─01fa40b0-c545-4f97-9e36-99fc1a62790c
# ╟─2bd8770e-b6f7-4fde-9f9f-f61ab9385977
# ╟─7e7568c2-819b-4480-b4cb-b883d1d6174b
# ╟─de85f919-aa9b-4856-b4bd-202404bc217b
# ╟─cfa6888d-d01d-4838-8230-f897598ffd92
# ╟─4726ebb2-b0d6-4d19-a2ab-6ba4be2c8c2f
# ╟─2c6bfd56-0293-4a4d-a327-75273d70ab93
# ╠═83ada7b6-77c4-490b-b139-e08a85193205
# ╟─d84e587c-59dd-42b6-acfc-531796982187
# ╟─878db965-8345-45e6-95e5-5d4ab972e807
# ╠═81c1127a-3402-439b-ae7a-4124dbda003c
