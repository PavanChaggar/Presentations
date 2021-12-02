### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 860ff68f-634c-466f-b10b-a20010231418
include("funcs.jl")

# ╔═╡ edcd2286-c11a-43e3-a1fa-32a71998f5f0
html"""<style>
main {
max-width: 900px;
}"""

# ╔═╡ 8a2f8a5f-8333-4ebd-8e3f-f37892bb3a29
html"<button onclick='present()'>present</button>"

# ╔═╡ c3860233-3c3a-46a7-9f13-a6af2d6c71cd
md" 
# Inference Methods for Modelling Neurodegeneration

**Pavan Chaggar, July 2021**


Joint work with Alain Goriely, Saad Jbabdi, Travis Thompson and the _OxMBM_ group

github.com/PavanChaggar/inference-methods-UCSF0721
"

# ╔═╡ 0a436ace-bd7e-4e80-9527-b0580f749bfc
md"
# Overview and Introduction

- Alzheimer's disease (AD)
- Mathematical models of AD
- Inference methods
  - Variational inference 
  - Hamiltonian Monte Carlo 
- Preliminary results
"

# ╔═╡ 9c94781c-e244-4222-8c63-9485628cccd9
md" 
# Alzheimer's Disease -- A Brief Summary

Alzheimer's is characterised by gradual neurodegeneration associated with pervasive spreading of toxic protein species. 

In particular, two proteins, Amyloid beta (Aβ) and tau-protein (τP) are believed to underlie and drive the development of pathology. 

Historically, Aβ was primarily invstigated as the primary cause of AD. However More recent work has focussed on τP, in part because it spreads very predictably and is more tightly coupled with atrophy and symotom onset.
"

# ╔═╡ fd6af805-4180-4625-a931-cecf5e40bc03
md" 
## A Pernicious Pair of Predictable Prion Proteins

Both Aβ and τP grow via an autocatalytic process resembling those displayed by prions. 

This process is summarised as: 
"

# ╔═╡ 41c083ee-5a88-48cb-9b99-9b2eb1415a24
html"""
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/heterodimerkinetics.png?raw=true" height=250 width=500 vspace=50, hspace=175>"""

# ╔═╡ 43082966-3134-4dbd-a2f2-bb8a8496b4f7
md"## Braak Stages of Tau protein

In most AD cases, τP follows a predictable pattern of spreading, 			starting in the entorhinal cortex before spreading through the hippocampal 			regions, lateral cortex and finally into the neocortex. Atrophy tends to closely follow the spreading pattern of Tau, more so than that of Aβ"

# ╔═╡ 61359800-edcf-4734-af95-d4913e9cdbe1
html"""<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/braak3.png?raw=true" height=450 width=650 hspace=100>"""

# ╔═╡ 58924c2a-1c48-4979-99af-ad1e5129069b
md"# Modelling: What Is It and Why do We Care? 

Mathematical models are an unreasonably effective tool for understanding complex processes with some basic ingredients and assumptions. In the case of AD, there are sufficiently many properties, such as tau propogation, that lend themselves to methods in mathematical biology.

(It's also what my funding grant is for)"

# ╔═╡ edeeaa2a-4126-48df-9204-c7c6358cc81c
md"## The Heterodimer Model"

# ╔═╡ 445289fb-2a9e-4fad-b0a0-ddef6ed62f7b
html"""
Recall the autocatalytic process
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/heterodimerkinetics.png?raw=true" height=150 width=350 vspace=10 hspace=250>"""

# ╔═╡ 4200bb84-250d-4a4d-9c91-5a89d136fdf9
md"
We can describe this process with the following reaction-diffusion equations, where the rates $$k_{ij}$$ correspond to the rates above.

>```math
>\begin{align}
>\frac{∂\mathbf{p}}{∂ t} &=  ∇⋅\left(\mathbf{D}∇ \mathbf{p} \right) +  k_0 &- k_1 \mathbf{p} - k_{12}\mathbf{p} \mathbf{\hat{p}}  \\
>\frac{∂\mathbf{\hat{p}}}{∂ t} &= ∇⋅\left(\mathbf{D}∇ \mathbf{\hat{p}} \right) &- \hat{k}_1 \mathbf{\hat{p}} + k_{12}\mathbf{p}\mathbf{\hat{p}}
>\end{align}
>```

"

# ╔═╡ f321b372-fe16-4742-8b37-e1ba4fcaf6a3
cite("Fornari, S., Schäfer, A., Jucker, M., Goriely, A. and Kuhl, E., 2019. Prion-like spreading of Alzheimer’s disease within the brain’s connectome. Journal of the Royal Society Interface, 16(159), p.20190356.")

# ╔═╡ f6d05fd3-3662-4afc-b348-3552d5acaf47
md" ## The FKPP Model"

# ╔═╡ 15cc2879-d8b1-447b-8f82-30b97a49bd5b
html"""
The protein dynamics (here shown for a single species) can be simplified in the following way: 
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/fkppkinetics.png?raw=true
" height=150 width=350 vspace=10 hspace=250>"""

# ╔═╡ 00db547f-8a71-4cbc-bb89-d96cdbb90446
md"""
With an appropriate diffusion term, we arrive at:

> $$\frac{∂\mathbf{p}}{∂ t} =  ∇⋅\left(\mathbf{D}∇ \mathbf{p} \right) + α\mathbf{p}(1 - \mathbf{p})$$
"""

# ╔═╡ 4eccecce-c6bb-4abc-97e8-7ffe6147c4a6
cite("Fornari, S., Schäfer, A., Jucker, M., Goriely, A. and Kuhl, E., 2019. Prion-like spreading of Alzheimer’s disease within the brain’s connectome. Journal of the Royal Society Interface, 16(159), p.20190356.")

# ╔═╡ 90d22568-db74-4040-a0e4-56e61896ac50
md"## Avoiding The Continuum
We typically want to avoid the heavy computations necessary to solve continuous equations on a sufficiently fine mesh. We can do this by assuming that all protein diffusion necessary for AD pathology can be described by transport through axonal pathways. With such an assumption, we can transform our continuum equation: 

> $$\frac{∂\mathbf{p}}{∂ t} =  ∇⋅\left(\mathbf{D}∇ \mathbf{p} \right) + α\mathbf{p}(1 - \mathbf{p})$$ 

To a discrete equation: 

> $$\frac{∂ \mathbf{p}_i}{∂t} = -\rho\sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j + \alpha \mathbf{p}_i\left(1-\mathbf{p}_i\right)$$

Where L is the *Laplacian* matrix.
" 

# ╔═╡ 466d2efa-0557-4b2e-910d-ea566b88c4b0
md"## Hey, Where Did You Get Your Laplacian?

This brings with it another problem: finding the Laplacian matrix. Since the Laplacian is $$L = A - D$$, we can construct it from a connectivity matrix, commonly obtained using tractography methods. However, these can also introduce a source of uncertainty into the modelling process..." 

# ╔═╡ cc91524e-fe69-4bf0-9a61-186d6e3f4e6b
A = deserialize("assets/graphs/A.jls");

# ╔═╡ edd31b8d-c76d-45ad-a068-7754c17348fe
begin
	two_cols(Plots.heatmap(A, c = :viridis, size=(350, 300), ticks=nothing, leg=false), html"""
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/connectome.png?raw=true" height=250 width=300>""")
end

# ╔═╡ bdda254a-3479-4079-9181-b5e4f119246c
md"# Uncertainty in Connectomes: The Laplacian"

# ╔═╡ 8cce04aa-dffe-40e0-9668-b52c55cae624
html"""<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/braid1.png?raw=true" height=350 width=650 hspace=100>"""

# ╔═╡ cf26119c-b472-4575-80c0-dedcc585191c
cite("Putra, P., Thompson, T.B. and Goriely, A., 2021. Braiding Braak and Braak: Staging patterns and model selection in network neurodegeneration. bioRxiv.")

# ╔═╡ 894f4981-7675-4b76-abcb-f4d5ed4056ca
md"# Uncertainty in Connectomes: Tractography"

# ╔═╡ ed86b84e-b6cd-4bb2-a45e-f32e01682db0
html"""<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/braid2.png?raw=true" height=350 width=650 hspace=100>"""

# ╔═╡ 73af0e1e-2887-4e7e-add3-4bd067dff0de
cite("Putra, P., Thompson, T.B. and Goriely, A., 2021. Braiding Braak and Braak: Staging patterns and model selection in network neurodegeneration. bioRxiv.")

# ╔═╡ 58c62223-a0f6-4ef2-8327-a7ad1cfa0f16
md"# Simulating the models

We can (and should) simulate the models to see if the dynamics the models produce are aligned with the expected behaviour of AD pathology"

# ╔═╡ 9a2eccbe-5ce3-4301-9832-44d3deddd22e
L = deserialize("assets/graphs/L.jls");

# ╔═╡ bb0056fb-624a-4ee7-9127-4b2537561a65
function NetworkFKPP(du, u, p, t)
	du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
end;

# ╔═╡ 010ac87d-b006-4f8e-93b4-e645694d0f44
begin
	u0 = zeros(83)
	u0[[37, 68]] .= 0.1
		
	p = [1.0, 1.0]
	t_span = (0.0,20.0)
		
	problem = ODEProblem(NetworkFKPP, u0, t_span, p)
		
	sol = solve(problem, Tsit5())
end;

# ╔═╡ 77e69101-b1ee-493c-a39c-a19378814976
function simulate(prob, p)
	solve(remake(prob, p=p), Tsit5())
end;

# ╔═╡ 505c6886-6034-4808-9190-61302c50aa2c
md"## FKPP Dynamics"

# ╔═╡ f1a47d6a-c20c-459b-87cf-813df614363b
two_cols(md"
$$\frac{d \mathbf{p}_i}{dt} = -\rho\sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j + \alpha \mathbf{p}_i\left(1-\mathbf{p}_i\right)$$", 
md"
ρ = $(@bind ρ Slider(0:0.05:3, show_value=true, default=1))

α = $(@bind α Slider(0:0.05:3, show_value=true, default=1))
")

# ╔═╡ df8c6488-9a4b-4f32-ac4f-56488c7d6d3d
begin

TwoColumn( 		
md"
The simulation to the right shows the dynamics of the FKPP model with seeding in the entorhinal cortex.

By varying the parameters, we can see how the dynamics change and heuristically determine regions of high sensitivity.
		
Notice how there are larger changes at smaller values of ρ and α. This wil be important later...
", 	

Plots.plot(simulate(problem, [ρ, α]), size=(450,300), labels=false, ylims=(0.0,1.0), xlims=(0.0,20.0)))
end

# ╔═╡ e7888f18-4f6d-45e1-9499-7e7fd298ba96
function NetworkHeterodimer(du, u, p, t)
	x = u[1:83]
	y = u[84:166]
	du[1:83] .= -p[1] * L * x .+ p[2] .- p[3] .* x .- p[4] .* x .* y
	du[84:166] .= -p[1] * L * y .- p[5]*y .+ p[4] .* x .* y
end;

# ╔═╡ 779cbc2e-f5c1-4b51-b415-7c8d1e32ef5d
md"## Heterodimer Dynamics"

# ╔═╡ 523c9d21-5a76-4f82-bdd5-2cd38e654786
two_cols(md"
$$\frac{d \mathbf{p}_i}{dt} = -\rho\sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{p}_j +  k_0 - k_1 \mathbf{p} - k_{12}\mathbf{p} \mathbf{\hat{p}}$$
$$\frac{d \mathbf{\hat{p}}_i}{dt} = -\rho\sum\limits_{j=1}^{N}\mathbf{L}_{ij}^{\omega}\mathbf{\hat{p}}_j - \hat{k}_1 \mathbf{\hat{p}} + k_{12}\mathbf{p}\mathbf{\hat{p}}$$

", md"
ρ₁ = $(@bind ρ1 Slider(0:0.1:1, show_value=true, default=0.1))

k₀ = $(@bind k0 Slider(0:0.1:1, show_value=true, default=0.1))

k₁ = $(@bind k1 Slider(0:0.1:1, show_value=true, default=0.1))

k₁₂ = $(@bind k12 Slider(0:0.1:1, show_value=true, default=0.1))

k̂₁ = $(@bind k̂1 Slider(0:0.1:1, show_value=true, default=0.1))
")

# ╔═╡ dc25cc6b-a238-4111-9b20-ece5dea74f1e
begin
	u_0 = zeros(166)
	u_0[1:83] .= k0/k1
	u_0[[110, 151]] .= 0.1
	
	prob1 = ODEProblem(NetworkHeterodimer, u_0, (0.0,50.0), [ρ1, k0, k1, k12, k̂1])
	sol1 = solve(prob1, Tsit5())	
end;

# ╔═╡ 10cb25cf-45e3-454d-b116-7a9c9161f416
begin
	l = @layout [a b]
		
	p1 = plot(sol1, vars=(1:83), labels=false, xlabel="t", title="[p]")
	p2 = plot(sol1, vars=(84:166), labels=false, xlabel="t", title="[p̂]")
	
	plot(p1, p2, layout=l, xlims=(0,50), ylims=(0,2), size=(900,250))
end

# ╔═╡ 5ec1e4ca-bbd8-45c0-a8e3-fc6f42e043f1
md"# Inference!

Now that we have some models, how do we fit them to data?

Or, more importantly, how do we fit them to data **and** account for sources of uncertainty?"

# ╔═╡ c23df8a1-6f19-4c7b-90ff-3bc92515028f
md"## Inverse Problems using Bayes-Price-Laplace
\
For observations $\mathbf{x} = x_{1:n}$ and latent variables  $\mathbf{\theta} = \theta_{1:m}$, we have a joint density

$$p(\mathbf{x}, \mathbf{\theta})$$

To evalulate a particular hypothesis, we need to evaluate the posterior $p(\mathbf{\theta} \mid \mathbf{x})$, thus we decompose the joint distribution:

$$p(\mathbf{x}, \mathbf{\theta}) = p(\mathbf{x} \mid \mathbf{\theta})p(\mathbf{\theta}) = p(\mathbf{\theta} \mid \mathbf{x})p(\mathbf{x})$$

Dividing through by the *evidence*, we obtain the Bayes-Price-Laplace rule: 

>$$p(\mathbf{\theta} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathbf{\theta})p(\mathbf{\theta})}{p(\mathbf{x})}$$
"

# ╔═╡ 23dcd614-9f93-4535-8f98-cb263ee497f9
md"## Reverand, What Does it Mean?
\
\

>$$\underbrace{p(\mathbf{\theta} \mid \mathbf{x})}_{posterior} = \frac{\overbrace{p(\mathbf{x} \mid \mathbf{\theta})}^{likelihood}\overbrace{p(\mathbf{\theta})}^{prior}}{\underbrace{p(\mathbf{x})}_{evidence}}$$

- Likelihood: Probability that a particular set of parameter values generate the observations.
- Prior: Probability representing our initial beliefs about the parameter values.
- Evidence: Normalising factor; probability of observing our data (given our model). Otherwise known as the marginal likelihood.
- Posterior: Probability that some data are _caused_ by some set of parameters.
"

# ╔═╡ 5b923ad7-4396-4c65-97a8-447e2237fe5c
md"## Why is Bayesian Inference Hard? Because Integration is hard! 

It's almost always the case that we cannot do Bayesian inference analytically. The principal reason for this comes from the evidence term: 

$${p(\mathbf{x})} = \int p(\mathbf{x} , \mathbf{\theta}) d\mathbf{\theta}$$

As θ becomes larger, the complexity of integration grows exponentially with the dimensionality. Hence, naive numerical integration (such as quadrature) becomes computationally infeasible. 
 
"

# ╔═╡ cbeb9b8c-3392-4a6e-a29c-f002fca5ae6d
html"""
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/dimensionality.png?raw=true" height=200 width=800 hspace=30>"""

# ╔═╡ a9c271d3-51fc-4470-83cb-cbf82a764fde
md"Thus, to do Bayesian inference, we either need some approximate methods or some clever algorithms for exploring posterior space and integrating wisely. I'll briefly discuss two classes of methods: variational inference and Hamiltonian Monte Carlo."

# ╔═╡ d54223c7-58b3-4f2a-bc2d-8e0d8d617184
md"# Variational Inference: Does My Posterior Look Big in this Distribution?
 
Variational inference aims to circumvent the large time complexity associated with inferring complicated models by casting the problem as one of optimisation. 

Consider a contrived density, Ω for latent variables θ. Within this family of distirbutions, we wish to find an a set of values that minimise the the Kullback-Leibler divergence to the true posterior: 

>$$q^{*}(\mathbf{\theta}) = \underset{q(\mathbf{\theta}) \in \mathfrak{Ω}}{argmin} \mathrm{KL}(q(\mathbf{\theta}) \mid \mid p(\mathbf{\theta} \mid \mathbf{x}))$$

However, this still depends on the intractable posterior..."


# ╔═╡ 67585da0-6bc8-44c6-952e-4ac7205cfa3a
md"## More Intuition...
"

# ╔═╡ c17c99cc-f3fd-49a8-aeef-cfac86c10418
html"""<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/vi.png?raw=true" height=400 width=700 hspace=70>"""

# ╔═╡ 863a7ee0-563c-49ab-8311-1e8b364781bd
md"## Making the Problem Tractable

We can show this dependence explicitly: 
>$$\mathrm{KL}(q(\mathbf{θ}) \mid \mid p(\mathbf{θ} \mid \mathbf{x})) = \mathbb{E}[\log q(\mathbf{\mathbf{θ}})] - \mathbb{E}[\log p(\mathbf{x} \mid \mathbf{θ})] - \mathbb{E}[\log p(\mathbf{θ})] + \log p(\mathbf{x})$$


We can then simply rearrange things slightly and drop the evidence term to give us an interpretable cost function, the Evidence Lower Bound (ELBO) or the Free Energy.

>```math
>\begin{align}
>-\mathrm{KL}(q(\mathbf{θ}) \mid \mid p(\mathbf{θ} \mid \mathbf{x})) &> \mathbb{E}[\log p(\mathbf{x} \mid \mathbf{θ})] - \mathbb{E}[\log q(\mathbf{θ})] + \mathbb{E}[\log p(\mathbf{θ})] \\
>\mathbf{F} &= \underbrace{\mathbb{E}[\log p(\mathbf{x} \mid \mathbf{θ})]}_{accuracy} - \underbrace{\mathrm{KL}(q(\mathbf{θ}) \mid \mid p(\mathbf{θ}))}_{complexity}
>\end{align}
>```

In practice, this optimisation is performed either through analytic update rules using variational optimisation or, in a more modern way that leverges modern computational science, using automatic derivatives and normalising flows/bijectors, an approach known as ADVI (automatic differentiation variational inference). 
"

# ╔═╡ d6348a05-2238-452c-9741-9ced8732d5de
cite("Blei, D.M., Kucukelbir, A. and McAuliffe, J.D., 2017. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), pp.859-877.")

# ╔═╡ 699553ca-c691-4f6e-a2e7-73e325c5a2c4
md"# Hamiltonian Monte Carlo 

MCMC methods allow one to sample from the posterior, as opposed to approximating it. Tradiational algortihms like Metroplis-Hastings are good for low dimensional problems, but don't sacle well. HMC allows one to take efficient steps during sampling, even in high dimensional space
"

# ╔═╡ 1df0368b-8a18-4278-800b-7402ab02c0b8
md"## Metropolis-Hastings: Brief Review
HMC follows a similar structure to the widely used Metroplis-Hastings algorithm. Let's briefly review it and identify the weak points. 

```julia
for i in 1:Number of iterations 
	θₚ ~ N(θ, Σ)		#Draw a proposal sample
	r = α(θ, θₚ)		#Calculate the acceptance probability based on a likelihood ratio
	if r ≥ 1 			#Accept if your new proposal increases your likelihood
		θ = θₚ 
	else 				#Else, accept relative to some probaility corresponding to a standard normal 
		x = N(0,1)
		if r > x
			θ = θₚ
		else
			reject θₚ
```

Particular deficiencies stem from the very first step, drawing a proposal, θₚ. Namely, we are exploring posterior sapace in a mostly random way! 

This is fine for small problems, but as dimensions increase and correlations between variables increase, the probability of making successful proposals drops significantly. 

The main advantage of Hamiltonian Monte Carlo algorithms is that that make informed samples that are very likely to be accepted.
"


# ╔═╡ 13320401-6bba-4c0c-9b60-f2a3d2134051
md"## Hamiltonian Monte Carlo Algorithm 

We can modify the algorithm to be the following: 

Expand your paramter space two fold by introducing an auxilliary momentum parameters, p, and forming a joint density with θ, -logπ(θ, p), and Hamiltonian, H(θ, p). 

Adding this structure allows us to make more informed proposal steps. 

```julia
for i in 1:Number of iterations 

	pₚ ~ Normal(0, M)		#Sample a new p from proposal distribution with std M (the mass matrix/metrix)
	θₚ <- H(θ, pₚ) 			#Generate a new θ by solving the Hamiltonian over time LΔ.

	r = α((θ,p), (θₚ, pₚ)) 	#Calculate the acceptance probability given a likelihood ratio
	if r ≥ 1 				#if the likelihood is increased, accept the new parameters 
		(θ, p) = (θₚ, pₚ)
	else 					#else, reject them and start again
		reject
	
```
"

# ╔═╡ cc3c3f2a-7b98-4391-8987-893ea7c31d7d
md"# More Intuition..."

# ╔═╡ 9c2f2fbe-95ce-4a58-951e-b096ac1e6697
html"""
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/gradient.png?raw=true" height=350 width=400 hspace=210>"""

# ╔═╡ 56d05008-f0a6-43c5-91a4-a879aa91ef3e
md"# More Intuition..."

# ╔═╡ f2b45189-6ee4-419f-9eb2-e22e9cc1d45f
html"""
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/gradient1.png?raw=true" height=370 width=400 hspace=210>"""

# ╔═╡ e97aa7b5-e766-4157-ac00-ec8801e56a6e
md"# More Intuition..."

# ╔═╡ 2b20d604-b105-4f20-a232-99fe92b68f84
html"""
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/satellite.png?raw=true" height=350 width=700 hspace=100>"""

# ╔═╡ 484d5613-5a07-4897-b181-cc0e20e1db1f
cite("Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo. arXiv preprint arXiv:1701.02434.")

# ╔═╡ b74abac6-dae1-409a-97ec-0b2e2940298f
md"# More Detail

By adding a nother parameter, p, the momentum, we lift from our target space in \theta, to a joint density π(\theta, p). We define a corresponding Hamiltonian density using the *canonical distribution* 

>$$H(\theta, p) = -\log π(\theta, p)$$

Which decomposes into: 
>```math
>\begin{align}
>H(\theta, p) &= -\log π(\theta, p) - \log π(\theta) \\
> &= \underbrace{K(\theta, p)}_{\text{Kinetic Energy}} + \underbrace{V(\theta)}_{\text{Potential energy}}
>\end{align}
>```

As a Hamiltonian, these of course satisfy Hamilton's equations and thus we obtain our desired vector field over parameter space (or more particular, the embedding of parameter space in Hamiltonian phase space). 

>```math
>\begin{align}
>\frac{d\theta}{dt} &= \frac{\partial H}{\partial p} \\
>\frac{dp}{dt} &= - \frac{\partial H}{\partial \theta} 
>\end{align}
>```

Notice that the time derivative of momentum $p$ depends on the gradients of $\theta$. we can make this more explict. 

>```math
>\begin{align}
>\frac{dp}{dt} &= - \frac{\partial K}{\partial \theta} - \frac{\partial V}{\partial \theta}
>\end{align}
>```

We see that the time deriative of $p$ contains exactly the information we want, namely the gradients the negative log posterior with our parameters $\theta$. Thus, be adding momentum parameters in this particular way, we create a way to move sample phase space in such a way that depends on gradients of our parameters of interest.
"

# ╔═╡ 76b25ab5-1d67-4736-8753-0eaa26ab77c1
cite("Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo. arXiv preprint arXiv:1701.02434.")

# ╔═╡ a28f219d-1f29-467c-b8f3-796b2d43c1d3
md"# Show me the Results! 

'Pavan, have you actually done any *real* work?'
"

# ╔═╡ d6b9f9d6-04f4-4563-a930-01b182ec8886
begin
	datadir = "/Users/pavanchaggar/Projects/TauPet/"
	suvr_mat = matread(datadir * "tau_code_python/suvr_ADNI_scaled.mat");
	suvr = suvr_mat["ord_suvr"];
		age_mat = matread(datadir * "tau_code_python/ages_ADNI.mat");
	age = age_mat["ages_mat"];
	min_age, max_age = extrema(age[age.>0])
	tₛ = clamp.(age .- min_age, 0, max_age - min_age);
	t_index = tₛ .!= 0.0;
end;

# ╔═╡ 338224ca-727c-4979-a980-b2389be6b1ad
p3 = histogram(tₛ[:,1] .+ min_age, bins=10, xlabel="Age at first scan", labels=false, alpha=0.7, size=(300,300));

# ╔═╡ 6cf4cc92-705a-479c-8eb3-e6957e26592d
md"## Comparing Inference Methods for Patient Data

Ultimately, for our dynamical systems $f(u0, θ, t)$, we want to infer likely values of θ. The dynamical systems proposed earlier in the talk describe the flow of proteins on a network. Thus, we need data that is apprpriate for the model. 

Fortunately, such data is available from the Alzheimer's Disease Neuroimaging Initiative (ADNI) in the form of Aβ and τP PET imaging. 

"

# ╔═╡ fa5c1d3e-8192-4960-a5fb-e2707bfe3c4a
two_cols(md"",md"""
subject = $(@bind subject Slider(1:78, show_value=true, default=1))
""")

# ╔═╡ 3827c858-3072-4df6-a204-c6ba90f9cddb
begin
	p4 = scatter(age[subject,t_index[subject,:]], suvr[:,t_index[subject,:],subject]', label=false, xlims=(min_age-5, max_age+5), ylims=(0.0,0.5), color=:grey, size=(550,290))
	scatter!(age[subject,t_index[subject,:]], suvr[[27,68],t_index[subject,:],subject]', label=false, xlims=(min_age-5, max_age+5), ylims=(0.0,0.5), color=:red)
	ylabel!("SUVR")
	xlabel!("Age")
end;

# ╔═╡ f8721c1c-ef12-4650-a35c-b219f49f8951
TwoColumn(p3,p4)

# ╔═╡ 7dd55ce8-936d-4568-b983-53a1c9222e17
md"## Defining a Probabalistic Model

Now that we have data, we should want to fit a model to it using Bayesian inference. We assume the following model structure: 


$$\mathbf{x} = f(\mathbf{u0}, θ) + \mathcal{N}(0, \sigma)$$

We're going to fit the Network FKPP model, the priors for which will be: 

$$\sigma \approx \Gamma^{-1}(2, 3)$$ 
$$\rho \approx \mathcal{N}^{+}(0,5)$$
$$\alpha \approx \mathcal{N}(0,5)$$
"

# ╔═╡ 2c8d2614-0c1e-4bcc-8c52-5c583df6a443
begin
	AB_mat = readdlm(datadir * "tau_code_python/amyloid_status.txt")
	
	pos = findall(x -> x==1, AB_mat[:,2])
	n_pos = length(pos)
	
	neg = findall(x -> x==0, AB_mat[:,2])
	n_neg = length(neg)
	
	na = findall(x -> x==2, AB_mat[:,2])
	n_na = length(na)
end;

# ╔═╡ 608aa6c1-4f33-49ab-a189-96a567580b03
begin
	chaindir = datadir * "chains/"
	chain_vector = Vector{Any}(undef, 78)
	q_vector = Vector{Any}(undef, 78)
	
	for i in 1:78
		chain_vector[i] = deserialize(chaindir * "nuts/" * "posteriorchain_$(i).jls")
		q = deserialize(chaindir * "advi/" * "q_$(i).jls")
		q_vector[i] = rand(q, 1000)
	end
	
end;

# ╔═╡ d730c2c7-b34a-48e9-9f8a-212b9c813fc7
md"## Results
We first ran a NUTS sampler on all single subjects, seperating them based on whether they were Aβ+ or Aβ-. The distributions for the diffusion coefficient, ρ, indicate a *very* slow rate of diffusion, on the order of mm/year, with no substantial differences between Aβ groups. 

On the other hand, the distributions for the growth rate, α, are largely centered around 0, displaying some multimodality for the growth rate parameters."

# ╔═╡ d5faa28b-a168-48d2-ba3a-cf4478a3e920
begin
	fig = plot(chain_vector[1][:k], seriestype = :density, labels=false, color=:white, size=(400,300))
	for i in neg
		plot!(fig, chain_vector[i][:k], seriestype = :density, labels=false, color=:red, linewidth=1)
	end
	for i in pos
		plot!(fig, chain_vector[i][:k], seriestype = :density, labels=false, color=:grey, linewidth=1, alpha=0.7)
	end
	title!("Diffusion coefficient")
	
	fig_a = plot(chain_vector[1][:a], seriestype = :density, labels=false, color=:white, size=(400,300))
	for i in neg
		plot!(fig_a, chain_vector[i][:a], seriestype = :density, labels=false, color=:red, linewidth=1)
	end
	for i in pos
		plot!(fig_a, chain_vector[i][:a], seriestype = :density, labels=false, color=:grey, linewidth=1, alpha=1.0)
	end
	title!("Growth Rate")
	fig_a

	two_cols(fig, fig_a)
end

# ╔═╡ e881d57e-4006-4894-b296-3f4eb576566e
md"## Comparison with mean-field Variational Inference"

# ╔═╡ 5a200baa-1539-4130-bca7-c60fe6574b1a
md"""
##### sub = $(@bind sub Slider(1:78, show_value=true, default=1))
"""

# ╔═╡ c26c2bda-8c86-4eb3-ab6b-5806866184fa
begin
	p5 = plot(chain_vector[sub][:k], seriestype= :density, xlims=(-0.5,0.5), ylims=(0,100), labels="NUTS: Diffusion", linewidth=3, color=:navy, size=(450,300))
	plot!(chain_vector[sub][:a], seriestype= :density, xlims=(-0.5,0.5), ylims=(0,100), label="NUTS: Growth", linewidth=3, color=:darkred)
	plot!(q_vector[sub][2,:], seriestype= :density, linewidth=3, label="VI: Diffusion", color=:skyblue)
	plot!(q_vector[sub][3,:], seriestype= :density, linewidth=3, label="VI: Growth", color=:coral)
	xlabel!("Parameter value")
end;

# ╔═╡ f064ba24-46dc-4d68-b390-2c4877023ce0
begin
	p6 = scatter(age[sub,t_index[sub,:]], suvr[:,t_index[sub,:],sub]', label=false, xlims=(min_age-5, max_age+5), ylims=(0.0,0.5), color=:grey, size=(400,300))
	scatter!(age[sub,t_index[sub,:]], suvr[[27,68],t_index[sub,:],sub]', label=false, xlims=(min_age-5, max_age+5), ylims=(0.0,0.5), color=:red)
	ylabel!("SUVR")
	xlabel!("Age")
end;

# ╔═╡ 47e37597-f273-4454-9b95-059c8c348206
TwoColumn(p5, p6)

# ╔═╡ 0ba3e43d-a505-42ba-88a4-371f586c41de
md"## Projecting Uncertainty Forward

A key advantage of Bayesian modelling is the quantification of uncertainty. Using the estimates of parameter distributions, we can project forward in time to simulate potential disease outcomes."

# ╔═╡ 00755cfc-8315-4627-86ac-b975457cd133
md"""
##### sub = $(@bind sub1 Slider(1:78, show_value=true, default=1))
"""

# ╔═╡ 5eb8957d-a450-483c-a37f-cf11886d1c38
begin
	postchain = chain_vector[sub1]
	ks = Array(postchain[:k])
	as = Array(postchain[:a])
	probpost = ODEProblem(NetworkFKPP, suvr[:,1,sub1], (0.0,30.0), [mean(ks),mean(as)])
	resolpost = solve(probpost, Tsit5(), abstol=1e-9, reltol=1e-6)
	
	p8 = plot(xlims=(0,30), ylims=(0,1), size=(500,350))	
	reltime1 = tₛ[sub1,t_index[sub1,:]] .- minimum(tₛ[sub1,t_index[sub1,:]])
	scatter!(reltime1, suvr[:,t_index[sub1,:],sub1]', label=false, color=:grey)
	scatter!(reltime1, suvr[[27,68],t_index[sub1,:],sub1]', label=false, color=:red)
	for i in 1:100
		n = rand(1:1000)
		prob = remake(probpost, p=[ks[n], as[n]])
		resol = solve(prob, Tsit5(), abstol=1e-9, reltol=1e-6)
		plot!(p8, resol.t, resol[27,:], alpha=0.5, color = "#BBBBBB", legend = false)
		plot!(p8, resol.t, resol[68,:], alpha=0.5, color = "#BBBBBB", legend = false)
	end
	plot!(p8, resolpost.t, resolpost[27,:], linewidth=3, alpha=0.5, color = :red, legend = false)
	plot!(p8, resolpost.t, resolpost[68,:], linewidth=3, alpha=0.5, color = :red, legend = false)
	xlabel!("Time")
	ylabel!("Protein Concentration")
#	scatter!(reltime1, suvr[node,:,sub1], legend = false)
end;

# ╔═╡ 37d0f4f1-f6a1-434c-94d8-47bbde4e5723
begin
	p7 = plot(chain_vector[sub1][:k], seriestype= :density, xlims=(-0.5,0.5), ylims=(0,100), labels="NUTS: Diffusion", linewidth=3, color=:navy, size=(350,360))
		plot!(chain_vector[sub1][:a], seriestype= :density, xlims=(-0.5,0.5), ylims=(0,100), label="NUTS: Growth", linewidth=3, color=:darkred)
	xlabel!("Parameter value")
end;

# ╔═╡ 74722be9-9c40-4511-9658-21e6e23590c1
TwoColumn(p8, p7)

# ╔═╡ 3731dc46-5f04-4bd0-8ac1-c8e4011d2cdc
md"# Hierarchical Inference?"

# ╔═╡ c8da718a-8286-4f6d-967a-bd329d68ae9b
md"# Integrating it All Together

There are a number of source of uncertainty present in the modelling process. It will be especially important to consider uncertainty in: 
- Tractography and the graph Laplacian 
- Model structure
- Parametric uncertainty and measurement error
" 

# ╔═╡ 1e5a4fd7-424d-47b7-8288-59b183e626ee
md"# Challenges and Next Steps
The most immediate step will be toward hierarchical modelling and structural/model uncertainty, addressing the question of model selection. This will help determine at least two important factors:
1) Whether one model is preferred over another model 
2) Whether there is enough information in the data to distinguish between two dynamic models

Following this, the next big source of uncertainty to quantify, and hopefully resolve, will be that of brain connectivity. We have seen that different graph structures yield significantly different dynamics, determining which features of graph topology are important for modelling the diffusion process will greatly reduce uncertainty and is a necessary step before such work is used for clinical purposes.
"

# ╔═╡ 31f3827b-a7c6-4428-b3b0-d7ca3b15de2a
md"## References and Resources

Fornari, S., Schäfer, A., Jucker, M., Goriely, A. and Kuhl, E., 2019. Prion-like spreading of Alzheimer’s disease within the brain’s connectome. Journal of the Royal Society Interface, 16(159), p.20190356.

Putra, P., Thompson, T.B. and Goriely, A., 2021. Braiding Braak and Braak: Staging patterns and model selection in network neurodegeneration. bioRxiv.

Blei, D.M., Kucukelbir, A. and McAuliffe, J.D., 2017. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518), pp.859-877.

Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo. arXiv preprint arXiv:1701.02434."

# ╔═╡ b5feb01d-ff0d-4909-864b-cc89cf2b1ff8
jpeg_joinpathsplit__FILE__1assetsimagejpeg = let
    import PlutoUI
    PlutoUI.LocalResource(joinpath(split(@__FILE__, '#')[1] * ".assets", "image.jpeg"))
end

# ╔═╡ 720418e0-241a-49e3-98b6-c1c3d231d476
begin
	using PlutoUI
	using Plots 
	using StatsPlots
	using PlotThemes
	using DifferentialEquations
	using Turing
	using Images
	using HypertextLiteral
	using Connectomes
	using MAT
	using Serialization
	using DelimitedFiles

	gr()
	theme(:ggplot2)
end

# ╔═╡ Cell order:
# ╠═720418e0-241a-49e3-98b6-c1c3d231d476
# ╠═860ff68f-634c-466f-b10b-a20010231418
# ╠═edcd2286-c11a-43e3-a1fa-32a71998f5f0
# ╟─8a2f8a5f-8333-4ebd-8e3f-f37892bb3a29
# ╟─c3860233-3c3a-46a7-9f13-a6af2d6c71cd
# ╟─0a436ace-bd7e-4e80-9527-b0580f749bfc
# ╟─9c94781c-e244-4222-8c63-9485628cccd9
# ╟─fd6af805-4180-4625-a931-cecf5e40bc03
# ╟─41c083ee-5a88-48cb-9b99-9b2eb1415a24
# ╟─43082966-3134-4dbd-a2f2-bb8a8496b4f7
# ╟─61359800-edcf-4734-af95-d4913e9cdbe1
# ╟─58924c2a-1c48-4979-99af-ad1e5129069b
# ╟─edeeaa2a-4126-48df-9204-c7c6358cc81c
# ╟─445289fb-2a9e-4fad-b0a0-ddef6ed62f7b
# ╟─4200bb84-250d-4a4d-9c91-5a89d136fdf9
# ╟─f321b372-fe16-4742-8b37-e1ba4fcaf6a3
# ╟─f6d05fd3-3662-4afc-b348-3552d5acaf47
# ╟─15cc2879-d8b1-447b-8f82-30b97a49bd5b
# ╟─00db547f-8a71-4cbc-bb89-d96cdbb90446
# ╟─4eccecce-c6bb-4abc-97e8-7ffe6147c4a6
# ╟─90d22568-db74-4040-a0e4-56e61896ac50
# ╟─466d2efa-0557-4b2e-910d-ea566b88c4b0
# ╟─cc91524e-fe69-4bf0-9a61-186d6e3f4e6b
# ╟─edd31b8d-c76d-45ad-a068-7754c17348fe
# ╟─bdda254a-3479-4079-9181-b5e4f119246c
# ╟─8cce04aa-dffe-40e0-9668-b52c55cae624
# ╟─cf26119c-b472-4575-80c0-dedcc585191c
# ╟─894f4981-7675-4b76-abcb-f4d5ed4056ca
# ╟─ed86b84e-b6cd-4bb2-a45e-f32e01682db0
# ╟─73af0e1e-2887-4e7e-add3-4bd067dff0de
# ╟─58c62223-a0f6-4ef2-8327-a7ad1cfa0f16
# ╟─9a2eccbe-5ce3-4301-9832-44d3deddd22e
# ╟─bb0056fb-624a-4ee7-9127-4b2537561a65
# ╟─010ac87d-b006-4f8e-93b4-e645694d0f44
# ╟─77e69101-b1ee-493c-a39c-a19378814976
# ╟─505c6886-6034-4808-9190-61302c50aa2c
# ╟─df8c6488-9a4b-4f32-ac4f-56488c7d6d3d
# ╟─f1a47d6a-c20c-459b-87cf-813df614363b
# ╟─e7888f18-4f6d-45e1-9499-7e7fd298ba96
# ╟─dc25cc6b-a238-4111-9b20-ece5dea74f1e
# ╟─779cbc2e-f5c1-4b51-b415-7c8d1e32ef5d
# ╟─10cb25cf-45e3-454d-b116-7a9c9161f416
# ╟─523c9d21-5a76-4f82-bdd5-2cd38e654786
# ╟─5ec1e4ca-bbd8-45c0-a8e3-fc6f42e043f1
# ╟─c23df8a1-6f19-4c7b-90ff-3bc92515028f
# ╟─23dcd614-9f93-4535-8f98-cb263ee497f9
# ╟─5b923ad7-4396-4c65-97a8-447e2237fe5c
# ╟─cbeb9b8c-3392-4a6e-a29c-f002fca5ae6d
# ╟─a9c271d3-51fc-4470-83cb-cbf82a764fde
# ╟─d54223c7-58b3-4f2a-bc2d-8e0d8d617184
# ╟─67585da0-6bc8-44c6-952e-4ac7205cfa3a
# ╟─c17c99cc-f3fd-49a8-aeef-cfac86c10418
# ╠═b5feb01d-ff0d-4909-864b-cc89cf2b1ff8
# ╟─863a7ee0-563c-49ab-8311-1e8b364781bd
# ╟─d6348a05-2238-452c-9741-9ced8732d5de
# ╟─699553ca-c691-4f6e-a2e7-73e325c5a2c4
# ╟─1df0368b-8a18-4278-800b-7402ab02c0b8
# ╟─13320401-6bba-4c0c-9b60-f2a3d2134051
# ╟─cc3c3f2a-7b98-4391-8987-893ea7c31d7d
# ╟─9c2f2fbe-95ce-4a58-951e-b096ac1e6697
# ╟─56d05008-f0a6-43c5-91a4-a879aa91ef3e
# ╟─f2b45189-6ee4-419f-9eb2-e22e9cc1d45f
# ╟─e97aa7b5-e766-4157-ac00-ec8801e56a6e
# ╟─2b20d604-b105-4f20-a232-99fe92b68f84
# ╟─484d5613-5a07-4897-b181-cc0e20e1db1f
# ╟─b74abac6-dae1-409a-97ec-0b2e2940298f
# ╟─76b25ab5-1d67-4736-8753-0eaa26ab77c1
# ╟─a28f219d-1f29-467c-b8f3-796b2d43c1d3
# ╟─d6b9f9d6-04f4-4563-a930-01b182ec8886
# ╟─338224ca-727c-4979-a980-b2389be6b1ad
# ╟─3827c858-3072-4df6-a204-c6ba90f9cddb
# ╟─6cf4cc92-705a-479c-8eb3-e6957e26592d
# ╟─f8721c1c-ef12-4650-a35c-b219f49f8951
# ╟─fa5c1d3e-8192-4960-a5fb-e2707bfe3c4a
# ╟─7dd55ce8-936d-4568-b983-53a1c9222e17
# ╟─2c8d2614-0c1e-4bcc-8c52-5c583df6a443
# ╟─608aa6c1-4f33-49ab-a189-96a567580b03
# ╟─d730c2c7-b34a-48e9-9f8a-212b9c813fc7
# ╟─d5faa28b-a168-48d2-ba3a-cf4478a3e920
# ╟─e881d57e-4006-4894-b296-3f4eb576566e
# ╟─47e37597-f273-4454-9b95-059c8c348206
# ╟─5a200baa-1539-4130-bca7-c60fe6574b1a
# ╟─c26c2bda-8c86-4eb3-ab6b-5806866184fa
# ╟─f064ba24-46dc-4d68-b390-2c4877023ce0
# ╟─0ba3e43d-a505-42ba-88a4-371f586c41de
# ╟─74722be9-9c40-4511-9658-21e6e23590c1
# ╟─00755cfc-8315-4627-86ac-b975457cd133
# ╟─5eb8957d-a450-483c-a37f-cf11886d1c38
# ╟─37d0f4f1-f6a1-434c-94d8-47bbde4e5723
# ╟─3731dc46-5f04-4bd0-8ac1-c8e4011d2cdc
# ╟─c8da718a-8286-4f6d-967a-bd329d68ae9b
# ╟─1e5a4fd7-424d-47b7-8288-59b183e626ee
# ╟─31f3827b-a7c6-4428-b3b0-d7ca3b15de2a
