### A Pluto.jl notebook ###
# v0.16.1

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

# ╔═╡ 8d93f866-2a10-4489-a1d4-1ac1da97f248
begin 
	using PlutoUI
	using Plots
	using StatsPlots
	using PlotThemes
	using DifferentialEquations
	using Turing
	using Images
	using HypertextLiteral
	using MAT
	using Serialization
	using DelimitedFiles
end

# ╔═╡ c1e20410-aed5-48a4-8f02-d78f957c15f0
include("functions.jl")

# ╔═╡ 5a0591b7-1a34-4e3e-9862-c772fc3159f4
html"""<style>
main {
max-width: 900px;
}"""

# ╔═╡ 18225564-8512-4fca-87c8-a95ec2fa0d05
html"<button onclick='present()'>present</button>"

# ╔═╡ 4ff67e50-ccdd-479f-8280-e04ab2354ce4
md" 
# Transfer Thesis Viva 

**Pavanjit Chaggar, October 2021**

DPhil student at the Mathematical Institute.
Supervised by Alain Goriely and Saad Jbabdi, with support from Stefano Magon and Gregory Klein at Roche.

Presentation: https://github.com/PavanChaggar/TransferPresentation"

# ╔═╡ abc58f7f-c4c1-47b6-861a-ab679d34bc95
md" 
# Overview and Introduction

- Alzheimer's disease (AD)
- Mathematical models of AD
- Inference Workflow
- Preliminary results
"

# ╔═╡ 95d6223a-c12e-4b26-8c4e-d59a59c7d129
md" 
# Alzheimer's Disease -- A Brief Summary
Alzheimer's is characterised by gradual neurodegeneration associated with the pervasive spreading of toxic proteins.

In particular, two proteins, Amyloid beta (Aβ) and tau-protein (τP), are believed to underlie and drive the development of pathology.

Historically, Aβ was thought to be the primary cause of AD, with research focussing solely on its pathology.

More recent work has focussed on role of τP, in part because it spreads very predictably and is more tightly coupled with atrophy and symptom onset.

For modelling, there are at least two important aspects of protein dynaimcs to consider: growth and diffusion.
"

# ╔═╡ d75eb4e7-2fbf-44ca-af86-bf67fc1d393d
md" 
## A Pernicious Pair of Predictable Prion Proteins
Both Aβ and τP grow via an autocatalytic process resembling those displayed by prions. 
This process is summarised as: 
"

# ╔═╡ a0cb7614-2ab3-44d1-9202-02f19915edf6
html"""
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/heterodimerkinetics.png?raw=true" height=250 width=500 vspace=50, hspace=175>"""

# ╔═╡ f45c2cd6-baf6-4ce3-84b0-5bf8fb9e67d4
md"## Braak Stages of Tau protein
In most AD cases, τP follows a predictable pattern of spreading, starting in the entorhinal cortex before spreading through the hippocampal regions, lateral cortex and finally into the neocortex. Atrophy tends to follow the spreading pattern of $\tau$P, more so than that of Aβ."


# ╔═╡ 703b4044-ab3f-4e6f-a567-ba41942abe72
html"""<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/braak-stages.png?raw=true" height=300 width=900 >"""


# ╔═╡ c484008a-ec30-4d73-bc6e-f462a5d187b1
md" 
## Transport and the Graph Laplacian

An important part of the modelling of $\tau$P in AD is describing transport. In this work, we do this using the graph Laplacian, a discrete counterpart to the Laplace operator used to describe diffusion in a continuous setting. The graph Laplacian is derived from a graph of brain connections, generated using tractography. "

# ╔═╡ a1e74c91-7f6a-4ca9-b497-151e2d5875c3
html"""
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/connectomes/connectome-pit.png?raw=true" height=300 width=900>"""

# ╔═╡ 5a8dc4c9-246c-4dd1-ba62-638f0879d7b7
md" 
## Transport and the Graph Laplacian 
 
Using tractography, we can define an adjacency matrix,

```math
\begin{equation}
\mathbf{A}_{ij} = \left\{\begin{array}{cl} n/l^{2} & \text{ an edge connects } v_i \text{ to } v_j\\ 0 
                            & \text{otherwise}\end{array}\right.,
\end{equation}
```
which, in turn, can be used to define the degree matrix,

```math
\begin{equation}
\mathbf{D}_{ij} = \delta_{ij} \sum_{j=1}^{N} A_{ij}.
\end{equation}
```

Finally, we can derive the graph Laplacian: 

$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

The graph Laplacian, $\mathbf{L}$, has similar properties to the continous Laplace operator and is therefore a suitable substitution to model diffusion on a network."


# ╔═╡ 9eaf7995-f369-4a8b-970f-9ba73127052b
begin
	function NetworkDiffusion(du, u, p, t)
        du .= -p * L * u
	end
	function simulate(prob, p)
		solve(remake(prob, p=p), Tsit5())
	end;

	
	L = deserialize("assets/graphs/L.jls");
	
	u1 = zeros(83)
	u1[[27, 68]] .= 0.1
		
	p1 = 1.0
	t_span1 = (0.0,20.0)
		
	problem1 = ODEProblem(NetworkDiffusion, u1, t_span1, p1)
		
	sol1 = solve(problem1, Tsit5())
end;

# ╔═╡ 22f603d3-86ba-41f2-a4cc-8d4cb97f1511
md" 
# Transport and the Graph Laplacian
"

# ╔═╡ 1f797438-6e29-42c1-abf7-86072cc5de1a
two_cols(md"", 
md"
ρ = $(@bind ρ1 Slider(0:0.1:5, show_value=true, default=0))
")

# ╔═╡ 0b40960b-4114-4357-a99d-b8389fe46835
TwoColumn( 		
md"
Using the graph Laplacian, we can define the network heat equation, which describes diffusion across a graph. To the left, I have shown a simulation with an initial seeding concentration placed in the entorhinal cortex. By changing the diffusion coefficient, $\rho$, we can see how the dynamics are affected.
\
\
\
$$\frac{d \mathbf{p}}{dt} = -\rho \mathbf{L} \mathbf{p}$$
", 	
Plots.plot(simulate(problem1, ρ1), size=(450,300), labels=false, ylims=(0.0,0.5), xlims=(0.0,20.0), ylabel="Concentration"))



# ╔═╡ 53811fc6-c78e-4439-8f8c-0a002d47371a
md" 
## The Heterodimer Model
"

# ╔═╡ 2967e74c-0f2b-4d7d-bc29-9c54c71cc242
html"""
Recall the autocatalytic process
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/heterodimerkinetics.png?raw=true" height=150 width=300 vspace=20, hspace=275>"""

# ╔═╡ 8907ecbb-2127-40e3-a012-acd52dfb2508
md"
We can describe this process with the following reaction-diffusion equations, where the rates $$k_{ij}$$ correspond to the rates above.
>```math
>\begin{align}
>\frac{dp_i}{dt} &=  -\kappa\sum\limits_{j=1}^{N}L_{ij}p_j +  k_0 &- k_1 p_i - k_{12}p_i \tilde{p}_i \\
>\frac{d\tilde{p}_i}{dt} &= -\tilde{\kappa}\sum\limits_{j=1}^{N}L_{ij}\tilde{p}_j &- \tilde{k}_1 \tilde{p}_i + k_{12}p_i\tilde{p}_i.
>\end{align}
>```"

# ╔═╡ 16a92048-e476-4b68-a445-657025a28fcd
md" ## The FKPP Model"

# ╔═╡ 432b0862-c18c-444b-8b4c-385c0c3d0405
html"""
The protein dynamics (here shown for a single toxic species) can be simplified in the following way:
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/fkppkinetics.png?raw=true
" height=150 width=300 vspace=10 hspace=290>"""

# ╔═╡ ade96765-12dd-4fe8-a4e7-be975304441d
md"""
With an appropriate diffusion term, we arrive at:
> $$\frac{d p_i}{dt} = -\rho\sum\limits_{j=1}^{N}L_{ij}p_j + \alpha p_i\left(1-p_i\right)$$

"""

# ╔═╡ ee429405-e34d-48b3-b93c-376d5eedd7ab
begin
	function NetworkFKPP(du, u, p, t)
        du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
	end
	
	u0 = zeros(83)
	u0[[27, 68]] .= 0.1
		
	p = [1.0, 1.0]
	t_span = (0.0,20.0)
		
	problem = ODEProblem(NetworkFKPP, u0, t_span, p)
		
	sol = solve(problem, Tsit5())
end;

# ╔═╡ 66d8ea22-9bf5-4362-90d5-71471e519bd5
md"## FKPP Dynamics"

# ╔═╡ 938625f5-ca78-44a8-b6cd-89988fccaf34
two_cols(md"", 
md"
ρ = $(@bind ρ Slider(0:0.1:3, show_value=true, default=0))
α = $(@bind α Slider(-3:0.1:3, show_value=true, default=0))
")

# ╔═╡ 703e9020-94de-493d-a15c-f11a684f0c95
TwoColumn( 		
md"
The simulation to the right shows the dynamics of the FKPP model with seeding in the entorhinal cortex.
We can vary the parameters for diffusion, $\rho$, and growth, $\alpha$, to see how the dynamics change and where critical points lie.
\
\
\
$$\frac{d p_i}{dt} = -\rho\sum\limits_{j=1}^{N}L_{ij}p_j + \alpha p_i\left(1-p_i\right)$$
", 	
Plots.plot(simulate(problem, [ρ, α]), size=(450,300), labels=false, ylims=(0.0,1.0), xlims=(0.0,20.0), ylabel="concentration"))


# ╔═╡ b01363ae-65cb-4c9b-be74-05647e7196f5
md"# Inference!
Now that we have some models, how do we fit them to data?
More importantly, how do we fit them to data **and** account for sources of uncertainty?"

# ╔═╡ 5bb63831-fec0-4fdc-a5a1-bf285e91168d
md"## Inverse Problems using Bayesian Inference
\
For observations $\mathbf{x} = x_{1:n}$ and latent variables  $\mathbf{\theta} = \theta_{1:m}$, we have a joint density
$$p(\mathbf{x}, \mathbf{\theta})$$.

To evalulate a particular hypothesis, we need to evaluate the posterior $p(\mathbf{\theta} \mid \mathbf{x})$. To do so, we first decompose the joint distribution:

$$p(\mathbf{x}, \mathbf{\theta}) = p(\mathbf{x} \mid \mathbf{\theta})p(\mathbf{\theta}) = p(\mathbf{\theta} \mid \mathbf{x})p(\mathbf{x})$$

Dividing through by the *evidence*, we obtain Bayes' rule: 

>$$p(\mathbf{\theta} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \mathbf{\theta})p(\mathbf{\theta})}{p(\mathbf{x})}$$
"

# ╔═╡ 677e9901-dd1e-4e2f-9bfe-666935342e73
md"## Reverand, What Does it Mean?

>$$\underbrace{p(\mathbf{\theta} \mid \mathbf{x})}_{posterior} = \frac{\overbrace{p(\mathbf{x} \mid \mathbf{\theta})}^{likelihood}\overbrace{p(\mathbf{\theta})}^{prior}}{\underbrace{p(\mathbf{x})}_{evidence}}$$
- Likelihood: Probability that a particular set of parameter values, $\theta$, generate the observations, $\mathbf{x}$.
- Prior: Probability representing our initial beliefs about the parameter values.
- Evidence: Normalising factor; probability of observing our data. Otherwise known as the marginal likelihood.
- Posterior: Probability of some parameter values $\theta$ given that we have made observations $\mathbf{x}$. 
"

# ╔═╡ cb865d40-3a15-469f-9a63-2c04a8840ca1
md"## Why is Bayesian Inference Hard? Because Integration is hard! 
It's almost always the case that we cannot do Bayesian inference analytically. The principal reason for this comes from the evidence term: 

$${p(\mathbf{x})} = \int p(\mathbf{x} , \mathbf{\theta}) d\mathbf{\theta}$$

As $\theta$ becomes larger, the complexity of integration grows exponentially with the dimensionality. Hence, naive numerical integration (such as quadrature) becomes computationally infeasible. "

# ╔═╡ 43daff92-8265-429e-af25-9b796b599825
md"# Hamiltonian Monte Carlo 
MCMC methods allow one to sample from the posterior, as opposed to approximating it. Tradiational algortihms like Metroplis-Hastings are good for low dimensional problems, but don't sacle well. HMC allows one to take efficient steps during sampling, even in high dimensional space
"

# ╔═╡ 6359cc18-634c-483c-8739-91ee80cf2dfe
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
The main advantage of Hamiltonian Monte Carlo algorithms is that it generates samples that are very likely to be accepted by taking advantage of the geometry of posterior space.
"

# ╔═╡ b3152568-f1d1-4990-959b-5ab135e29d57
md"## Hamiltonian Monte Carlo Algorithm 
We can modify the algorithm to be the following: 
Expand your paramter space two fold by introducing an auxilliary momentum parameters, $p$, and forming a joint density with $\theta$, $-\log\pi(\theta, p)$, and Hamiltonian, $\mathcal{H}(\theta, p)$. 
Adding this structure allows us to make more informed proposal steps. 
```julia
for i in 1:Number of iterations 
	pₚ ~ Normal(0, M)		#Sample a new p from proposal distribution with std M (the mass matrix)
	θₚ <- H(θ, pₚ) 			#Generate a new θ by solving the Hamiltonian over time LΔ.
	r = α((θ,p), (θₚ, pₚ)) 	#Calculate the acceptance probability given a likelihood ratio
	if r ≥ 1 				#if the likelihood is increased, accept the new parameters 
		(θ, p) = (θₚ, pₚ)
	else 					#Else, accept relative to some probaility drawn from a uniform distribution  
		x = U(0,1)
		if r > x
			θ = θₚ
		else
			reject θₚ
	
```
"

# ╔═╡ f07caceb-559e-4e76-b52c-890290efa64e
md"# Show me the Results! 
'Pavan, have you actually done any *real* work?'"

# ╔═╡ 9066ba33-ddc3-4497-b896-393458faad92
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

# ╔═╡ 7b020a5c-5657-41d3-bb34-cfc1df999494
md"## Inference Methods for Patient Data
Ultimately, for our dynamical systems $f(u_0, t; \theta, \mathbf{L})$, we want to infer likely values of $\theta$. The dynamical systems proposed earlier in the talk describe the flow of proteins on a network. Thus, we need data that is apprpriate for the model. 
Fortunately, such data is available from the Alzheimer's Disease Neuroimaging Initiative (ADNI) in the form of Aβ and τP PET imaging."

# ╔═╡ 63136cdf-8481-41b2-98c7-04c3d4e46778
two_cols(md"",md"""
subject = $(@bind subject Slider(1:78, show_value=true, default=1))
""")

# ╔═╡ 358c91c1-cbd5-4802-a9d7-72a390a6b91b
begin 
	p3 = histogram(tₛ[:,1] .+ min_age, bins=10, xlabel="Age at first scan", labels=false, alpha=0.7, size=(300,300));

	p4 = scatter(age[subject,t_index[subject,:]], suvr[:,t_index[subject,:],subject]', label=false, xlims=(min_age-5, max_age+5), ylims=(0.0,0.5), color=:grey, size=(550,290));
	scatter!(age[subject,t_index[subject,:]], suvr[[27,68],t_index[subject,:],subject]', label=false, xlims=(min_age-5, max_age+5), ylims=(0.0,0.5), color=:red)
	ylabel!("SUVR");
	xlabel!("Age");
end;

# ╔═╡ 6749c00c-38ef-4f5b-a446-f8e43d8dddb2
TwoColumn(p3,p4)

# ╔═╡ 96a18032-5d54-4d06-a00f-acdd4d4b25ed
md"## Defining a Probabalistic Model
Now that we have data, we should want to fit a model to it using Bayesian inference. We assume the following model structure: 

$$\mathbf{x} = f(\mathbf{u_0}, t; \theta, \mathbf{L}) + \mathcal{N}(0, \sigma)$$
We're going to fit the Network FKPP model, the priors for which will be: 

```math
\begin{align} 
\sigma &\sim \Gamma^{-1}(2, 3) \\
\rho &\sim \mathcal{N}^{+}(0,10) \\
\alpha &\sim \mathcal{N}(0,10) \\
\end{align}
```
"

# ╔═╡ 95b23324-b326-4ef7-9031-9afb3cc92d7b
begin
	chaindir = datadir * "chains/"
	chain_vector = Vector{Any}(undef, 78)	
	for i in 1:78
		chain_vector[i] = deserialize(chaindir * "nuts/" * "posteriorchain_$(i).jls")
	end
end;

# ╔═╡ 1488dec0-8e37-4e0c-84bb-3d4abcdebfd0
md"## Results
We sample using NUTS, a form of Hamiltonian Monte Carlo. The distributions for the diffusion coefficient, $\rho$, indicate a *very* slow rate of diffusion, on the order of mm/year, across all subjects. There is more variation in the individual posteriors for growth rate $\alpha$, suggesting that it may be the more important driver of disease pathology."

# ╔═╡ 04464d64-fb37-4106-8322-0cbbec21f07a
begin
	fig = plot(chain_vector[1][:k], seriestype = :density, labels=false, color=:white, size=(400,300))
	for i in 1:78
		plot!(fig, chain_vector[i][:k], seriestype = :density, labels=false, color=:blue, alpha=0.2, linewidth=1)
	end
	title!("Diffusion coefficient")
	
	fig_a = plot(chain_vector[1][:a], seriestype = :density, labels=false, color=:white, size=(400,300))
	for i in 1:78
		plot!(fig_a, chain_vector[i][:a], seriestype = :density, labels=false, color=:red, alpha=0.3, linewidth=1)
	end
	title!("Growth Rate")
	fig_a
	two_cols(fig, fig_a)
end

# ╔═╡ 1c86e2f4-bed6-4327-b977-6d732c57dd81
md"## Projecting Uncertainty Forward
A key advantage of Bayesian modelling is the quantification of uncertainty. Using the estimates of parameter distributions, we can project forward in time to simulate potential disease outcomes."


# ╔═╡ 5acc04c0-ba56-4d07-b2b0-323596c8a420
md"""
##### sub1 = $(@bind sub1 Slider(1:78, show_value=true, default=1))
"""

# ╔═╡ b8e8837e-345d-4539-aeb4-5d5575e08cdc
begin
	A = deserialize("assets/graphs/A.jls")
	a_max = maximum(A)
	postchain = chain_vector[sub1]
	ks = Array(postchain[:k]) .* a_max
	as = Array(postchain[:a])
	σ = Array(postchain[:σ])

	probpost = ODEProblem(NetworkFKPP, suvr[:,1,sub1], (0.0,20.0), [mean(ks),mean(as)])
	resolpost = solve(probpost, Tsit5(), abstol=1e-9, reltol=1e-6)
	
	p8 = plot(xlims=(0,20), ylims=(0,1), size=(500,350))	
	reltime1 = tₛ[sub1,t_index[sub1,:]] .- minimum(tₛ[sub1,t_index[sub1,:]])
	scatter!(reltime1, suvr[:,t_index[sub1,:],sub1]', label=false, color=:grey)
	scatter!(reltime1, suvr[[81],t_index[sub1,:],sub1]', label=false, color=:red)
	for i in 1:100
		n = rand(1:1000)
		prob = remake(probpost, p=[ks[n], as[n]])
		resol = solve(prob, Tsit5(), abstol=1e-9, reltol=1e-6) 
		soln = clamp.(resol .+ rand(Normal(0.0, σ[n]), size(resol)), 0.0,1.0)
		plot!(p8, resol.t, soln[81,:], alpha=0.5, color = "#BBBBBB", legend = false)
#		plot!(p8, resol.t, resol[68,:], alpha=0.5, color = "#BBBBBB", legend = false)
	end
	plot!(p8, resolpost.t, resolpost[81,:], linewidth=3, alpha=0.5, color = :red, legend = false)
#	plot!(p8, resolpost.t, resolpost[68,:], linewidth=3, alpha=0.5, color = :red, legend = false)
	xlabel!("Time")
	ylabel!("Protein Concentration")
#	scatter!(reltime1, suvr[node,:,sub1], legend = false)
end;

# ╔═╡ e48ce2b1-7c15-481c-97ad-5cb7d8ea4b54
begin
	p7 = plot(chain_vector[sub1][:k], seriestype= :density, xlims=(-0.5,0.5), ylims=(0,100), labels="NUTS: Diffusion", linewidth=3, color=:navy, size=(350,360))
		plot!(chain_vector[sub1][:a], seriestype= :density, xlims=(-0.5,0.5), ylims=(0,100), label="NUTS: Growth", linewidth=3, color=:darkred)
	xlabel!("Parameter value")
end;

# ╔═╡ 78f4378e-6ddb-416f-9d66-4eae762ce8b0
TwoColumn(p8, p7)

# ╔═╡ b515a700-2ca8-4efc-b38f-32e63803b53d
md"
## A More Complex Model

The probabilistic model we considered first is a simplification of the true generative process. Significantly, we ignore uncertainty about the initial conditions.

In the simple model, the initial conditions are assumed to be fixed at the value from the initial scan. However, since our model depends on the initial conditions, it should contain parametric uncertainty and therefore, we should include priors for the initial conditions.

```math
\begin{align} 
\sigma &\sim \Gamma^{-1}(2, 3) \\
\rho &\sim \mathcal{N}^{+}(0,10) \\
\alpha &\sim \mathcal{N}(0,10) \\
\mathbf{u_0} &\sim \mathcal{N}(\mathbf{x_0}, 5)
\end{align}
```

Including the initial conditions increases the uncertainty of trajectories projected forward.
"

# ╔═╡ 8e7fb27c-80ab-421d-b631-b739dac80efd
md"## A More Complex Model"

# ╔═╡ b8714b83-8615-4533-92c8-baa489c25ad0
html"""
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/results/subject36-81-fullposterior.png?raw=true" height=250 width=800>
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/results/subject36-u0.png?raw=true" height=250 width=800>
"""

# ╔═╡ 6a6843fe-72b3-4a05-9176-ba9c434e676d
md" 
## Utilising Multimodal Data

We could also use multimodal data, to take complete advantage of the data available through open source repositories. Using structural MRI data, we can inference atrophy dynamics, using a simple logistic model fo atrophy that depends on protein concentration. 
An atrophy model is given by: 

$$\frac{dq_i}{dt} = G_c p_i( 1 - q_i ),$$

which is coupled to a FKPP model of protein dynamics. We have the following priors:

```math
\begin{align}
    \sigma_t &\sim \Gamma^{-1}(2, 3) \\
    \sigma_a &\sim \Gamma^{-1}(2, 3) \\
    \rho &\sim \mathcal{N}^{+}(0, 10) \\
    \alpha &\sim \mathcal{N}(0, 10)  \\
    G_c &\sim \mathcal{N}(0, 10)
\end{align}
```
"

# ╔═╡ 15409a15-c7fb-40bd-916c-a0d28123b030
md" 
## Utilising Multimodal Data"

# ╔═╡ 9edae25e-1a41-4b9f-a01a-1f0afee68f0d
html"""
<img src=https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/atrophy/post_pred_mod1_4subs.png?raw=true" height=400 width=500 hspace=200>"""

# ╔═╡ 65748ba6-52bf-442f-bb1e-ecaabbd7c909
cite("Schäfer, A., Chaggar, P., Thompson T.B., Goriely, A., Kuhl, E,. Alzheimer’s Disease Neuroimaging Initiative, 2021. Predicting brain atrophy from tau pathology: A summary of clinical findings and their translation into personalized models. Submitted to Brain Multiphysics.")

# ╔═╡ e8f4c7a4-970b-40bf-b9aa-b1bad49888c2
md"
# Future Work
Going forward, I will focus on three major themes: 
* Applications of modelling and inference to AD
* Developing a software pipeline for Bayesian inference 
* Quantifying uncertainty associated with network topology
"

# ╔═╡ 2c2a58ff-e916-495f-bc96-a0530392a273
md"
## Applications of modelling and inference to AD 

I'm currently working on two collaborations: 
1. **Collaboration with Ashish Raj, UCSF**
Working on using Bayesian inference with sparsity priors to infer seeding locations. 
2. **Collaboration with Ellen Kuhl, Stanford**
Following our recent work together, we will continue to investigate the use of multimodal data. In particular, whether sMRI data alone can be used to infer protein dynamics as well as atrophy dynamics. Secondly, eschewing global parameters and determining whether regional parameters are identifiable. 
"

# ╔═╡ 48073ebb-3ed7-4f54-a6a2-b6201eb08f89
md"
## Software Pipeline

For ongoing work, it will be important to have robust tools capable of reproducing simulation and inference results. Software should aim to make the following tasks easier: 
* Solving differential equations on brain networks
* Interfacing with tools such as FreeSurfer and SPM
* Creating and running inference on probabilistic models

## Investigating the Graph Laplacian

In the thesis, I have not addressed the significance of uncertainty around the graph Laplacian. 

Previous work has shown that network topology significantly affects the dynamics given by models of network pathology. 

There is currently no concensus about whether some tractography procedures are more suitable than others for macroscale brain dynamical modelling.

This introduces huge uncertainty into modelling process. Future work should assess: 1) whether estimated dynamical regimes are conserved across different connectomes. 2) quantifying uncertainty of the graph Laplacian by combining variational inference with random matrix theory.
"

# ╔═╡ c973adef-46eb-4ec6-a23d-1e9b0aad220a
md"
# Thank you!

To my academic supervisors, Alain Goriely and Saad Jbabdi. And to my industrial supervisors, Stefano Magon and Gregory Klein at Roche.

With the help of my supervisors and collaborators, the work presented here has been included in three publications over the past year:

* Thompson, T.B., **Chaggar, P**., Kuhl, E., Goriely, A. and Alzheimer’s Disease Neuroimaging Initiative, 2020. Protein-protein interactions in neurodegenerative diseases: a conspiracy theory. PLoS computational biology, 16(10), p.e1008267.
* Putra, P., Thompson, T.B., **Chaggar, P**., Goriely, A., Alzheimer’s Disease Neuroimaging Initiative and Alzheimer’s Disease Neuroimaging Initiative, 2021. Braiding Braak and Braak: Staging patterns and model selection in network neurodegeneration. Network Neuroscience, pp.1-41.
* Schäfer, A., **Chaggar, P**., Thompson T.B., Goriely, A., Kuhl, E,. Alzheimer’s Disease Neuroimaging Initiative, 2021. Predicting brain atrophy from tau pathology: A summary of clinical findings and their translation into personalized models. Submitted to Brain Multiphysics. **(Joint First Author)**"

# ╔═╡ Cell order:
# ╠═4cf73862-25c7-11ec-174b-6dcf37428913
# ╠═5a0591b7-1a34-4e3e-9862-c772fc3159f4
# ╠═18225564-8512-4fca-87c8-a95ec2fa0d05
# ╠═8d93f866-2a10-4489-a1d4-1ac1da97f248
# ╠═c1e20410-aed5-48a4-8f02-d78f957c15f0
# ╟─4ff67e50-ccdd-479f-8280-e04ab2354ce4
# ╟─abc58f7f-c4c1-47b6-861a-ab679d34bc95
# ╟─95d6223a-c12e-4b26-8c4e-d59a59c7d129
# ╟─d75eb4e7-2fbf-44ca-af86-bf67fc1d393d
# ╟─a0cb7614-2ab3-44d1-9202-02f19915edf6
# ╟─f45c2cd6-baf6-4ce3-84b0-5bf8fb9e67d4
# ╟─703b4044-ab3f-4e6f-a567-ba41942abe72
# ╟─c484008a-ec30-4d73-bc6e-f462a5d187b1
# ╟─a1e74c91-7f6a-4ca9-b497-151e2d5875c3
# ╟─5a8dc4c9-246c-4dd1-ba62-638f0879d7b7
# ╟─9eaf7995-f369-4a8b-970f-9ba73127052b
# ╟─22f603d3-86ba-41f2-a4cc-8d4cb97f1511
# ╟─0b40960b-4114-4357-a99d-b8389fe46835
# ╟─1f797438-6e29-42c1-abf7-86072cc5de1a
# ╟─53811fc6-c78e-4439-8f8c-0a002d47371a
# ╟─2967e74c-0f2b-4d7d-bc29-9c54c71cc242
# ╟─8907ecbb-2127-40e3-a012-acd52dfb2508
# ╟─16a92048-e476-4b68-a445-657025a28fcd
# ╟─432b0862-c18c-444b-8b4c-385c0c3d0405
# ╟─ade96765-12dd-4fe8-a4e7-be975304441d
# ╟─ee429405-e34d-48b3-b93c-376d5eedd7ab
# ╟─66d8ea22-9bf5-4362-90d5-71471e519bd5
# ╟─703e9020-94de-493d-a15c-f11a684f0c95
# ╟─938625f5-ca78-44a8-b6cd-89988fccaf34
# ╟─b01363ae-65cb-4c9b-be74-05647e7196f5
# ╟─5bb63831-fec0-4fdc-a5a1-bf285e91168d
# ╟─677e9901-dd1e-4e2f-9bfe-666935342e73
# ╟─cb865d40-3a15-469f-9a63-2c04a8840ca1
# ╟─43daff92-8265-429e-af25-9b796b599825
# ╟─6359cc18-634c-483c-8739-91ee80cf2dfe
# ╟─b3152568-f1d1-4990-959b-5ab135e29d57
# ╟─f07caceb-559e-4e76-b52c-890290efa64e
# ╟─9066ba33-ddc3-4497-b896-393458faad92
# ╟─358c91c1-cbd5-4802-a9d7-72a390a6b91b
# ╟─7b020a5c-5657-41d3-bb34-cfc1df999494
# ╟─6749c00c-38ef-4f5b-a446-f8e43d8dddb2
# ╟─63136cdf-8481-41b2-98c7-04c3d4e46778
# ╟─96a18032-5d54-4d06-a00f-acdd4d4b25ed
# ╟─95b23324-b326-4ef7-9031-9afb3cc92d7b
# ╟─1488dec0-8e37-4e0c-84bb-3d4abcdebfd0
# ╟─04464d64-fb37-4106-8322-0cbbec21f07a
# ╟─b8e8837e-345d-4539-aeb4-5d5575e08cdc
# ╟─e48ce2b1-7c15-481c-97ad-5cb7d8ea4b54
# ╟─1c86e2f4-bed6-4327-b977-6d732c57dd81
# ╟─78f4378e-6ddb-416f-9d66-4eae762ce8b0
# ╟─5acc04c0-ba56-4d07-b2b0-323596c8a420
# ╟─b515a700-2ca8-4efc-b38f-32e63803b53d
# ╟─8e7fb27c-80ab-421d-b631-b739dac80efd
# ╠═b8714b83-8615-4533-92c8-baa489c25ad0
# ╟─6a6843fe-72b3-4a05-9176-ba9c434e676d
# ╟─15409a15-c7fb-40bd-916c-a0d28123b030
# ╟─9edae25e-1a41-4b9f-a01a-1f0afee68f0d
# ╟─65748ba6-52bf-442f-bb1e-ecaabbd7c909
# ╟─e8f4c7a4-970b-40bf-b9aa-b1bad49888c2
# ╟─2c2a58ff-e916-495f-bc96-a0530392a273
# ╟─48073ebb-3ed7-4f54-a6a2-b6201eb08f89
# ╟─c973adef-46eb-4ec6-a23d-1e9b0aad220a
