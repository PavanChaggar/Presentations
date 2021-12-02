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

# ╔═╡ 4cf73862-25c7-11ec-174b-6dcf37428913
begin
	using Pkg
	Pkg.activate("../../TransferPresentation/")
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
include("../../TransferPresentation/functions.jl")

# ╔═╡ 5a0591b7-1a34-4e3e-9862-c772fc3159f4
html"""<style>
main {
max-width: 900px;
}"""

# ╔═╡ 18225564-8512-4fca-87c8-a95ec2fa0d05
html"<button onclick='present()'>present</button>"

# ╔═╡ 4ff67e50-ccdd-479f-8280-e04ab2354ce4
md" 
# Understanding Alzheimer's Disease using Mathematics
**Pavanjit Chaggar, October 2021**

Supervised by Alain Goriely, Saad Jbabdi (Oxford), Stefano Magon and Gregory Klein (Roche)"

# ╔═╡ abc58f7f-c4c1-47b6-861a-ab679d34bc95
md" 
# Overview and Introduction

I'm a second SABS DPhil student based at the Mathematical Institute. 

I'm going ot talk about the following:
- Alzheimer's disease (AD)
- Mathematical models of AD
- Inference Workflow
- Some results
"

# ╔═╡ 95d6223a-c12e-4b26-8c4e-d59a59c7d129
md" 
# Alzheimer's Disease -- A Brief Summary
Alzheimer's is characterised by gradual neurodegeneration associated with the pervasive spreading of toxic proteins. 

In particular, two proteins, Amyloid beta (Aβ) and tau-protein (τP), are believed to underlie and drive the development of pathology. 

Historically, Aβ was thought to be the primary cause of AD, with research focussing solely on its pathology. 

More recent work has focussed on role of τP, in part because it spreads very predictably and is more tightly coupled with atrophy and symotom onset.
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

# ╔═╡ 1678deeb-ea59-408e-b574-5f28dc7214a0
cite("Fornari, S., Schäfer, A., Jucker, M., Goriely, A. and Kuhl, E., 2019. Prion-like spreading of Alzheimer’s disease within the brain’s connectome. Journal of the Royal Society Interface, 16(159), p.20190356.")

# ╔═╡ f45c2cd6-baf6-4ce3-84b0-5bf8fb9e67d4
md"## Braak Stages of Tau protein
In most AD cases, τP follows a predictable pattern of spreading, starting in the entorhinal cortex before spreading through the hippocampal regions, lateral cortex and finally into the neocortex. Atrophy tends to follow the spreading pattern of $\tau$P, more so than that of Aβ"


# ╔═╡ 703b4044-ab3f-4e6f-a567-ba41942abe72
html"""<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/braak-stages.png?raw=true" height=300 width=900 >"""


# ╔═╡ 53811fc6-c78e-4439-8f8c-0a002d47371a
md" 
## The Heterodimer Model
"

# ╔═╡ 2967e74c-0f2b-4d7d-bc29-9c54c71cc242
html"""
Recall the autocatalytic process
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/heterodimerkinetics.png?raw=true" height=125 width=300 vspace=50, hspace=275>"""

# ╔═╡ 8907ecbb-2127-40e3-a012-acd52dfb2508
md"
We can describe this process with the following reaction-diffusion equations, where the rates $$k_{ij}$$ correspond to the rates above.
>```math
>\begin{align}
>\frac{∂\mathbf{p}}{∂ t} &=  ∇⋅\left(\mathbf{D}∇ \mathbf{p} \right) +  k_0 &- k_1 \mathbf{p} - k_{12}\mathbf{p} \mathbf{\hat{p}}  \\
>\frac{∂\mathbf{\hat{p}}}{∂ t} &= ∇⋅\left(\mathbf{D}∇ \mathbf{\hat{p}} \right) &- \hat{k}_1 \mathbf{\hat{p}} + k_{12}\mathbf{p}\mathbf{\hat{p}}
>\end{align}
>```"

# ╔═╡ 7b75824e-5312-474c-8704-28f399a1ad88
cite("Fornari, S., Schäfer, A., Jucker, M., Goriely, A. and Kuhl, E., 2019. Prion-like spreading of Alzheimer’s disease within the brain’s connectome. Journal of the Royal Society Interface, 16(159), p.20190356.")

# ╔═╡ e8857f50-fdd0-49cc-950a-5490359b752f
md" ## The FKPP Model"

# ╔═╡ acefe8e5-f2a1-43a5-b585-1621a8a5fb71
html"""
The protein dynamics (here shown for a single toxic species) can be simplified in the following way: 
<img src="https://github.com/PavanChaggar/inference-methods-UCSF0721/blob/main/assets/images/fkppkinetics.png?raw=true
" height=150 width=300 vspace=10 hspace=290>"""

# ╔═╡ 3c078fae-fa21-4256-b172-ba4b4734cdc8
md"""
With an appropriate diffusion term, we arrive at:
> $$\frac{∂\mathbf{c}}{∂ t} =  ∇⋅\left(\mathbf{D}∇ \mathbf{c} \right) + α\mathbf{c}(\mathbf{1} - \mathbf{c})$$
"""

# ╔═╡ 287e23e4-f01c-44f8-a0d4-7fbc0467c16f
cite("Fornari, S., Schäfer, A., Jucker, M., Goriely, A. and Kuhl, E., 2019. Prion-like spreading of Alzheimer’s disease within the brain’s connectome. Journal of the Royal Society Interface, 16(159), p.20190356.")

# ╔═╡ 113f696c-129a-4862-a3cb-a814fde5e8b2
md"
## Avoiding the Continuum
We typically want to avoid the heavy computations necessary to solve PDEs on a sufficiently fine mesh. There is evidence to support the modelling assumption that most of the transport of $\tau$P occurs through neuronal pathways. Using tractography, we can create a graph of the brain that represents neuronal connections between some set of regions.
"

# ╔═╡ a1e74c91-7f6a-4ca9-b497-151e2d5875c3
html"""
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/connectomes/connectome-pit.png?raw=true" height=300 width=900>"""

# ╔═╡ 7db1dc72-3b20-49a0-89e0-b53d8576a447
md"## Avoiding the Continuum
With such an assumption, we can transform our burdensome PDE, 
> $$\frac{∂\mathbf{p}}{∂ t} =  ∇⋅\left(\mathbf{D}∇ \mathbf{p} \right) + α\mathbf{p}(1 - \mathbf{p}),$$ 
into a nice, friendly ODE:
> $$\frac{d p_i}{dt} = -\rho\sum\limits_{j=1}^{N}L_{ij}p_j + \alpha p_i\left(1-p_i\right)$$
Where $\mathbf{L}$ is the graph Laplacian.
" 

# ╔═╡ ee429405-e34d-48b3-b93c-376d5eedd7ab
begin
	function NetworkFKPP(du, u, p, t)
        du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
	end
	function simulate(prob, p)
		solve(remake(prob, p=p), Tsit5())
	end;

	
	L = deserialize("../../TransferPresentation/assets/graphs/L.jls");
	
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
We can vary the parameters for diffusion, $\rho$, and growth, $\alpha$, to see how the dynamics change. 
\
\
\
$$\frac{d p_i}{dt} = -\rho\sum\limits_{j=1}^{N}L_{ij}p_j + \alpha p_i\left(1-p_i\right)$$", 	
Plots.plot(simulate(problem, [ρ, α]), size=(450,300), labels=false, ylims=(0.0,1.0), xlims=(0.0,20.0)))


# ╔═╡ b01363ae-65cb-4c9b-be74-05647e7196f5
md"# Inference!
Now that we have some models, how do we fit them to data?
More importantly, how do we fit them to data **and** account for sources of uncertainty?"

# ╔═╡ 5bb63831-fec0-4fdc-a5a1-bf285e91168d
md"## Inverse Problems using Bayesian Inference
\
For observations $\mathbf{x} = x_{1:n}$ and latent variables  $\mathbf{\theta} = \theta_{1:m}$, we have a joint density
$$p(\mathbf{x}, \mathbf{\theta})$$
To evalulate a particular hypothesis, we need to evaluate the posterior $p(\mathbf{\theta} \mid \mathbf{x})$. To do so, we first decompose the joint distribution:

$$p(\mathbf{x}, \mathbf{\theta}) = p(\mathbf{x} \mid \mathbf{\theta})p(\mathbf{\theta}) = p(\mathbf{\theta} \mid \mathbf{x})p(\mathbf{x})$$

Dividing through by the *evidence*, we obtain Bayes' rule: 


>$$\underbrace{p(\mathbf{\theta} \mid \mathbf{x})}_{posterior} = \frac{\overbrace{p(\mathbf{x} \mid \mathbf{\theta})}^{likelihood}\overbrace{p(\mathbf{\theta})}^{prior}}{\underbrace{p(\mathbf{x})}_{evidence}}$$

"

# ╔═╡ cb865d40-3a15-469f-9a63-2c04a8840ca1
md"## Why is Bayesian Inference Hard? Because Integration is hard! 
It's almost always the case that we cannot do Bayesian inference analytically. The principal reason for this comes from the evidence term: 

$${p(\mathbf{x})} = \int p(\mathbf{x} , \mathbf{\theta}) d\mathbf{\theta}$$

As the dimension of $\theta$ becomes larger, the complexity of integration grows exponentially with the dimensionality. Hence, naive numerical integration (such as quadrature) becomes computationally infeasible. "

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
For our dynamical systems $f(u_0, θ, t)$, we want to infer the values of θ. The dynamical systems proposed earlier in the talk describe the flow of proteins on a network. Thus, we need data that is apprpriate for the model. 
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

$$\mathbf{x} = f(\mathbf{u_0}, θ) + \mathcal{N}(0, \sigma)$$
We're going to fit the Network FKPP model, the priors for which will be: 

```math
\begin{align} 
\sigma &\sim \Gamma^{-1}(2, 3) \\
\rho &\sim \mathcal{N}^{+}(0,5) \\
\alpha &\sim \mathcal{N}(0,5) \\
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
md"## Result
We sample using NUTS, a form of Hamiltonian Monte Carlo. The distributions for the diffusion coefficient, $\rho$, indicate a *very* slow rate of diffusion, on the order of mm/year, across all subjects. There is more variation in the individual posteriors for growth rate $\alpha$, suggesting that it may be the more important driver of disease pathology."

# ╔═╡ 1c86e2f4-bed6-4327-b977-6d732c57dd81
md"## Projecting Uncertainty Forward
A key advantage of Bayesian modelling is the quantification of uncertainty. Using the estimates of parameter distributions, we can project forward in time to simulate potential disease outcomes."


# ╔═╡ 5acc04c0-ba56-4d07-b2b0-323596c8a420
md"""
##### sub1 = $(@bind sub1 Slider(1:78, show_value=true, default=1))
"""

# ╔═╡ b8e8837e-345d-4539-aeb4-5d5575e08cdc
begin
	A = deserialize("../../TransferPresentation/assets/graphs/A.jls")
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

# ╔═╡ 04464d64-fb37-4106-8322-0cbbec21f07a
begin
	fig = plot(chain_vector[1][:k], seriestype = :density, labels=false, color=:white, size=(400,300))
	for i in 1:78
		plot!(fig, chain_vector[i][:k] * a_max, seriestype = :density, labels=false, color=:blue, alpha=0.2, linewidth=1, xlabel="mm/year")
	end
	title!("Diffusion coefficient")
	
	fig_a = plot(chain_vector[1][:a], seriestype = :density, labels=false, color=:white, size=(400,300))
	for i in 1:78
		plot!(fig_a, chain_vector[i][:a], seriestype = :density, labels=false, color=:red, alpha=0.3, linewidth=1, xlabel="1/year")
	end
	title!("Growth Rate")
	fig_a
	two_cols(fig, fig_a)
end

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

In the simple model, the initial conditions are assumed to be fixed at the value from the initial scan. However, since our model depends on the initial conditions, it should contain parametric uncertainty and therefore we should include priors for the initial conditions.

```math
\begin{align} 
\sigma &\sim \Gamma^{-1}(2, 3) \\
\rho &\sim \mathcal{N}^{+}(0,5) \\
\alpha &\sim \mathcal{N}(0,5) \\
\mathbf{u_0} &\sim \mathcal{N}^{+}(\mathbf{x_0}, 5)
\end{align}
```

Including the initial conditions increases the uncertainty of trajectories projected forward.
"

# ╔═╡ 8e7fb27c-80ab-421d-b631-b739dac80efd
md"## A More Complex Model"

# ╔═╡ b8714b83-8615-4533-92c8-baa489c25ad0
html"""
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/results/subject36-81-fullposterior.png?raw=true" height=200 width=750>
<img src="https://github.com/PavanChaggar/TransferPresentation/blob/main/assets/images/TransferImages/results/subject36-u0.png?raw=true" height=200 width=750>
"""

# ╔═╡ e8f4c7a4-970b-40bf-b9aa-b1bad49888c2
md"
# Concluding remarks
Pavan, have you solved Alzheimer's disease?! 
"

# ╔═╡ 2188979a-9d1e-4500-bb8a-f44a2cf80368
md" 
# What still needs to be done?
Going forward, work will be focussed on three major themes: 
* Continued applications of modelling and inference to AD (collaborations with UCSF and Stanford).
* Developing a software pipeline for simulation and Bayesian inference.
* Quantifying uncertainty associated with network topology."

# ╔═╡ 0650ff17-68a3-4939-a86b-459abb72b3cb
md"
# Thank you! 

To my academic supervisors, Alain Goriely and Saad Jbabdi. And to my industrial supervisors, Stefano Magon and Gregory Klein at Roche. 

With the help of my supervisors and collaborators, the work presented here has been included in three publications over the past year:

* Thompson, T.B., **Chaggar, P**., Kuhl, E., Goriely, A. and Alzheimer’s Disease Neuroimaging Initiative, 2020. Protein-protein interactions in neurodegenerative diseases: a conspiracy theory. PLoS computational biology, 16(10), p.e1008267.
* Putra, P., Thompson, T.B., **Chaggar, P**., Goriely, A., Alzheimer’s Disease Neuroimaging Initiative and Alzheimer’s Disease Neuroimaging Initiative, 2021. Braiding Braak and Braak: Staging patterns and model selection in network neurodegeneration. Network Neuroscience, pp.1-41.
* Schäfer, A., **Chaggar, P**., Thompson T.B., Goriely, A., Kuhl, E,. Alzheimer’s Disease Neuroimaging Initiative, 2021. Predicting brain atrophy from tau pathology: A summary of clinical findings and their translation into personalized models. Submitted to Brain Multiphysics. **(Joint First Author)**

"

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
# ╟─1678deeb-ea59-408e-b574-5f28dc7214a0
# ╟─f45c2cd6-baf6-4ce3-84b0-5bf8fb9e67d4
# ╟─703b4044-ab3f-4e6f-a567-ba41942abe72
# ╟─53811fc6-c78e-4439-8f8c-0a002d47371a
# ╟─2967e74c-0f2b-4d7d-bc29-9c54c71cc242
# ╟─8907ecbb-2127-40e3-a012-acd52dfb2508
# ╟─7b75824e-5312-474c-8704-28f399a1ad88
# ╟─e8857f50-fdd0-49cc-950a-5490359b752f
# ╟─acefe8e5-f2a1-43a5-b585-1621a8a5fb71
# ╟─3c078fae-fa21-4256-b172-ba4b4734cdc8
# ╟─287e23e4-f01c-44f8-a0d4-7fbc0467c16f
# ╟─113f696c-129a-4862-a3cb-a814fde5e8b2
# ╟─a1e74c91-7f6a-4ca9-b497-151e2d5875c3
# ╟─7db1dc72-3b20-49a0-89e0-b53d8576a447
# ╟─ee429405-e34d-48b3-b93c-376d5eedd7ab
# ╟─66d8ea22-9bf5-4362-90d5-71471e519bd5
# ╟─703e9020-94de-493d-a15c-f11a684f0c95
# ╟─938625f5-ca78-44a8-b6cd-89988fccaf34
# ╟─b01363ae-65cb-4c9b-be74-05647e7196f5
# ╟─5bb63831-fec0-4fdc-a5a1-bf285e91168d
# ╟─cb865d40-3a15-469f-9a63-2c04a8840ca1
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
# ╟─b8714b83-8615-4533-92c8-baa489c25ad0
# ╟─e8f4c7a4-970b-40bf-b9aa-b1bad49888c2
# ╟─2188979a-9d1e-4500-bb8a-f44a2cf80368
# ╟─0650ff17-68a3-4939-a86b-459abb72b3cb
