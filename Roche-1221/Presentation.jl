### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
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
# Mathematical Modelling and Inference Methods for Alzheimer's Disease

**Pavanjit Chaggar, December 2021**

DPhil student at the Mathematical Institute.
Supervised by Alain Goriely and Saad Jbabdi, with support from Stefano Magon and Gregory Klein at Roche.
"

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


# ╔═╡ 70d3f5ff-aa7e-4cc3-8aca-db403a7de855
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/braak-stages.png"; h=300, w=900)

# ╔═╡ c484008a-ec30-4d73-bc6e-f462a5d187b1
md" 
## Transport and the Graph Laplacian

An important part of the modelling of $\tau$P in AD is describing transport. In this work, we do this using the graph Laplacian, a discrete counterpart to the Laplace operator used to describe diffusion in a continuous setting. The graph Laplacian is derived from a graph of brain connections, generated using tractography. "

# ╔═╡ a1e74c91-7f6a-4ca9-b497-151e2d5875c3
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/connectomes/connectome-pit.png"; h =300, w=900)

# ╔═╡ 5a8dc4c9-246c-4dd1-ba62-638f0879d7b7
md" 
## Transport and the Graph Laplacian 
 
Using tractography, we can define an adjacency matrix,

```math
\begin{equation}
\mathbf{A}_{ij} = \left\{\begin{array}{cl} n_{ij}/l^{2}_{ij} & \text{ an edge connects } v_i \text{ to } v_j\\ 0 
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
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/heterodimerkinetics.png"; h=150, w=300, vspace=20, hspace=275)

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
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/fkppkinetics.png"; h=150, w=300, vspace=10, hspace=290)

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
md"# Single Subject Results
We sample using NUTS, a form of Hamiltonian Monte Carlo. The distributions for the diffusion coefficient, $\rho$, indicate a *very* slow rate of diffusion, on the order of mm/year, across all subjects. There is more variation in the individual posteriors for growth rate $\alpha$, suggesting that it may be the more important driver of disease pathology."

# ╔═╡ 5d9fc177-587c-4037-aa69-d946613fd677
md"## Posterior Distributions" 

# ╔═╡ 403886c8-ece8-45cc-a086-f2424665c704
md"### Diffusion"

# ╔═╡ 99976a74-fe57-4474-947a-fed3853da11d
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/diffusion-chain.png"; h=200, w=900)

# ╔═╡ 9bef967a-f058-4c73-9e27-0e35b87b0762
md"### Growth"

# ╔═╡ c7b9ce4f-d1f7-4f10-a001-895088b7bce6
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/growth-chain.png"; h=200, w=900)

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
	scatter!(reltime1, suvr[:,t_index[sub1,:],sub1]', label=false, color=:grey)
	scatter!(reltime1, suvr[[81],t_index[sub1,:],sub1]', label=false, color=:red)
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
<img src="https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/subject-36-fullposterior.png?raw=true" height=250 width=800>
<img src="https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/subject36-u0.png?raw=true" height=250 width=800>
"""

# ╔═╡ 366907e2-a44e-465a-9b64-15e2ea636de3
md"# Hierarchical Inference

Can we use all of the available data from ADNI at the same time to group together information about dynamics? 

Yes!

To maximise the amount of data we have in the model, we use a coupled model of protein dynamics with atrophy. Additionally, we seperate ADNI data into groups based whether subjects are $A\beta^{+}$ or $A\beta^{-}$.
>```math
>\begin{align}
>\frac{d p_i}{dt} &= -\rho\sum\limits_{j=1}^{N}L_{ij}p_j + \alpha p_i\left(1-p_i\right) \\
>\frac{dq_i}{dt} &= \beta p_i( 1 - q_i ),
>\end{align}
>```" 

# ╔═╡ 3fb86091-51cb-48ce-b5c9-bf94e1ae693c
md" ## Defining the Hierarchical Model

Shown here is a *plate diagram* that shows the model structure we assume for the hierarchical model
"

# ╔═╡ 314c6df5-3a59-4a48-98e9-3fc2ed602565
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/hierarchical-plate.jpeg"; h=550, w=900)

# ╔═╡ 48cc4be3-f6e9-459e-8305-454f1949e69c
md" ## Posterior Distributions"

# ╔═╡ 46520185-aa1e-4736-96be-0a13e00f8938
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/hierarchical-diffusion.png"; h=250, w=900)

# ╔═╡ 98921efa-e07c-41f6-82ea-68c0a69b3b44
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/hierarchical-growth.png"; h=250, w=900)

# ╔═╡ 08906542-caf1-4a2a-866f-70c57e0449c6
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/hierarchical-atrophy.png"; h=250, w=900)

# ╔═╡ 2e3c459e-2e48-4706-8e80-8846cdce8a5f
md"## Predictions: Hierarchical vs Individual Inference"

# ╔═╡ 90f982da-9dd9-47bb-892c-7f9dd0ab1779
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/results/subject36-pos-v-hpos.png"; h=300, w=900)

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
PlotThemes = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
DifferentialEquations = "~6.20.0"
HypertextLiteral = "~0.9.3"
Images = "~0.25.0"
MAT = "~0.10.1"
PlotThemes = "~2.0.1"
Plots = "~1.24.3"
PlutoUI = "~0.7.21"
StatsPlots = "~0.14.29"
Turing = "~0.19.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "db0a7ff3fbd987055c43b4e12d2fa30aaae8749c"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "3.2.1"

[[AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "15f34cc635546ac072d03fc2cc10083adb4df680"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.2.0"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "0a655e9a59ee1c8bafc3af18e96d90f980b08600"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.2"

[[AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "8ad8bfddf8bb627d689ecb91599c349cbf15e971"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.6"

[[AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "06da6c283cf17cf0f97ed2c07c29b6333ee83dc9"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.2.4"

[[AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "265b06e2b1f6a216e0e8f183d28e4d354eab3220"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.1"

[[ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e1ba79094cae97b688fb42d31cbbfd63a69706e4"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.7.8"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AutoHashEquals]]
git-tree-sha1 = "45bb6705d93be619b81451bb2006b7ee5d4e4453"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "0.2.0"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "ce68f8c2162062733f9b4c9e3700d5efc4a8ec47"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "0.16.11"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "0ad226aa72d8671f20d0316e03028f0ba1624307"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.32"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[Bijections]]
git-tree-sha1 = "705e7822597b432ebe152baa844b49f8026df090"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.3"

[[Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "Functors", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "369af32fcb9be65d496dc43ad0bb713705d4e859"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.9.11"

[[BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "bc1317f71de8dce26ea67fcdf7eccc0d0693b75b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.1"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "217da19d6f3a94753e580a8bc241c7cbefd9281f"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.1"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e747dac84f39c62aff6956651ec359686490134e"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.0+0"

[[BoundaryValueDiffEq]]
deps = ["BandedMatrices", "DiffEqBase", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "NLsolve", "Reexport", "SparseArrays"]
git-tree-sha1 = "fe34902ac0c3a35d016617ab7032742865756d7d"
uuid = "764a87c0-6b3e-53db-9096-fe964310641d"
version = "2.7.1"

[[BufferedStreams]]
deps = ["Compat", "Test"]
git-tree-sha1 = "5d55b9486590fdda5905c275bb21ce1f0754020f"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.0.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CPUSummary]]
deps = ["Hwloc", "IfElse", "Static"]
git-tree-sha1 = "87b0c9c6ee0124d6c1f4ce8cb035dcaf9f90b803"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.6"

[[CSTParser]]
deps = ["Tokenize"]
git-tree-sha1 = "f9a6389348207faf5e5c62cbc7e89d19688d338a"
uuid = "00ebfdb7-1f24-5e51-bd34-a7502290713f"
version = "3.3.0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "RealDot", "Statistics"]
git-tree-sha1 = "65a0826ac25869556070076459925d1700316aab"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.14.0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "7b8f09d58294dc8aa13d91a8544b37c8a1dcbc06"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.4"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonMark]]
deps = ["Crayons", "JSON", "URIs"]
git-tree-sha1 = "393ac9df4eb085c2ab12005fc496dae2e1da344e"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.8.3"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[DEDataArrays]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecursiveArrayTools", "SciMLBase", "StaticArrays"]
git-tree-sha1 = "31186e61936fbbccb41d809ad4338c9f7addf7ae"
uuid = "754358af-613d-5f8d-9788-280bf1605d4c"
version = "0.2.0"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[DelayDiffEq]]
deps = ["ArrayInterface", "DataStructures", "DiffEqBase", "LinearAlgebra", "Logging", "NonlinearSolve", "OrdinaryDiffEq", "Printf", "RecursiveArrayTools", "Reexport", "UnPack"]
git-tree-sha1 = "9edf8c0f27c16df239d31dd55b154096e4c41cda"
uuid = "bcd4f6db-9728-5f36-b5f7-82caef46ccdb"
version = "5.31.2"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiffEqBase]]
deps = ["ArrayInterface", "ChainRulesCore", "DEDataArrays", "DataStructures", "Distributions", "DocStringExtensions", "FastBroadcast", "ForwardDiff", "FunctionWrappers", "IterativeSolvers", "LabelledArrays", "LinearAlgebra", "Logging", "MuladdMacro", "NonlinearSolve", "Parameters", "PreallocationTools", "Printf", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "Requires", "SciMLBase", "Setfield", "SparseArrays", "StaticArrays", "Statistics", "SuiteSparse", "ZygoteRules"]
git-tree-sha1 = "5c3d877ddfc2da61ce5cc1f5ce330ff97789c57c"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.76.0"

[[DiffEqCallbacks]]
deps = ["DataStructures", "DiffEqBase", "ForwardDiff", "LinearAlgebra", "NLsolve", "OrdinaryDiffEq", "Parameters", "RecipesBase", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "35bc7f8be9dd2155336fe999b11a8f5e44c0d602"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "2.17.0"

[[DiffEqFinancial]]
deps = ["DiffEqBase", "DiffEqNoiseProcess", "LinearAlgebra", "Markdown", "RandomNumbers"]
git-tree-sha1 = "db08e0def560f204167c58fd0637298e13f58f73"
uuid = "5a0ffddc-d203-54b0-88ba-2c03c0fc2e67"
version = "2.4.0"

[[DiffEqJump]]
deps = ["ArrayInterface", "Compat", "DataStructures", "DiffEqBase", "FunctionWrappers", "Graphs", "LinearAlgebra", "PoissonRandom", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "StaticArrays", "TreeViews", "UnPack"]
git-tree-sha1 = "0aa2d003ec9efe2a93f93ae722de05a870ffc0b2"
uuid = "c894b116-72e5-5b58-be3c-e6d8d4ac2b12"
version = "8.0.0"

[[DiffEqNoiseProcess]]
deps = ["DiffEqBase", "Distributions", "LinearAlgebra", "Optim", "PoissonRandom", "QuadGK", "Random", "Random123", "RandomNumbers", "RecipesBase", "RecursiveArrayTools", "Requires", "ResettableStacks", "SciMLBase", "StaticArrays", "Statistics"]
git-tree-sha1 = "d6839a44a268c69ef0ed927b22a6f43c8a4c2e73"
uuid = "77a26b50-5914-5dd7-bc55-306e6241c503"
version = "5.9.0"

[[DiffEqPhysics]]
deps = ["DiffEqBase", "DiffEqCallbacks", "ForwardDiff", "LinearAlgebra", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "Reexport", "StaticArrays"]
git-tree-sha1 = "8f23c6f36f6a6eb2cbd6950e28ec7c4b99d0e4c9"
uuid = "055956cb-9e8b-5191-98cc-73ae4a59e68a"
version = "3.9.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "d8f468c5cd4d94e86816603f7d18ece910b4aaf1"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.5.0"

[[DifferentialEquations]]
deps = ["BoundaryValueDiffEq", "DelayDiffEq", "DiffEqBase", "DiffEqCallbacks", "DiffEqFinancial", "DiffEqJump", "DiffEqNoiseProcess", "DiffEqPhysics", "DimensionalPlotRecipes", "LinearAlgebra", "MultiScaleArrays", "OrdinaryDiffEq", "ParameterizedFunctions", "Random", "RecursiveArrayTools", "Reexport", "SteadyStateDiffEq", "StochasticDiffEq", "Sundials"]
git-tree-sha1 = "91df208ee040be7960c408d4681bf91974bcb4f4"
uuid = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
version = "6.20.0"

[[DimensionalPlotRecipes]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "af883a26bbe6e3f5f778cb4e1b81578b534c32a6"
uuid = "c619ae07-58cd-5f6d-b883-8f17bd6a98f9"
version = "1.2.0"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "837c83e5574582e07662bbbba733964ff7c26b9d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.6"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "7f3bec11f4bcd01bc1f507ebce5eadf1b0a78f47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.34"

[[DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "55c82c84759cff92189c9ad07e7bd0a0df63e900"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.34"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "5f5f0b750ac576bcf2ab1d7782959894b304923e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.9"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "fe385ec95ac5533650fb9b1ba7869e9bc28cdd0a"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.5"

[[DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "Random", "ZygoteRules"]
git-tree-sha1 = "532397f64ad49472fb60e328369ecd5dedeff02f"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.15.1"

[[DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "1b4665a7e303eaa7e03542cfaef0730cb056cb00"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.3.21"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "3fe985505b4b667e1ae303c9ca64d181f09d5c05"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.3"

[[EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "254182080498cce7ae4bc863d23bf27c632688f7"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.4.4"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExponentialUtilities]]
deps = ["ArrayInterface", "LinearAlgebra", "Printf", "Requires", "SparseArrays"]
git-tree-sha1 = "1b873816d2cfc8c0fcb1edcb08e67fdf630a70b7"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.10.2"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FastBroadcast]]
deps = ["LinearAlgebra", "Polyester", "Static"]
git-tree-sha1 = "e32a81c505ab234c992ca978f31ed8b0dabbc327"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.1.11"

[[FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "6406b5112809c08b1baa5703ad274e1dded0652f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.23"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[Functors]]
git-tree-sha1 = "e4768c3b7f597d5a352afa09874d16e3c3f6ead2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.7"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "74ef6288d071f58033d54fd6708d4bc23a8b8972"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+1"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "92243c07e786ea3458532e199eb3feee0e7e08eb"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.4.1"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HDF5]]
deps = ["Blosc", "Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "698c099c6613d7b7f151832868728f426abe698b"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.15.7"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fd83fa0bde42e01952757f01149dd968c06c4dba"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.0+1"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8f0dc80088981ab55702b04bba38097a44a1a3a9"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.5"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "7a20463713d239a19cbad3f6991e404aca876bda"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.15"

[[ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "a2951c93684551467265e0e32b577914f69532be"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.9"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ea2b6fd947cdfc43c6b8c15cff982533ec1f72cd"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+0"

[[ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "5581e18a74a5838bd919294a7138c2663d065238"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.0"

[[ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1d2d73b14198d10f7f12bf7f8481fd4b3ff5cd61"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.0"

[[ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "d0ac64c9bee0aed6fdbb2bc0e5dfa9a3a78e3acc"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.3"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "b4b161abc8252d68b13c5cc4a5f2ba711b61fec5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.3"

[[Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "35dc1cd115c57ad705c7db9f6ef5cc14412e8f00"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.0"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InitialValues]]
git-tree-sha1 = "7f6a4508b4a6f46db5ccd9799a3fc71ef5cad6e6"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.11"

[[InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[IntegralArrays]]
deps = ["IntervalSets"]
git-tree-sha1 = "4fdfe55b432bbb97adbb0c85c39dd208a7b2bd36"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.1"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "46b7834ec8165c541b0b5d1c8ba63ec940723ffb"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.15"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[JuliaFormatter]]
deps = ["CSTParser", "CommonMark", "DataStructures", "Pkg", "Tokenize"]
git-tree-sha1 = "e1a91af84efba5228480bd20aeef1dd902d553cf"
uuid = "98e50ef6-434e-11e9-1051-2b60c6c9e899"
version = "0.19.2"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "3609bbf5feba7b22fb35fe7cb207c8c8d2e2fc5b"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.6.7"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "83b56449c39342a47f3fcdb3bc782bd6d66e1d97"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.4"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "71be1eb5ad19cb4f61fa8c73395c0338fd092ae0"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtask]]
deps = ["Libtask_jll", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "90c6ed7f9ac449cddacd80d5c1fca59c97d203e7"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.5.3"

[[Libtask_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "901fc8752bbc527a6006a951716d661baa9d54e9"
uuid = "3ae2931a-708c-5973-9c38-ccf7496fb450"
version = "0.4.3+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "Requires", "SIMDDualNumbers", "SLEEFPirates", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "9e10579c154f785b911d9ceb96c33fcc1a661171"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.99"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "5c62992f3d46b8dce69bdd234279bb5a369db7d5"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.1"

[[MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "04c3fd6da28ebd305120ffb05f0a3b8f9baced0a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.0.1"

[[MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "6aba85ab1ac0515a1d14330bf347a8634ea5776c"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.2"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0174e9d180b0cae1f8fe7976350ad52f0e70e0d8"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.3.3"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[ManualMemory]]
git-tree-sha1 = "9cb207b18148b2199db259adfa923b45593fe08e"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.6"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[Metatheory]]
deps = ["AutoHashEquals", "DataStructures", "Dates", "DocStringExtensions", "Parameters", "Reexport", "TermInterface", "ThreadsX", "TimerOutputs"]
git-tree-sha1 = "0d3b2feb3168e4deb78361d3b5bb5c2e51ea5271"
uuid = "e9d8d322-4543-424a-9be4-0cc815abe26c"
version = "1.3.2"

[[MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "4f65bdbbe93475f6ff9ea6969b21532f88d359be"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[ModelingToolkit]]
deps = ["AbstractTrees", "ArrayInterface", "ConstructionBase", "DataStructures", "DiffEqBase", "DiffEqCallbacks", "DiffEqJump", "DiffRules", "Distributed", "Distributions", "DocStringExtensions", "DomainSets", "Graphs", "IfElse", "InteractiveUtils", "JuliaFormatter", "LabelledArrays", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "NaNMath", "NonlinearSolve", "RecursiveArrayTools", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SafeTestsets", "SciMLBase", "Serialization", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "Symbolics", "UnPack", "Unitful"]
git-tree-sha1 = "d9bbe2b0141a2387dcb6cd00a7019a5b39d7d8b9"
uuid = "961ee093-0014-501f-94e3-6117800e7a78"
version = "7.1.3"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MuladdMacro]]
git-tree-sha1 = "c6190f9a7fc5d9d5915ab29f2134421b12d24a68"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.2"

[[MultiScaleArrays]]
deps = ["DiffEqBase", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "OrdinaryDiffEq", "Random", "RecursiveArrayTools", "SparseDiffTools", "Statistics", "StochasticDiffEq", "TreeViews"]
git-tree-sha1 = "258f3be6770fe77be8870727ba9803e236c685b8"
uuid = "f9640e96-87f6-5992-9c3b-0743c6a49ffa"
version = "1.8.1"

[[MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "45c9940cec79dedcdccc73cc6dd09ea8b8ab142c"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.3.18"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "8d9496b2339095901106961f44718920732616bb"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.22"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "2eb305b13eaed91d7da14269bf17ce6664bfee3d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.31"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "e9ffc92217b8709e0cf7b8808f6223a4a0936c95"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.11"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "35d435b512fbab1d1a29138b5229279925eba369"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.5.0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[OrdinaryDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "ExponentialUtilities", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "Logging", "LoopVectorization", "MacroTools", "MuladdMacro", "NLsolve", "Polyester", "PreallocationTools", "RecursiveArrayTools", "Reexport", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "758a3cc612baa4ec194dd2e71625d091edd8675a"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "5.68.0"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "6d105d40e30b635cfed9d52ec29cf456e27d38f8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.12"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[ParameterizedFunctions]]
deps = ["DataStructures", "DiffEqBase", "DocStringExtensions", "Latexify", "LinearAlgebra", "ModelingToolkit", "Reexport", "SciMLBase"]
git-tree-sha1 = "3baa1ad75b77f406988be4dc0364e01cf16127e7"
uuid = "65888b18-ceab-5e60-b2b9-181511a3b968"
version = "5.12.2"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "d73736030a094e8d24fdf3629ae980217bf1d59d"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.24.3"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "b68904528fd538f1cb6a3fbc44d2abdc498f9e8e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.21"

[[PoissonRandom]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "44d018211a56626288b5d3f8c6497d28c26dc850"
uuid = "e409e4f3-bfea-5376-8464-e040bb5c01ab"
version = "0.4.0"

[[Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "892b8d9dd3c7987a4d0fd320f0a421dd90b5d09d"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.5.4"

[[PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "a3ff99bf561183ee20386aec98ab8f4a12dc724a"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.2"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "LabelledArrays"]
git-tree-sha1 = "ba819074442cd4c9bda1a3d905ec305f8acb37f2"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.2.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[Quaternions]]
deps = ["DualNumbers", "LinearAlgebra"]
git-tree-sha1 = "adf644ef95a5e26c8774890a509a55b7791a139f"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Random123]]
deps = ["Libdl", "Random", "RandomNumbers"]
git-tree-sha1 = "0e8b146557ad1c6deb1367655e052276690e71a3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.4.2"

[[RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "c944fa4adbb47be43376359811c0a14757bdc8a8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.20.0"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "b7edd69c796b30985ea6dfeda8504cdb7cf77e9f"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.5"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[ResettableStacks]]
deps = ["StaticArrays"]
git-tree-sha1 = "256eeeec186fa7f26f2801732774ccf277f05db9"
uuid = "ae5879a3-cd67-5da8-be7f-38c6eb64a37b"
version = "1.1.1"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "51ee572776905ee34c0568f5efe035d44bf59f74"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.3.11"

[[Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "dbf5f991130238f10abbf4f2d255fb2837943c43"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.1.0"

[[RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cdc1e4278e91a6ad530770ebb327f9ed83cf10c4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMDDualNumbers]]
deps = ["ForwardDiff", "IfElse", "SLEEFPirates", "VectorizationBase"]
git-tree-sha1 = "62c2da6eb66de8bb88081d20528647140d4daa0e"
uuid = "3cdde19b-5bb0-4aaf-8931-af3e248e098b"
version = "0.1.0"

[[SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "1410aad1c6b35862573c01b96cd1f6dbe3979994"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.28"

[[SafeTestsets]]
deps = ["Test"]
git-tree-sha1 = "36ebc5622c82eb9324005cc75e7e2cc51181d181"
uuid = "1bc83da4-3b8d-516f-aca4-4fe02f6d838f"
version = "0.0.1"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "b3d23aa4e5f621b574b3b0d41c62c8624d27192a"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.19.5"

[[ScientificTypesBase]]
git-tree-sha1 = "185e373beaf6b381c1e7151ce2c2a722351d6637"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.3.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "def0718ddbabeb5476e51e5a43609bee889f285d"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.0"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SparseDiffTools]]
deps = ["Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays", "VertexSafeGraphs"]
git-tree-sha1 = "f87076b43379cb0bd9f421cfe7c649fb510d8e4e"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "1.18.1"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "d6956cefe3766a8eb5caae9226118bb0ac61c8ac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.29"

[[SteadyStateDiffEq]]
deps = ["DiffEqBase", "DiffEqCallbacks", "LinearAlgebra", "NLsolve", "Reexport", "SciMLBase"]
git-tree-sha1 = "3e057e1f9f12d18cac32011aed9e61eef6c1c0ce"
uuid = "9672c7b4-1e72-59bd-8a11-6ac3964bc41f"
version = "1.6.6"

[[StochasticDiffEq]]
deps = ["Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DiffEqJump", "DiffEqNoiseProcess", "DocStringExtensions", "FillArrays", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "Logging", "MuladdMacro", "NLsolve", "OrdinaryDiffEq", "Random", "RandomNumbers", "RecursiveArrayTools", "Reexport", "SparseArrays", "SparseDiffTools", "StaticArrays", "UnPack"]
git-tree-sha1 = "d6756d0c66aecd5d57ad9d305d7c2526fb5922d9"
uuid = "789caeaf-c7a9-5a7d-9973-96adeb23e2a0"
version = "6.41.0"

[[StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "12cf3253ebd8e2a3214ae171fbfe51e7e8d8ad28"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.2.9"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"

[[Sundials]]
deps = ["CEnum", "DataStructures", "DiffEqBase", "Libdl", "LinearAlgebra", "Logging", "Reexport", "SparseArrays", "Sundials_jll"]
git-tree-sha1 = "12d529a67c232bd27e9868fbcfad4997435786a5"
uuid = "c3572dad-4567-51f8-b174-8c6c989267f4"
version = "4.6.0"

[[Sundials_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl", "OpenBLAS_jll", "Pkg", "SuiteSparse_jll"]
git-tree-sha1 = "013ff4504fc1d475aa80c63b455b6b3a58767db2"
uuid = "fb77eaff-e24c-56d4-86b1-d163f2edb164"
version = "5.2.0+1"

[[SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "Metatheory", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TermInterface", "TimerOutputs"]
git-tree-sha1 = "5255e65d129c8edbde92fd2ede515e61098f93df"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.18.1"

[[Symbolics]]
deps = ["ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "Metatheory", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TermInterface", "TreeViews"]
git-tree-sha1 = "1f738ebade1567d461c4db9ef15470a1fdf0b9b5"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "4.1.1"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[TermInterface]]
git-tree-sha1 = "7aa601f12708243987b88d1b453541a75e3d8c7a"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "0.2.3"

[[TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "03013c6ae7f1824131b2ae2fc1d49793b51e8394"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.6"

[[ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "abcff3ac31c7894550566be533b512f8b059104f"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.8"

[[TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "c342ae2abf4902d65a0b0bf59b28506a6e17078a"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.5.2"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7cb456f358e8f9d102a8b25e8dfedf58fa5689bc"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.13"

[[Tokenize]]
git-tree-sha1 = "0952c9cee34988092d73a5708780b3917166a0dd"
uuid = "0796e94c-ce3b-5d07-9a54-7f471281c624"
version = "0.5.21"

[[Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "bccb153150744d476a6a8d4facf5299325d5a442"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.67"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "ec9a310324dd2c546c07f33a599ded9c1d00a420"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.8"

[[Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "c40eac01a46b8efc8af9481982a618d6e54b506d"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.19.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "0992ed0c3ef66b0390e5752fe60054e5ff93b908"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.9.2"

[[VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "17e5847bb36730d90801170ecd0ce4041a3dde86"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.22"

[[VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
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
# ╟─70d3f5ff-aa7e-4cc3-8aca-db403a7de855
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
# ╟─f07caceb-559e-4e76-b52c-890290efa64e
# ╟─9066ba33-ddc3-4497-b896-393458faad92
# ╟─358c91c1-cbd5-4802-a9d7-72a390a6b91b
# ╟─7b020a5c-5657-41d3-bb34-cfc1df999494
# ╟─6749c00c-38ef-4f5b-a446-f8e43d8dddb2
# ╟─63136cdf-8481-41b2-98c7-04c3d4e46778
# ╟─96a18032-5d54-4d06-a00f-acdd4d4b25ed
# ╟─95b23324-b326-4ef7-9031-9afb3cc92d7b
# ╟─1488dec0-8e37-4e0c-84bb-3d4abcdebfd0
# ╟─5d9fc177-587c-4037-aa69-d946613fd677
# ╟─403886c8-ece8-45cc-a086-f2424665c704
# ╟─99976a74-fe57-4474-947a-fed3853da11d
# ╟─9bef967a-f058-4c73-9e27-0e35b87b0762
# ╟─c7b9ce4f-d1f7-4f10-a001-895088b7bce6
# ╟─b8e8837e-345d-4539-aeb4-5d5575e08cdc
# ╟─e48ce2b1-7c15-481c-97ad-5cb7d8ea4b54
# ╟─1c86e2f4-bed6-4327-b977-6d732c57dd81
# ╟─78f4378e-6ddb-416f-9d66-4eae762ce8b0
# ╟─5acc04c0-ba56-4d07-b2b0-323596c8a420
# ╟─b515a700-2ca8-4efc-b38f-32e63803b53d
# ╟─8e7fb27c-80ab-421d-b631-b739dac80efd
# ╟─b8714b83-8615-4533-92c8-baa489c25ad0
# ╟─366907e2-a44e-465a-9b64-15e2ea636de3
# ╟─3fb86091-51cb-48ce-b5c9-bf94e1ae693c
# ╟─314c6df5-3a59-4a48-98e9-3fc2ed602565
# ╟─48cc4be3-f6e9-459e-8305-454f1949e69c
# ╟─46520185-aa1e-4736-96be-0a13e00f8938
# ╟─98921efa-e07c-41f6-82ea-68c0a69b3b44
# ╟─08906542-caf1-4a2a-866f-70c57e0449c6
# ╟─2e3c459e-2e48-4706-8e80-8846cdce8a5f
# ╟─90f982da-9dd9-47bb-892c-7f9dd0ab1779
# ╟─65748ba6-52bf-442f-bb1e-ecaabbd7c909
# ╟─e8f4c7a4-970b-40bf-b9aa-b1bad49888c2
# ╟─2c2a58ff-e916-495f-bc96-a0530392a273
# ╟─48073ebb-3ed7-4f54-a6a2-b6201eb08f89
# ╟─c973adef-46eb-4ec6-a23d-1e9b0aad220a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
