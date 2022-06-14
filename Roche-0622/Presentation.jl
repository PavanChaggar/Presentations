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

# ╔═╡ 1f540848-eb08-11ec-32c6-d78736f8362e
begin
	using Pkg
	Pkg.activate("/Users/pavanchaggar/ResearchDocs/Presentations/Roche-0622")
end

# ╔═╡ bc07c567-bf0e-449f-adcf-afb127700900
begin
	using PlutoUI
	using Plots
	using DifferentialEquations
	using HypertextLiteral: @htl
	using Connectomes
end

# ╔═╡ 91107cb3-a72e-47a7-8d21-42b2ea11e521
include("/Users/pavanchaggar/ResearchDocs/Presentations/Roche-0622/functions.jl")

# ╔═╡ c051a690-e854-436f-993f-0fa80b000a73
html"""<style>
main {
max-width: 900px;
}"""

# ╔═╡ caeee295-cecd-4261-8846-2511cbb91d41
html"<button onclick='present()'>present</button>"

# ╔═╡ 9b698c74-dbc7-4510-ae29-ead164bcf830
c = filter(Connectome("/Users/pavanchaggar/.julia/dev/Connectomes/assets/connectomes/Connectomes-hcp-scale1.xml"));

# ╔═╡ 2aae6fa9-a3f0-451a-80dd-8b119f48072d
const L = laplacian_matrix(c);

# ╔═╡ 2e3cfdf8-2c92-422f-917e-fc5c8b2a3451
md"
# Mathematical Modelling and Inference Methods for Alzheimer's Disease

**Pavanjit Chaggar, June 2022** \
pavanjit.chaggar@maths.ox.ac.uk \
@ChaggarPavan on Twitter

DPhil student at the Mathematical Institute.
Supervised by Alain Goriely, Saad Jbabdi, Stefano Magon and Gregory Klein.
"

# ╔═╡ 0f3da277-c6ca-484f-9f83-b5899a3b2d5f
md"
# Aim
#### To build up models of AD that effectively describe our data.
"

# ╔═╡ a828a333-df39-4a4a-8744-0c235fb4342e
md"
# Overview and Introduction

- Alzheimer's disease (AD)
- Mathematical models of AD
- Patient Inference
"

# ╔═╡ 5ff7a99d-0ea0-4919-8ffc-a41ab94984fe
md"
# Alzheimer's Disease -- A Brief Summary
Alzheimer's is characterised by gradual neurodegeneration associated with the pervasive spreading of toxic proteins.

In particular, two proteins, Amyloid beta (Aβ) and tau-protein (τP), are believed to underlie and drive the development of pathology.

Here, I will focus on the role of τP, in part because it spreads very predictably and is more tightly coupled with atrophy and symptom onset.
"


# ╔═╡ c7bd3abf-e615-4acd-a0b5-24e80ecfee68
md"## Braak Stages of Tau protein
In most AD cases, τP follows a predictable pattern of spreading, starting in the entorhinal cortex before spreading through the hippocampal regions, lateral cortex and finally into the neocortex. Atrophy tends to follow the spreading pattern of $\tau$P, more so than that of Aβ."

# ╔═╡ 654bdbd1-3190-45dc-9d71-a6eb9ade28c5
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/braak-stages.png"; h=300, w=900)

# ╔═╡ 1212f837-541e-48cc-9caf-3115aee37987
md" 
# Modelling on Brain Networks! 
(The reason you keep me around.)

We want to build up models, like lego. 
"

# ╔═╡ b0618ecd-e43e-4378-b90b-5f480a601749
md" 
## Structural Connectomes 
"

# ╔═╡ 5c30120e-7923-4891-8f7f-b086bbf7f3e6
md"
The first important part of the modelling of $\tau$P in AD is describing **transport through the brain**. In this work, we do this by modelling transport as diffusion, which can be easily achieved using the graph Laplacian. The graph Laplacian is derived from a graph of brain connections, generated using tractography."

# ╔═╡ 83539771-b2bd-4ab0-b1e5-2444323c21e9
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-1221/assets/images/connectomes/connectome-diffusive.png"; h =300, w=900)

# ╔═╡ 2b2e5e0b-7ac6-40c6-84ed-d5e12fd64e95
begin
        function NetworkDiffusion(du, u, p, t)
        du .= -p * L * u
        end
        function simulate(prob, p)
                solve(remake(prob, p=p), Tsit5())
        end;

        u1 = zeros(83)
        u1[[27, 68]] .= 0.5

        p1 = 1.0
        t_span1 = (0.0,20.0)

        prob_diffusion = ODEProblem(NetworkDiffusion, u1, t_span1, p1)

        sol_diffusion = solve(prob_diffusion, Tsit5())
end;

# ╔═╡ bf14982a-7d07-4a24-8353-bab87a06f56f
md"## Diffusion Model
"

# ╔═╡ 6df47d17-e4a5-4f8b-82d6-f8c12b8f19e9
two_cols(md"",md"
ρ = $(@bind ρ1 PlutoUI.Slider(0:0.1:5, show_value=true, default=0))")

# ╔═╡ 8309cc04-26a7-4896-bf83-76977c6dd28f
TwoColumn(
md"
Using the graph Laplacian, we can define the network heat equation, which describes diffusion across a graph. To the left, I have shown a simulation with an initial seeding concentration placed in the entorhinal cortex. By changing the diffusion coefficient, $\rho$, we can see how the dynamics are affected.
\
\
$$\frac{dp_i}{dt} = \underbrace{-\rho L_{ij} p_j}_{transport}$$  
"
,     
Plots.plot(simulate(prob_diffusion, ρ1), size=(450,300), labels=false, ylims=(0.0,0.5), xlims=(0.0,20.0), ylabel="Concentration")
)

# ╔═╡ 93e36f97-bd13-43c3-8d02-d756c223383c
md" ## Diffusion Model"

# ╔═╡ 3af2f496-23b4-41fc-8072-69cf83b1d2fa
LocalResource("/Users/pavanchaggar/Projects/model-selection/diffusion.mp4")

# ╔═╡ dd63aa8e-2ef9-4d18-8f2e-cda1a825efaa
begin
		prob_fkpp = ODEProblem(NetworkFKPP, u1, t_span1, [0.1,1.0])
        sol_fkpp = solve(prob_fkpp, Tsit5())
end;

# ╔═╡ 1e157396-03ae-43ff-a3a1-dec8776507e6
md" 
## FKPP Model
"

# ╔═╡ a11bfbd6-703b-427b-ae20-931dc40e7973
two_cols(md"",
md"
ρ = $(@bind ρ Slider(0:0.1:3, show_value=true, default=0)) \
α = $(@bind α Slider(-3:0.1:3, show_value=true, default=0))
")

# ╔═╡ ed5dfdbf-db67-47cd-8a06-dbf7c80dc336
TwoColumn(
md"
The next piece we want to add to the model is **autocatalytic protein growth**, to more accurately describe toxic protein dynamics. The most simple such term we could include is a quadratic term that is bounded between $0$ and $1$. 
\
\
The effect of this quadratic term is exponential growth given a positive concentration of toxic protein that saturates as the concentration grows.
\
\
$$\frac{d p_i}{dt} = \underbrace{-\rho L_{ij}p_j}_{transport} + \underbrace{\alpha p_i\left(1-p_i\right)}_{growth}$$
",     
Plots.plot(simulate(prob_fkpp, [ρ, α]), size=(450,300), labels=false, ylims=(0.0,1.0), xlims=(0.0,20.0), ylabel="concentration"))


# ╔═╡ 84f50b04-25d1-412e-bacf-5c0e9299eb63
md"
## FKPP Model"

# ╔═╡ 607d0291-89f3-4d4e-bb53-cc4de43de049
LocalResource("/Users/pavanchaggar/Projects/model-selection/fkpp.mp4")

# ╔═╡ 57f7b7e2-ded0-4eac-87a4-2077b3522535
md"## Regional FKPP model"

# ╔═╡ 15fbec7e-ae2c-4ffe-86c4-b6b1beacdfb3
two_cols(md"
Now we have transport and growth, the next piece we want to add is **regional specificity**, such as baseline SURV values and carrying capacities.

We *estimate* these using Gaussian mixture modelling of population SUVR data per region.

GMMs are fit to data from BioFinder, which has much better coverage of late stage AD subjects, which results in less sampling bias compared to ADNI.",
	
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-0622/assets/images/gmm-lEC.png"; h = 275, w=450))

# ╔═╡ ece49802-e660-48fb-8592-f9a4098f10e8
md"
## Regional FKPP Model"

# ╔═╡ ef098338-1b67-4682-bd05-e4154e5a420f
LocalResource("/Users/pavanchaggar/Projects/model-selection/exfkpp.mp4")

# ╔═╡ dc8da42d-afdb-423b-812e-01160ccf637a
md" 
# Summary

* Models are like lego!
* Each piece we add make the dynamics more expressive and better able to describe AD. 
"

# ╔═╡ d3a9829f-7ac4-4465-acb5-277d09cacce4
md" 
# Fitting to Subject Data
'Ok, Pavan, you've shown off your models, now tell us how they're useful?'
"

# ╔═╡ a582faef-85ac-4a51-ba4f-5bbf1e2e630f
md" 
## Single Subject Inference and Predictions
"

# ╔═╡ 5905156f-3fd2-4ab2-ac53-305eba364f03
TwoColumn(
	md"As a case study, we'll look at a single subject form the ADNI dataset.

* The subject is A$\beta^{+}$ with 4 tau-PET scans.
* We use Bayesian inference to fit models to patient data. 
* For each model, we infer the model parameters, initial conditions and observation noise. 
* We use a NUTS sampler for efficient sampling
	", 
	
	pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-0622/assets/images/sub12/data.png"; h=350, w=1000)
)

# ╔═╡ 297e476a-bd0d-4306-9495-65ef12c7c7bf
md" 
## Inference and Posterior Summary
" 

# ╔═╡ 78ae1e7f-9e12-47cd-899a-b8e1fa0c6ed5
TwoColumn(
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-0622/assets/images/sub12/posteriorsummary.png"; h = 400, w=550), 
md"
We can do model comparison using the AIC score:

| Diffusion |  FKPP | Ex. FKPP |
|-----------|-------|----------|
|-57.41     |-355.81| -564.46  |

"
)

# ╔═╡ af440978-f6f2-4a37-9af1-b9972a1240d2
md" 
##  Making Predictions...
"

# ╔═╡ b60c10fd-3407-4130-b265-62b7cfb622cb
md"
Since we used a Bayesian approach, we can examine the posterior distributions for parameters and run forward simulations.
"

# ╔═╡ 492cc06c-27a4-4260-bb2f-09b6543df85b
pic("https://github.com/PavanChaggar/Presentations/blob/master/Roche-0622/assets/images/sub12/sub-12-EC.png"; h = 350, w=900)

# ╔═╡ 9b8853ca-4c6f-421a-8c88-379180500225
md" 
# Predicting Patient Trajectories
"

# ╔═╡ 73b8faa9-2cd2-406a-aaa6-ff01225a5267
LocalResource("/Users/pavanchaggar/Projects/model-selection/sub12-exfkpp.mp4")

# ╔═╡ 077c5434-21c1-4660-a4ab-e1165daac50c
md"
# Limitations + Conclusions"

# ╔═╡ 255536ce-88eb-474b-b465-84a75edbd767
md" 
## Limitations + Conclusions 
* We can use models to describe various processes in AD. 
* ...However, the models are macro-scale descriptions of AD that don't provide a lot of mechanistic **explanation**.

* Using Bayesian inference, we can calibrate models and estimate patient disease trajectories. 
* ...However, we have very limited and noisy longitudinal patient data that makes it hard to perform model comparison. 
"

# ╔═╡ 82411fe9-4773-4aea-8710-f2ae15692585
md"# Questions?"

# ╔═╡ Cell order:
# ╠═1f540848-eb08-11ec-32c6-d78736f8362e
# ╠═c051a690-e854-436f-993f-0fa80b000a73
# ╠═caeee295-cecd-4261-8846-2511cbb91d41
# ╠═bc07c567-bf0e-449f-adcf-afb127700900
# ╠═9b698c74-dbc7-4510-ae29-ead164bcf830
# ╠═2aae6fa9-a3f0-451a-80dd-8b119f48072d
# ╠═91107cb3-a72e-47a7-8d21-42b2ea11e521
# ╟─2e3cfdf8-2c92-422f-917e-fc5c8b2a3451
# ╟─0f3da277-c6ca-484f-9f83-b5899a3b2d5f
# ╟─a828a333-df39-4a4a-8744-0c235fb4342e
# ╟─5ff7a99d-0ea0-4919-8ffc-a41ab94984fe
# ╟─c7bd3abf-e615-4acd-a0b5-24e80ecfee68
# ╟─654bdbd1-3190-45dc-9d71-a6eb9ade28c5
# ╟─1212f837-541e-48cc-9caf-3115aee37987
# ╟─b0618ecd-e43e-4378-b90b-5f480a601749
# ╟─5c30120e-7923-4891-8f7f-b086bbf7f3e6
# ╟─83539771-b2bd-4ab0-b1e5-2444323c21e9
# ╟─2b2e5e0b-7ac6-40c6-84ed-d5e12fd64e95
# ╟─bf14982a-7d07-4a24-8353-bab87a06f56f
# ╟─8309cc04-26a7-4896-bf83-76977c6dd28f
# ╟─6df47d17-e4a5-4f8b-82d6-f8c12b8f19e9
# ╟─93e36f97-bd13-43c3-8d02-d756c223383c
# ╟─3af2f496-23b4-41fc-8072-69cf83b1d2fa
# ╟─dd63aa8e-2ef9-4d18-8f2e-cda1a825efaa
# ╟─1e157396-03ae-43ff-a3a1-dec8776507e6
# ╟─ed5dfdbf-db67-47cd-8a06-dbf7c80dc336
# ╟─a11bfbd6-703b-427b-ae20-931dc40e7973
# ╟─84f50b04-25d1-412e-bacf-5c0e9299eb63
# ╟─607d0291-89f3-4d4e-bb53-cc4de43de049
# ╟─57f7b7e2-ded0-4eac-87a4-2077b3522535
# ╟─15fbec7e-ae2c-4ffe-86c4-b6b1beacdfb3
# ╟─ece49802-e660-48fb-8592-f9a4098f10e8
# ╟─ef098338-1b67-4682-bd05-e4154e5a420f
# ╟─dc8da42d-afdb-423b-812e-01160ccf637a
# ╟─d3a9829f-7ac4-4465-acb5-277d09cacce4
# ╟─a582faef-85ac-4a51-ba4f-5bbf1e2e630f
# ╟─5905156f-3fd2-4ab2-ac53-305eba364f03
# ╟─297e476a-bd0d-4306-9495-65ef12c7c7bf
# ╟─78ae1e7f-9e12-47cd-899a-b8e1fa0c6ed5
# ╟─af440978-f6f2-4a37-9af1-b9972a1240d2
# ╟─b60c10fd-3407-4130-b265-62b7cfb622cb
# ╟─492cc06c-27a4-4260-bb2f-09b6543df85b
# ╟─9b8853ca-4c6f-421a-8c88-379180500225
# ╟─73b8faa9-2cd2-406a-aaa6-ff01225a5267
# ╟─077c5434-21c1-4660-a4ab-e1165daac50c
# ╟─255536ce-88eb-474b-b465-84a75edbd767
# ╟─82411fe9-4773-4aea-8710-f2ae15692585
