using Connectomes
using GLMakie, FileIO
using Colors, ColorSchemes
using DifferentialEquations
using DelimitedFiles
using Turing
using MAT
using Serialization
using DrWatson
using DataFrames
using Random
include(projectdir("plots/functions.jl"))

suvr_mat = matread("/Users/pavanchaggar/Projects/TauPet/tau_code_python/suvr_ADNI_scaled.mat")
const suvr = suvr_mat["ord_suvr"]
age_mat = matread("/Users/pavanchaggar/Projects/TauPet/tau_code_python/ages_ADNI.mat")
const age = age_mat["ages_mat"]
ts, t_index = zero_age(age)

ids = DataFrame(readdlm(projectdir("plots/ID_amyloid.txt")), [:ind, :id, :st])
abpos = filter(x -> x.st == 1, ids)
abpos

connectome_path = "/Users/pavanchaggar/.julia/dev/Connectomes/assets/connectomes/hcp-scale1-standard-master.graphml"
c = Connectome(connectome_path);

function NetworkFKPP(du, u, p, t; L = c.L)
    du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
end

subject = 36
hsub = 21
# Plot Dists 
chain_path = "/Users/pavanchaggar/Projects/TauPet/chains/nuts/"
pos = deserialize(chain_path * "posteriorchain_$(subject).jls");

pospath = "/Users/pavanchaggar/Projects/TauPet/pos_posterior_chain.jls"
hpos = deserialize(pospath);

f = Figure(resolution = (1600,600), fontsize = 20)
ax = Axis(f[1, 1])
plot_density!(pos, :k; color=(:green, 0.5), bins=50, label="ρ")
plot_density!(hpos, Symbol("k[$hsub]"); color=(:darkgreen, 0.9), bins=50, label="ρ - hierarchical"); 
axislegend(ax, position = :rt)
ax = Axis(f[2, 1])
plot_density!(pos, :a; color=(:blue, 0.5), bins=50, label="α")
plot_density!(hpos, Symbol("a[$hsub]"); color=(:darkblue, 0.9), bins=50, label="α - hierarchical"); 
axislegend(ax, position = :rt)
f

u0 = suvr[:,1,subject]
p = [0.05,1.0]
t_span = (0.0,30.0)

problem = ODEProblem(NetworkFKPP, u0, t_span, p);
sol = solve(problem, Tsit5(), saveat=0.1);

sols = clamp.(simulate_sub(pos, problem, u0, [27]; N=1000), 0.0, 1.0)
q2 = [quantile(sols[i,:], 0.025) for i in 1:301]
q9 = [quantile(sols[i,:], 0.975) for i in 1:301]

hsols = clamp.(simulate_sub_h(hpos, problem, u0, hsub, [27]; N=1000), 0.0, 1.0)
hq2 = [quantile(hsols[i,:], 0.025) for i in 1:301]
hq9 = [quantile(hsols[i,:], 0.975) for i in 1:301]

# Plot prior_predictive_noise
ax = Axis(f[:,2:3], xlabel = "Time / years", ylabel = "Concentration")
band!(collect(0:0.1:30), q9, q2, color=(:grey, 0.3), label="95% Intervel")
plot_predictive_mean!(pos, problem; nodes=[27], add_noise=false)

band!(collect(0:0.1:30), hq9, hq2, color=(:green, 0.3), label="95% Intervel - Hierarchical")
plot_predictive_mean_h!(hpos, problem; sub=hsub, nodes=[27], add_noise=false)
axislegend(ax, merge = true, unique = true, position=:rb)

f

save("assets/images/results/subject36-pos-v-hpos.png", f)