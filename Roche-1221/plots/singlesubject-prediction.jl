using Connectomes
using GLMakie, FileIO
using Colors, ColorSchemes
using DifferentialEquations
using DelimitedFiles
using Turing
using MAT
using Serialization
using Random
include("functions.jl")

suvr_mat = matread("/Users/pavanchaggar/Projects/TauPet/tau_code_python/suvr_ADNI_scaled.mat")
const suvr = suvr_mat["ord_suvr"]
age_mat = matread("/Users/pavanchaggar/Projects/TauPet/tau_code_python/ages_ADNI.mat")
const age = age_mat["ages_mat"]
ts, t_index = zero_age(age)

connectome_path = "/Users/pavanchaggar/.julia/dev/Connectomes/assets/connectomes/hcp-scale1-standard-master.graphml"
c = Connectome(connectome_path)
fc =  graph_filter(c, 0.001)

function NetworkFKPP(du, u, p, t; L = fc.L)
    du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
end

u0 = suvr[:,1,36]
p = [0.05,1.0]
t_span = (0.0,30.0)

problem = ODEProblem(NetworkFKPP, u0, t_span, p);
sol = solve(problem, Tsit5(), saveat=0.1);

# Plot Dists 
chain_path = "/Users/pavanchaggar/Projects/TauPet/chains/"
p1 = deserialize(chain_path * "nuts/posteriorchain_36.jls");
p2 = deserialize(chain_path * "nutsfull/posterior_36.jls");

f = Figure(resolution = (1600,600), fontsize = 20)
ax = Axis(f[1, 1])
plot_density!(p1, :k; color=(:green, 0.5), bins=50, label="ρ")
plot_density!(p2, :k; color=(:darkgreen, 0.9), bins=50, label="ρ - full")
axislegend(ax, position = :rt)
ax = Axis(f[2, 1])
plot_density!(p1, :a; color=(:blue, 0.5), bins=50, label="α")
plot_density!(p2, :a; color=(:darkblue, 0.9), bins=50, label="α - full")
axislegend(ax, position = :rt)


sols = clamp.(simulate_sub(p1, problem, u0, [27]; N=1000), 0.0, 1.0)
q2 = [quantile(sols[i,:], 0.025) for i in 1:301]
q9 = [quantile(sols[i,:], 0.975) for i in 1:301]


fsols = clamp.(simulate_sub(p2, problem, u0, [27]; N=1000), 0.0, 1.0)
fq2 = [quantile(fsols[i,:], 0.025) for i in 1:301]
fq9 = [quantile(fsols[i,:], 0.975) for i in 1:301]

# Plot prior_predictive_noise
ax = Axis(f[1:2,2:3], xlabel = "Time / years", ylabel = "Concentration")
plot_data!([0.0,1.0,2.0], suvr[:,1:3,36]; markersize=10.0, color=(:lightgrey,0.8), label="Data")
plot_data!([0.0,1.0,2.0], suvr[:,1:3,36], 27; markersize=10.0, color=(:red,0.8), label="EC - Data")

band!(collect(0:0.1:30), q9, q2, color=(:grey, 0.2), label="95% Interval")
plot_predictive_mean!(p1, problem; nodes=[27], add_noise=false)

band!(collect(0:0.1:30), fq9, fq2, color=(:green, 0.3), label="95% Interval -- full")
plot_predictive_mean!(p2, problem; nodes=[27], add_noise=false, color=(:blue, 0.5), label="Mean Predicted - full")
axislegend(ax, merge = true, unique = true, position=:lt)

f

save("assets/images/results/subject-36-fullposterior.png", f)