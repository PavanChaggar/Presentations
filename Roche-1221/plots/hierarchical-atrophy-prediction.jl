using Connectomes
using GLMakie, FileIO
using Colors, ColorSchemes
using DifferentialEquations
using DelimitedFiles, NPZ
using Turing
using MAT
using Serialization
using DrWatson
using DataFrames
using Random
include(projectdir("plots/functions.jl"))

dir = "/Users/pavanchaggar/Projects/hierarchical-atrophy/"
const L = readdlm(joinpath(dir, "connectomes/master-std33-L.csv"))

f64(x::AbstractArray) = convert(Array{Float64}, x)

data = npzread(joinpath(dir, "data/tau_atr_pos.npy")) |> f64
datamask = map(x -> isnan(x) ? x = 0 : x = 1, data)
const scans = sum(datamask[1,:,:], dims=1) |> vec

const N = length(scans)

tau_times = npzread(joinpath(dir, "data/times_tau_pos.npy")) |> transpose |> Array |> f64
const times = tau_times .* datamask[1,:,:]

# get cortical node index
cortical_nodes = (npzread(joinpath(dir, "data/sel_nodes.npy")) |> f64 .|> Int) .+ 1
tau_nodes = cortical_nodes[1:68]
atr_nodes = cortical_nodes[69:end]
tau_data = data[tau_nodes, :, :]
atr_data = data[atr_nodes, :, :]

const vec_tau_data = reshape(tau_data, (68*5, N))
const vec_atr_data = reshape(atr_data, (83*5, N))
const initial_conditions = data[:, 1, :]

function NetworkAtrophy(du, u0, p, t)
    x = u0[1:83] # hardcoded seperation of state vectors since we're not working with different graphs 
    y = u0[84:166]

    du[1:83] .= -p[1] * L * x .+ p[2] .* x .* (1.0 .- x) #fkpp 
    du[84:166] .= p[3] * x .* (1.0 .- y) #atrophy
end

u0 = initial_conditions[:,16]
p = [1.0, 1.0, 1.0]
t_span = (0.0,30.0)

problem = ODEProblem(NetworkAtrophy, u0, t_span, p);
sol = solve(problem, Tsit5(), saveat=0.1);

hsub = 16

pospath = joinpath(dir, "chains/pos-chain-indn-serial-4x2000.jls")
hpos = deserialize(pospath);

f = Figure(resolution = (1600,600), fontsize = 20)
ax = Axis(f[1:2, 1])
plot_density!(hpos, Symbol("k[$hsub]"); color=(:darkgreen, 0.5), bins=50, label=L"\rho"); 
axislegend(ax, position = :rt)
ax = Axis(f[3:4, 1])
plot_density!(hpos, Symbol("a[$hsub]"); color=(:darkblue, 0.5), bins=50, label=L"\alpha"); 
axislegend(ax, position = :rt)
ax = Axis(f[5:6, 1])
plot_density!(hpos, Symbol("b[$hsub]"); color=(:red, 0.5), bins=50, label=L"\beta"); 
axislegend(ax, position = :rt)


tausols, atrsols = simulate_sub_a(hpos, problem, u0, 16, [81]; N=1000)
tau = clamp.(tausols, 0.0,1.0)
atr = atrsols

tq2 = [quantile(tau[i,:], 0.025) for i in 1:301]
tq9 = [quantile(tau[i,:], 0.975) for i in 1:301]

aq2 = [quantile(atr[i,:], 0.025) for i in 1:301]
aq9 = [quantile(atr[i,:], 0.975) for i in 1:301]

# Plot prior_predictive_noise
ax = Axis(f[1:3,2:3], xlabel = "Time / years", ylabel = "Concentration")
plot_data!([0.0,1.0,2.0], data[1:83,1:3,16]; markersize=10.0, color=(:lightgrey,0.8), label="PET Data")
plot_data!([0.0,1.0,2.0], data[1:83,1:3,16], 81; markersize=10.0, color=(:red,0.8), label="HC - PET Data")
band!(collect(0:0.1:30), tq9, tq2, color=(:grey, 0.3), label="95% Interval")
plot_predictive_mean_a!(hpos, problem; sub=16, nodes=[81], add_noise=false)
axislegend(ax, merge = true, unique = true, position=:rb)

ax = Axis(f[4:6,2:3], xlabel = "Time / years", ylabel = "Atrophy")
plot_data!([0.0,1.0,2.0], data[84:end,1:3,16]; markersize=10.0, color=(:lightgrey,0.8), label="Atr Data")
plot_data!([0.0,1.0,2.0], data[84:end,1:3,16], 81; markersize=10.0, color=(:red,0.8), label="HC - Atr Data")
band!(collect(0:0.1:30), aq9, aq2, color=(:green, 0.3), label="95% Interval")
plot_predictive_mean_a!(hpos, problem; sub=16, nodes=[81+83], add_noise=false, color=(:blue, 0.5))
axislegend(ax, merge = true, unique = true, position=:lt)
f

save("assets/images/results/subject36-hpos-atr.png", f)