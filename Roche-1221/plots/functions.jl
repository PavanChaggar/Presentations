function plot_predictive(chain, prob, sol, data, t; N=200, node=27) 
	u = Array(prior_chain[vars[2:84]])
    plot(sol, vars=(node), w=2, labels = false)
    for i in 1:N
        resol = solve(remake(prob, u0=u[i,:],
							 p=[chain[:k][i], chain[:a][i]]),
							 Tsit5())
        plot!(resol, vars=(node), alpha=0.1, color=:grey,labels=false)
	end
    #scatter!([t], data[node]', legend = false)
    plot!()
end

function edges!(connectome::Connectome, color)
    x, y, z = connectome.parc.x[:], connectome.parc.y[:], connectome.parc.z[:]
    coordindex = findall(x->x>0, LowerTriangular(connectome.A))
    
    for i ∈ 1:length(coordindex)
        j, k = coordindex[i][1], coordindex[i][2]
        weight = connectome.A[j, k]
        lines!(x[[j,k]], y[[j,k]], z[[j,k]],
               color=Colors.alphacolor(get(color, weight), clamp(weight, 0.2,1.0)), #matter
               linewidth=clamp(10*weight,2,10),
               transparency=true)
    end
end

degree_norm(C::Connectome) = LightGraphs.degree(C) |> max_norm

function plot_sol!(sol, regions, color, linewidth=5, label="label")
    for i in regions    
        lines!(sol.t, sol[i,:], color = color, linewidth=linewidth, label=label)
    end
end


function plot_sol_t!(t, sol, regions, color, linewidth=5, label="label")
    for i in regions    
        lines!(t, sol[i,:], color = color, linewidth=linewidth, label=label)
    end
end

function plot_avg_sol!(sol, regions, color, linewidth=5, label="label")
    lines!(sol.t, vec(mean(sol[regions,:], dims=1)), color = color, linewidth=linewidth, label=label)
end

function get_col(color, weight)
    Colors.alphacolor.(color, weight)
end

function vertex!(connectome::Connectome, node_size, color, node)
    x, y, z = connectome.parc.x[node], connectome.parc.y[node], connectome.parc.z[node]
    meshscatter!(x, y, z, markersize=node_size, color=color)
end

function make_bmap(cmap, bs)
    bmap = Dict()
    for (i, b) in enumerate(bs)
        [push!(bmap, b[k] => cmap[i]) for k in 1:length(b)]
    end
    bmap
end

function plot_data!(t, data, regions=1:83; kwargs...)
    for i in regions
        scatter!(t, Array(data)[i,:]; kwargs...)
    end
end

function plot_sol!(sol; kwargs...)
    for i in 1:83    
        lines!(sol.t, sol[i,:]; kwargs...)
    end
end

function plot_predictive!(chain, prob, data, data_t, n=100; nodes=[27], plot_true=true, add_noise=true)
    sol =  solve(prob, Tsit5(), saveat=0.1)
    for i in 1:n    
        κ, α, σ = chain[:k][i], chain[:a][i], chain[:σ][i]
        prob = remake(prob, p = [κ, α])
        sol2 = solve(prob, Tsit5(), saveat=0.1)
        if add_noise
            soln = clamp.(sol2 .+ rand(Normal(0.0, σ), size(sol2)), 0.0,1.0)
            plot_sol_t!(sol2.t, soln, nodes, (:grey, 0.1), 2.0, "Predicted")
        else
            plot_sol!(sol2, nodes, (:grey, 0.1), 2.0, "Predicted")
        end
    end
    if plot_true
        plot_sol!(sol, nodes, (:red, 0.5), 3.0, "True")
    end
    plot_data!(data_t, data, nodes; markersize=10.0, color=(:red,0.8), label="Generated Data")
end

function plot_predictive_mean!(chain, prob; nodes=[27], add_noise=true, color=(:red, 0.5), label="Mean Predicted")
    means = mean(chain)
    κ, α, σ = means[:k, :mean], means[:a, :mean], means[:σ, :mean]
    prob = remake(prob, p = [κ, α])
    sol2 = solve(prob, Tsit5(), saveat=0.1)
    if add_noise
        soln = clamp.(sol2 .+ rand(Normal(0.0, σ), size(sol2)), 0.0,1.0)
        plot_sol_t!(sol2.t, soln, nodes, color, 2.0, label)
    else
        plot_sol!(sol2, nodes, color, 2.0, label)
    end
end

function plot_predictive_mean_h!(chain, prob; sub=1, nodes=[27], add_noise=true)
    means = mean(chain)
    κ, α, σ = means[Symbol("k[$sub]"), :mean], means[Symbol("a[$sub]"), :mean], means[:σ, :mean]
    prob = remake(prob, p = [κ, α])
    sol2 = solve(prob, Tsit5(), saveat=0.1)
    if add_noise
        soln = clamp.(sol2 .+ rand(Normal(0.0, σ), size(sol2)), 0.0,1.0)
        plot_sol_t!(sol2.t, soln, nodes, (:blue, 0.5), 2.0, "Mean Predicted - Hierarchical")
    else
        plot_sol!(sol2, nodes, (:blue, 0.5), 2.0, "Mean Predicted - Hierarchical")
    end
end

function simulate_sub(c, prob, init, nodes; N=10, samples=1000)
    σ = c[:σ]

    ρ, α = c[:k], c[:a]
    prob1 = remake(prob, u0=init, p = [ρ[1], α[1]])
    sol = solve(prob1, Tsit5(), saveat=0.1)
    sols = Array{Float64}(undef, size(sol)[1], size(sol)[2], N)
    for (i, j) in enumerate(shuffle(rand(1:samples, N)))
        prob1 = remake(prob, u0=init, p = [ρ[j], α[j]])
        sol = Array(solve(prob1, Tsit5(), saveat=0.1))
        sols[:,:,i] .= (sol .+ randn(size(sol)) .* σ[j])
    end
    meanval = mean(sols[nodes, :, :] , dims=1)
    dropdims(meanval, dims=1)
end

function simulate_sub_h(c, prob, init, sub, nodes; N=10)
    σ = c[:σ]

    ρ, α = c[Symbol("k[$sub]")], c[Symbol("a[$sub]")]
    prob1 = remake(prob, u0=init, p = [ρ[1], α[1]])
    sol = solve(prob1, Tsit5(), saveat=0.1)
    sols = Array{Float64}(undef, size(sol)[1], size(sol)[2], N)
    for (i, j) in enumerate(shuffle(rand(1:2000, N)))
        prob1 = remake(prob, u0=init, p = [ρ[j], α[j]])
        sol = Array(solve(prob1, Tsit5(), saveat=0.1))
        sols[:,:,i] .= (sol .+ randn(size(sol)) .* σ[j])
    end
    meanval = mean(sols[nodes, :, :] , dims=1)
    dropdims(meanval, dims=1)
end

function plot_predictive_data!(chain, prob, data, data_t, n=100; nodes=[27], plot_true=true, add_noise=true)
    plot_data!(data_t, data; markersize=10.0, color=(:lightgrey,0.8), label="Data")
    sols = Array{Float64}(undef, 83, 301, n)

    for i in 1:n    
        κ, α, σ = chain[:k][i], chain[:a][i], chain[:σ][i]
        prob = remake(prob, p = [κ, α])
        sol2 = solve(prob, Tsit5(), saveat=0.1)
        if add_noise
            soln = clamp.(sol2 .+ rand(Normal(0.0, σ), size(sol2)), 0.0,1.0)
            plot_sol_t!(sol2.t, soln, nodes, (:grey, 0.1), 2.0, "Predicted")
        else
            plot_sol!(sol2, nodes, (:grey, 0.1), 2.0, "Predicted")
        end
    end
    if plot_true
        plot_sol!(sol, nodes, (:red, 0.5), 3.0, "True")
    end
    plot_predictive_mean!(chain, prob)
    plot_data!(data_t, data, nodes; markersize=10.0, color=(:red,0.8), label="right EC Data")
end

function plot_predictive_data_hierarchical!(chain, prob, data, data_t, n=100; sub=1, nodes=[27], plot_true=true, add_noise=true)
    for i in 1:n    
        κ, α, σ = chain[Symbol("k[$(sub)]")][i], chain[Symbol("a[$(sub)]")][i], chain[:σ_t][i]
        prob = remake(prob, p = [κ, α])
        sol2 = solve(prob, Tsit5(), saveat=0.1)
        if add_noise
            soln = clamp.(sol2 .+ rand(Normal(0.0, σ), size(sol2)), 0.0,1.0)
            plot_sol_t!(sol2.t, soln, nodes, (:blue, 0.1), 2.0, "Predicted")
        else
            plot_sol!(sol2, nodes, (:blue, 0.1), 2.0, "Predicted")
        end
    end
    if plot_true
        plot_sol!(sol, nodes, (:red, 0.5), 3.0, "True")
    end
    plot_predictive_mean!(chain, prob)
end

function get_u0(chain, i)
    u0 = Vector{Float64}(undef, 83)
    [u0[j] = chain[Symbol("u0[$j]")][i] for j in 1:83]
end

function plot_density!(chain, key; kwargs...)
    GLMakie.hist!(vec(chain[key]), normalization=:pdf; kwargs...)
end

function zero_age(age)
    t = Matrix{Float64}(undef, size(age))
    for i in 1:size(age)[1]
        min = minimum(filter(x -> x > 0, age[i, :]))
        t[i, :] .= age[i,:] .- min
    end 
    clamp!(t, 0, 10)
    return transpose(t), transpose(age .> 0)
end