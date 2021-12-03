using Turing 
using GLMakie
using Serialization

pospath = "/Users/pavanchaggar/Projects/hierarchical-atrophy/chains/pos-chain-serial-4x2000.jls"
negpath = "/Users/pavanchaggar/Projects/hierarchical-atrophy/chains/neg-chain-serial-4x2000.jls"

pos = deserialize(pospath)
neg = deserialize(negpath)

f = Figure(resolution = (1600,800), fontsize = 30)
ax = Axis(f[1, 1])
GLMakie.density!(vec(pos[:Km]), normalization=:pdf, color=(:darkred,0.5), strokecolor = :darkred, strokewidth = 3, strokearound = true, label=L"A\beta^{+}")
GLMakie.density!(vec(neg[:Km]), normalization=:pdf, color=(:darkblue,0.5), strokecolor = :darkblue, strokewidth = 3, strokearound = true, label=L"A\beta^{-}")
#axislegend(ax, position=:rt)
Legend(f[1, 1], ax, patchsize=(150,75), rowgap = 10, tellheight = false, tellwidth = false, valign=:top, halign=:right, margin = (10, 10, 10, 10))
f
save("assets/images/results/hierarchical-diffusion.png", f)

f = Figure(resolution = (1600,800), fontsize = 30)
ax = Axis(f[1, 1])
GLMakie.density!(vec(pos[:Am]), normalization=:pdf, color=(:darkred,0.5), strokecolor = :darkred, strokewidth = 3, strokearound = true, label=L"A\beta^{+}")
GLMakie.density!(vec(neg[:Am]), normalization=:pdf, color=(:darkblue,0.5), strokecolor = :darkblue, strokewidth = 3, strokearound = true, label=L"A\beta^{-}")
#axislegend(ax, position=:rt)
Legend(f[1, 1], ax, patchsize=(150,75), rowgap = 10, tellheight = false, tellwidth = false, valign=:top, halign=:right, margin = (10, 10, 10, 10))
f
save("assets/images/results/hierarchical-growth.png", f)

f = Figure(resolution = (1600,800), fontsize = 30)
ax = Axis(f[1, 1])
GLMakie.density!(vec(pos[:Bm]), normalization=:pdf, color=(:darkred,0.5), strokecolor = :darkred, strokewidth = 3, strokearound = true, label=L"A\beta^{+}")
GLMakie.density!(vec(neg[:Bm]), normalization=:pdf, color=(:darkblue,0.5), strokecolor = :darkblue, strokewidth = 3, strokearound = true, label=L"A\beta^{-}")
#axislegend(ax, position=:rt)
Legend(f[1, 1], ax, patchsize=(150,75), rowgap = 10, tellheight = false, tellwidth = false, valign=:top, halign=:right, margin = (10, 10, 10, 10))
f
save("assets/images/results/hierarchical-atrophy.png", f)