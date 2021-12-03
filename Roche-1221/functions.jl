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

gr()
theme(:ggplot2)

function NetworkFKPP(du, u, p, t)
	du .= -p[1] * L * u .+ p[2] .* u .* (1 .- u)
end


rootpath = "/Users/pavanchaggar/Documents/ResearchDocs/Presentations/inference-methods-UCSF0721"
meshpath = "/Users/pavanchaggar/.julia/dev/Connectomes/assets/meshes/"

struct TwoColumn{A, B}
    left::A
    right::B
end

function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
    write(io,
        """
        <div style="display: flex;">
            <div style="50%;">
        """)
    show(io, mime, tc.left)
    write(io,
        """
            </div>
            <div style="50%;">
        """)
    show(io, mime, tc.right)
    write(io,
        """
            </div>
        </div>
    """)
end

function two_cols(left, right)
    @htl("""
        <style>
        div.two-cols {
            display: flex;
            width: 100%;
        }
        div.two-cols > div {
            width: 50%;
            padding: 1em;
        }
        div.two
        </style>
        <div class="two-cols">
            <div>$(left)</div>
            <div>$(right)</div>
        </div>
        """)
end

function cite(citation)
	@htl("""<p style="font-size:small"> $(citation)</p>""")
end

function pic(str::String; h=100, w=100, vspace=0, hspace=0)
	@htl("""
		<img src="$(str)?raw=true" height=$(h) width=$(w) vspace=$(vspace) hspace=$(hspace)>
			""")
end
