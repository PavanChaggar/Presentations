	
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

function plot_image(path::String)
    img_data = load(path)
    Plots.plot(img_data, showaxis=:hide, ticks=false)
end