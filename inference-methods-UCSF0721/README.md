# Presentation to Raj Group, UCSF, July 2021

Presentation will be on inference methods for neurodegeneration, covering variational inference and mcmc methods (Hamiltonian Monte Carlo). 

# Running the presentation 

The presentation is in the form of a [Pluto](https://github.com/fonsp/Pluto.jl) notebook. To run the notebook using the latest version of [Julia](https://github.com/JuliaLang/julia) and Pluto do the following: 

1. Launch a Julia repl in the presentation directory 
2. Enter package mode  `]` and create an environment using `activate .`
3. Reminaing in package mode add Pluto to the environment using `add Pluto`
4. Return to the Julia repl and launch Pluto with `using Pluto; Pluto.run()`
5. In the browser, open `presentation.jl`

If you are using the latest version of Pluto, the Pluto pkg manager will install the necessary dependencies and run the notebook.
