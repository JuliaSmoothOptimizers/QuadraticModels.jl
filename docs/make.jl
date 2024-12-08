using Documenter, QuadraticModels

makedocs(
  modules = [QuadraticModels],
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    ansicolor = true,
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "QuadraticModels.jl",
  pages = ["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/QuadraticModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
