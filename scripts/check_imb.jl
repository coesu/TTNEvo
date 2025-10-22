
function peak_finder(arr, thresh)
  for i in eachindex(arr)
  end
end

function check_imb(dir, L, h, grid)
  entries = readdir(dir)
  single = first(filter(x -> contains(x, "L=$(L)") && contains(x, "h=$(h)") && contains(x, "gridnum=$(grid)") && contains(x, "D=$(64)"), entries))

  full_path = joinpath(dir, single)
  d = load(full_path)["results"]


  df = DataFrame()
  flat_dict = merge(d["parameters"], d["observables"])
  push!(df, flat_dict, cols=:union)
  x = only(df)

  tree_opt = [y[2]["tree_opt"] for y in x.errors]
  adapt_bond = [y[2]["adapt_bond_dim"] for y in x.errors]

  fig = Figure()
  ax = Axis(fig[1, 1])
  lines!(ax, x.times[2:end], tree_opt)
  lines!(ax, x.times[2:end], adapt_bond)
  display(fig)

  @show x.num_size

  imbalance = TTNEvo.columnar_imbalance_total(x.sz)
  fig = Figure()
  ax = Axis(fig[1, 1])
  lines!(ax, x.times, imbalance)
  display(fig)

  fig = Figure()
  ax = Axis(fig[1, 1])
  for i in 1:L
    for j in 1:L
      sz = [z[i, j] for z in x.sz]
      lines!(ax, x.times[4:end], abs.(diff(diff(diff(sz)))))
    end
  end

  display(fig)
end
