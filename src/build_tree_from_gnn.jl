using NamedGraphs
using Graphs


function build_tree_from_edges(L, edges)
  g = NamedGraph(Graph(L^2), Tuple.(CartesianIndices((L, L))))
  @show g
end
