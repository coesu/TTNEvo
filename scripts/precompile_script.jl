using TTNEvo
using NamedGraphs
using Graphs
using ITensors
using ITensorNetworks: siteinds, ttn, tdvp
using Random
using MKL

function precompile_main()
  maxdim = 10
  L = 4
  config = TreeConfig(
    TimeEvolutionConfig(t_range=(0, 0.1), t_step=0.1, maxdim=maxdim, nsite=1, cutoff=1e-12, qns=false, method=:onesite),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => 12, :qns => false)),
    "testing",
    HeisenbergModel(J1=1, JZ=1, h=50.0),
    FreeGraph(L=L, gridnum=1, with_ancilla=false, full_interaction=true, optimize_structure=false, optimize_bonddim=true),
    Observer()
  )
  TTNEvo.run_simulation(config)
  config = TreeConfig(
    TimeEvolutionConfig(t_range=(0, 0.1), t_step=0.1, maxdim=maxdim, nsite=1, cutoff=1e-12, qns=false, method=:onesite),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => 12, :qns => false)),
    "testing",
    HeisenbergModel(J1=1, JZ=1, h=50.0),
    SnakeGraph(4, 1, false, true),
    Observer()
  )
  TTNEvo.run_simulation(config)
  maxdim = 4
  config = PepsConfig(
    TimeEvolutionConfig((0.0, 0.1), 0.1, maxdim, 1, 1e-14, false, :peps, nothing, nothing),
    InitialStateConfig(:columnar_neel, Dict(:initial_maxdim => maxdim, :qns => false)),
    "test_peps",
    HeisenbergModel(1, 1, 10),
    PepsGraph(4, 1),
    PepsObserver(),
  )
  TTNEvo.peps_simulation(config)
end

precompile_main()
