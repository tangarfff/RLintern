# File simplifies communication with DWave solvers. 
# DwaveSolvers.py contains interface for our solvers to communicate with D-Wave
from dwave_qbsolv import QBSolv
import hybrid


# Creates hybrid solver with hardcoded configuration.
def hybrid_solver():
    workflow = hybrid.Loop(
        hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(size=30, rolling=True, rolling_history=0.75)
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()) | hybrid.ArgMin(), convergence=1)
    return hybrid.HybridSampler(workflow)


# Gets cpu or qpu solver.
# For qpu hybrid solver is used. For cpu qbsolv.
def get_solver(solver_type):
    solver = None
    if solver_type == 'qpu':
        solver = hybrid_solver()
        print("solver_type is QPU",solver)
    if solver_type == 'cpu':
        solver = QBSolv()
        print("solver_type is CPU")
    return solver


# Solves qubo on qpu. Returns list of solutions.
def solve_qubo(qubo, solver_type='qpu'):
    sampler = get_solver(solver_type)
    response = sampler.sample_qubo(qubo.dict)
    return list(response)[0]

