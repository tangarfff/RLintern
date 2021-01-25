from qubo_helper import Qubo
from vrp_problem import VRPProblem
from vrp_solution import VRPSolution
from itertools import product
import DWaveSolvers
import networkx as nx
import numpy as np
from queue import Queue

# Abstract class for VRP solvers.
class VRPSolver:
    # Attributes : VRPProblem
    def __init__(self, problem):
        self.problem = problem

    def set_problem(self, problem):
        self.problem = problem

    # only_one_const - const in qubo making solution correct
    # order_const - multiplier for costs in qubo
    # It is recommended to set order_const = 1 and only_one_const
    # big enough to make solutions correct. Bigger than sum of all
    # costs should be enough.
    def solve(self, only_one_const, order_const, solver_type = 'qpu'):
        pass

# Solver solves VRP only by QUBO formulation.
class FullQuboSolver(VRPSolver):
    def solve(self, only_one_const, order_const, solver_type = 'qpu'):
        qubo = self.problem.get_full_qubo(only_one_const, order_const)
        sample = DWaveSolvers.solve_qubo(qubo, solver_type = solver_type)
        solution = VRPSolution(self.problem, sample)
        return solution


