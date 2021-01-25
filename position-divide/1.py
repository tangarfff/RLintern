# This example shows using SolutionPartitioningSolver with FullQuboSolver.
# It reduces size of Qubo for FullQuboSolver and improves the solution of vrp.

import sys
#sys.path.insert(1, '../src')

from vrp_solvers import FullQuboSolver
#import DWaveSolvers
from input import *
import time

def decode(part_solution,solution):
    result_list =[]
    for i in solution:
        result = []
        for j in i:
            if j ==0:
                result.append(j)
            else:
                result.append(part_solution[j-1])
        result_list.append(result)
    return result_list

if __name__ == '__main__':


    # Parameters for solve function.
    only_one_const = 10000000.
    order_const = 1.
    part_solution = [4, 12, 18, 35, 38, 44, 46, 48, 50, 53, 56, 59, 60, 64]
    path = '/clean_test90603_AM.csv'
    t1=time.time()
    # Reading problem from file.
    problem = read_test(part_solution,path ,capacity = False)
    # Solving problem on SolutionPartitioningSolver.
    #solver = SolutionPartitioningSolver(problem, FullQuboSolver(problem))
    #solution = solver.solve(only_one_const, order_const, solver_type = 'cpu')
    solver = FullQuboSolver(problem)
    print("solver",solver)
    solution = solver.solve(only_one_const, order_const, solver_type='qpu')


    t2 = time.time()
    print("Solution : ", solution.solution)
    print(type(solution.solution))
    result = decode(part_solution, solution.solution)
    print(result)
    print("Total cost : ", solution.total_cost())
    print("time",t2-t1)
    print("\n")
