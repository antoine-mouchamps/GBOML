# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).


"""Solver API contains the API and option files of every solver supported.

All the APIs work similarly, they take the matrices and vectors of the problem

    min : c^T * X + X^T * Q_obj * X
    s.t. (A * X)_r + X^T * Q_r * X  <=  b_r,   for every row r

and pass it to the corresponding solver.

  Args:
    A_eq -> coo_matrix of equality constraints
    b_eq -> np.ndarray of independent terms of each equality constraint
    Q_eq -> coo_array of the degree two part of the equalities, over the
            rows of A_eq
    A_ineq -> coo_matrix of inequality constraints
    b_ineq -> np.ndarray of independent terms of each inequality constraint
    Q_ineq -> coo_array of the degree two part of the inequalities, over the
              rows of A_ineq
    c -> np.ndarray of objective vector
    objective_offset -> float of the objective offset
    Q_obj -> coo_array of the degree two part of the objective, over the
             rows of c
    name_tuples -> dictionary of <node_name variables> used to get the type

  Returns:
    solution -> np.ndarray of the flat solution
    objective -> float of the objective value
    status -> solution status
    solver_info -> dictionary of solver information

  Typical usage example:

   solution, objective, status, solver_info = api_solver(
       matrix_a_eq, vector_b_eq, quadratic_eq,
       matrix_a_ineq, vector_b_ineq, quadratic_ineq,
       vector_c, objective_offset, quadratic_objective,
       name_tuples)
   print("the solution is "+str(solution))
   print("the objective found : "+str(objective))
"""

import sys

from .gurobi_solver import gurobi_solver
from .clp_solver import clp_solver
from .cplex_solver import cplex_solver
from .xpress_solver import xpress_solver
from .scipy_solver import scipy_solver
from .dsppy import DSPpy
from .dsp_solver import dsp_solver
from .pyhighs import PyHighs
from .highs_solver import highs_solver
from .cbc_solver import cbc_solver
from .pycbc import PyCBC


def supports_quadratic(solver_function) -> bool:
    """
    supports_quadratic function : whether a solver API accepts the constraints
    and objectives holding a product of two variables. Each API declares it for
    itself, with a SUPPORTS_QUADRATIC flag in its own module, so that no caller
    ever has to name a solver of its own.
    INPUT:  solver_function -> function calling a solver, or None
    OUTPUT: bool
    """

    if solver_function is None:
        return False

    module = sys.modules.get(solver_function.__module__)

    return getattr(module, "SUPPORTS_QUADRATIC", False)


__all__ = [
    "gurobi_solver",
    "clp_solver",
    "cplex_solver",
    "xpress_solver",
    "scipy_solver",
    "DSPpy",
    "dsp_solver",
    "PyHighs",
    "highs_solver",
    "cbc_solver",
    "PyCBC",
    "supports_quadratic",
]
