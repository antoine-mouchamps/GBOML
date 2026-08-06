# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).


"""Gurobi Solver file, contains the interface to Gurobi solver .

Takes the matrices and vectors of the problem

    min : c^T * X + X^T * Q_obj * X
    s.t. (A * X)_r + X^T * Q_r * X  <=  b_r,   for every row r

where Q_r is the degree two part of row r, handed over alongside the degree
one part it shares its rows with,
and passes it to the gurobi solver.

  Typical usage example:

   solution, objective, status, solver_info = gurobi_solver(matrix_a, vector_b,
                                                            vector_c,
                                                            objective_offset,
                                                            name_tuples)
   print("the solution is "+str(solution))
   print("the objective found : "+str(objective))
"""

import numpy as np
from scipy.sparse import coo_matrix
from gboml.compiler.utils import flat_nested_list_to_two_level, read_attributes_in_file


SUPPORTS_QUADRATIC = True


def _split_rows_by_degree(quadratic_terms, nb_rows) -> tuple:
    """_split_rows_by_degree

    the rows of a block which hold a product of two variables, and the rows
    which do not, as two masks.

    Args:
        quadratic_terms -> coo_array of the degree two part, or None
        nb_rows -> number of rows of the block

    Returns:
        two ndarrays of bool, one entry per row each

    """

    with_a_product = np.zeros(nb_rows, dtype=bool)
    if quadratic_terms is not None and quadratic_terms.nnz:
        rows, _, _ = quadratic_terms.coords
        with_a_product[rows] = True

    return with_a_product, np.logical_not(with_a_product)


def _add_constraint_block(model, x, matrix_a, vector_b, sense, quadratic_terms) -> None:
    """_add_constraint_block

    adds one block of constraints to the model. The rows holding no product
    of two variables are handed over as a whole matrix, in a single call; the
    ones holding a product are added one by one.

    Args:
        model -> gurobi model
        x -> gurobi MVar of every variable of the problem
        matrix_a -> coo_matrix of the degree one part of the block
        vector_b -> np.ndarray of the right hand side of the block
        sense -> gurobi sense of the block
        quadratic_terms -> coo_array of the degree two part, or None

    """

    import gurobipy as grbp

    nb_rows = len(vector_b)
    matrix_csr = matrix_a.tocsr()
    with_a_product, without_a_product = _split_rows_by_degree(quadratic_terms, nb_rows)

    if without_a_product.any():
        model.addMConstr(
            matrix_csr[without_a_product], x, sense, vector_b[without_a_product]
        )

    if not with_a_product.any():
        return

    variables = x.tolist()

    rows, columns_i, columns_j = quadratic_terms.coords
    values = quadratic_terms.data
    order = np.argsort(rows, kind="stable")
    rows, values = rows[order], values[order]
    columns_i, columns_j = columns_i[order], columns_j[order]
    starts = np.searchsorted(rows, np.arange(nb_rows + 1))

    for row in np.flatnonzero(with_a_product):
        start, end = starts[row], starts[row + 1]
        expression = grbp.QuadExpr()
        expression.addTerms(
            values[start:end].tolist(),
            [variables[column] for column in columns_i[start:end]],
            [variables[column] for column in columns_j[start:end]],
        )

        start, end = matrix_csr.indptr[row], matrix_csr.indptr[row + 1]
        if end > start:
            expression.addTerms(
                matrix_csr.data[start:end].tolist(),
                [variables[column] for column in matrix_csr.indices[start:end]],
            )

        model.addQConstr(expression, sense, float(vector_b[row]))


def gurobi_solver(
    matrix_a_eq: coo_matrix,
    vector_b_eq: np.ndarray,
    quadratic_eq,
    matrix_a_ineq: coo_matrix,
    vector_b_ineq: np.ndarray,
    quadratic_ineq,
    vector_c: np.ndarray,
    objective_offset: float,
    quadratic_objective,
    name_tuples: dict,
    opt_file: str = None,
    details=False,
) -> tuple:
    """gurobi_solver

    takes as input the matrices and vectors of the problem stated at
    the top of this file. It returns the solution found by the gurobi solver

    Args:
        matrix_a_eq -> coo_matrix of equality constraints
        vector_b_eq -> np.ndarray of independent terms of each equality constraint
        matrix_a_ineq -> coo_matrix of inequality constraints
        vector_b_ineq -> np.ndarray of independent terms of each inequality constraint
        vector_c -> np.ndarray of objective vector
        objective_offset -> float of the objective offset
        name_tuples -> dictionary of <node_name variables> used to get
                       the type
        opt_file -> optimization parameters file
        details -> tuple of bool and path to attributes to retrieve
        quadratic_eq -> coo_array of the degree two part of the
                        equalities, over the rows of matrix_a_eq
        quadratic_ineq -> same, over the rows of matrix_a_ineq
        quadratic_objective -> same, over the rows of the objective

    Returns:
        solution -> np.ndarray of the flat solution
        objective -> float of the objective value
        status -> solution status
        solver_info -> dictionary of solver information
        constraints_additional_information -> dict of additional
                                              information concerning
                                              constraints
        variables_additional_information -> dict of additional information
                                            concerning variables

    """

    try:
        import gurobipy as grbp
        from gurobipy import GRB

    except ImportError:
        print("Warning: Did not find the gurobipy package")
        exit(0)

    if opt_file is None:
        opt_file = 'src/gboml/solver_api/gurobi.opt'

    solution = None
    objective = None

    # Fix shapes and types of input
    matrix_a_eq = matrix_a_eq.astype(float)
    matrix_a_ineq = matrix_a_ineq.astype(float)
    nb_eq_constr, _ = matrix_a_eq.shape
    nb_ineq_constr, _ = matrix_a_ineq.shape

    _, n = np.shape(vector_c)
    b_eq = vector_b_eq.reshape(-1)
    b_ineq = vector_b_ineq.reshape(-1)
    vector_c = vector_c.reshape(-1)

    # the rows Gurobi holds as linear constraints, which is what the attributes
    # are read back over further down
    _, linear_ineq_rows = _split_rows_by_degree(quadratic_ineq, nb_ineq_constr)
    _, linear_eq_rows = _split_rows_by_degree(quadratic_eq, nb_eq_constr)
    nb_linear_ineq = int(linear_ineq_rows.sum())

    # Build Gurobi model
    model = grbp.Model()
    x = model.addMVar(shape=n, lb=-float('inf'), ub=float('inf'),
                      vtype=GRB.CONTINUOUS, name="x")

    _add_constraint_block(model, x, matrix_a_ineq, b_ineq, "<", quadratic_ineq)
    _add_constraint_block(model, x, matrix_a_eq, b_eq, "=", quadratic_eq)

    quadratic_part = 0.0
    if quadratic_objective.nnz:
        _, columns_i, columns_j = quadratic_objective.coords
        quadratic_part = (
            x
            @ coo_matrix(
                (quadratic_objective.data, (columns_i, columns_j)), shape=(n, n)
            )
            @ x
        )

    model.setObjective(quadratic_part + vector_c @ x + objective_offset, GRB.MINIMIZE)

    flat_name_tuples = flat_nested_list_to_two_level(name_tuples)

    for index, _, var_type, var_size in flat_name_tuples:

        if var_type == "integer":
            x[index:index + var_size].vtype = GRB.INTEGER
        if var_type == "binary":
            x[index:index + var_size].vtype = GRB.BINARY

    # Gather and retrieve solver information
    solver_info = dict()
    solver_info["name"] = "gurobi"
    print("\nReading Gurobi options from file gurobi.opt")
    option_info = {}

    try:

        with open(opt_file, 'r') as optfile:
            lines = optfile.readlines()
    except IOError:

        print("Options file not found")
    else:

        for line in lines:

            line = line.strip()
            option = line.split(" ", 1)
            if option[0] != "":

                parinfo = model.getParamInfo(option[0])
                if parinfo:

                    if len(option) == 2:

                        key = option[0]
                        try:

                            value = parinfo[1](option[1])
                        except ValueError as e:

                            print("Skipping option \'%s\' with invalid "
                                  "given value \'%s\' (expected %s)"
                                  % (option[0], option[1], parinfo[1]))
                        else:

                            model.setParam(key, value)
                            option_info[key] = value
                    else:

                        print("Skipping option \'%s\' with no given value"
                              % option[0])
                else:

                    print("Skipping unknown option \'%s\'" % option[0])

    solver_info["options"] = option_info
    print("")

    # Solve model and return solution
    try:
        model.optimize()
        status_code = model.getAttr("Status")
        solver_info["status"] = status_code
        solution = None
        objective = None
        if status_code == 2:

            status = "optimal"
            solution = x.X
            objective = model.getObjective().getValue()
        elif status_code == 3:

            status = "infeasible"
            objective = float('inf')
        elif status_code == 5:

            status = "unbounded"
            objective = float('-inf')
        elif status_code == 13:

            status = "feasible"
            solution = x.X
            objective = model.getObjective().getValue()
        else:

            status = "unknown"
    except RuntimeError as e:

        print(e)
        status = "error"

    constraints_additional_info_eq = dict()
    constraints_additional_info_ineq = dict()
    variables_additional_information = dict()
    if details is not False:
        if isinstance(details, str):
            attributes_to_retrieve = read_attributes_in_file(details)
        else:
            attributes_to_retrieve = ["Pi", "Slack",
                                      "CBasis", "SARHSLow", "SARHSUp",
                                      "RC", "VBasis", "SAObjLow", "SAObjUp",
                                      "SALBLow", "SALBUp", "SAUBLow",
                                      "SAUBUp"]

        def _scatter_over_rows(values, linear_rows) -> list:
            """puts the attributes Gurobi reports back at the row of the block they
            belong to"""

            scattered = [None] * len(linear_rows)
            for position, row in enumerate(np.flatnonzero(linear_rows)):
                scattered[row] = values[position]

            return scattered

        for attribute in attributes_to_retrieve:
            try:
                additional_constr_info = \
                    model.getAttr(attribute, model.getConstrs())

                constraints_additional_info_ineq[attribute] = _scatter_over_rows(
                    additional_constr_info[:nb_linear_ineq], linear_ineq_rows
                )
                constraints_additional_info_eq[attribute] = _scatter_over_rows(
                    additional_constr_info[nb_linear_ineq:], linear_eq_rows
                )
            except (grbp.GurobiError, AttributeError):
                try:
                    variables_additional_information[attribute] = \
                        model.getAttr(attribute, model.getVars())
                except (grbp.GurobiError, AttributeError):
                    print("Warning : Unable to retrieve ", attribute,
                          " information")

    constraints_additional_information = {"eq": constraints_additional_info_eq,
                                          "ineq": constraints_additional_info_ineq}

    return solution, objective, status, solver_info, \
           constraints_additional_information, \
           variables_additional_information
