# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

from .classes import Program
from .utils import error_, get_only_objects_in_nested_dict_layer

import numpy as np  # type: ignore
from scipy.sparse import coo_array, coo_matrix  # type: ignore
import multiprocessing as mp


def has_quadratic_terms(quadratic_eq, quadratic_ineq, quadratic_objective) -> bool:
    """
    has_quadratic_terms function: whether a product of two variables survived
    the extension, a constraint whose quadratic terms all cancel out being a
    linear one
    INPUT:  quadratic_eq, quadratic_ineq, quadratic_objective -> coo_array of
                                                                 each block
    OUTPUT: bool
    """

    return any(
        quadratic_terms is not None and quadratic_terms.nnz > 0
        for quadratic_terms in [quadratic_eq, quadratic_ineq, quadratic_objective]
    )


def flatten_definitions_and_get_values(definitions):
    flatten_dict = {}
    for key, value in definitions.items():
        if isinstance(value, dict):
            for sub_key, sub_param in value.items():
                flatten_key = key+"."+sub_key
                if sub_param.get_type() == "expression":
                    flatten_dict[flatten_key] = sub_param.get_value()[0]
                else:
                    flatten_dict[flatten_key] = np.array(sub_param.get_value())
        else:
            if value.get_type() == "expression":
                flatten_dict[key] = value.get_value()[0]
            else:
                flatten_dict[key] = np.array(value.get_value())
    return flatten_dict


def get_flat_nodes_edges_ordered(nodes, hyperedges, without_hyperedges=False):
    all_nodes_edges = []
    for node in nodes:
        all_nodes_edges.append(node)
        sub_nodes = node.get_sub_nodes()
        sub_edges = node.get_sub_hyperedges()
        all_sub_nodes_edges = get_flat_nodes_edges_ordered(sub_nodes,
                                                           sub_edges,
                                                           without_hyperedges)
        all_nodes_edges += all_sub_nodes_edges

    if not without_hyperedges:
        all_nodes_edges += hyperedges

    return all_nodes_edges


def get_flat_nodes_edges_and_definition(nodes, hyperedges,
                                        definitions, accumulator_dict):
    node_definition_tuples = []
    edge_definition_tuples = []
    for node in nodes:
        node_name = node.get_name()
        sub_nodes = node.get_sub_nodes()
        sub_edges = node.get_sub_hyperedges()
        accumulator_dict[node_name] = \
            get_only_objects_in_nested_dict_layer(definitions[node_name])
        node_definition_tuples.append([node, accumulator_dict.copy()])
        sub_node_tuples, sub_edge_tuples = \
            get_flat_nodes_edges_and_definition(sub_nodes, sub_edges,
                                                definitions[node_name],
                                                accumulator_dict)
        node_definition_tuples += sub_node_tuples
        edge_definition_tuples += sub_edge_tuples
        accumulator_dict.pop(node_name)

    for edge in hyperedges:
        edge_name = edge.get_name()
        accumulator_dict[edge_name] = \
            get_only_objects_in_nested_dict_layer(definitions[edge_name])
        edge_definition_tuples.append([edge, accumulator_dict.copy()])
        accumulator_dict.pop(edge_name)

    return node_definition_tuples, edge_definition_tuples


def extend_node_factors_mapping(list_elements_definitions_tuple):
    list_elements, definitions = list_elements_definitions_tuple
    all_triplet_matrix_independent_terms_type = []
    for element in list_elements:
        element.extend(definitions)
        all_triplet_matrix_independent_terms_type.append(
            [
                element.sparse,
                element.independent_terms,
                element.extension_type,
                element.quadratic_terms,
            ]
        )
    return all_triplet_matrix_independent_terms_type


def extend_factor_on_multiple_processes(root: Program,
                                        definitions,
                                        nb_processes) -> None:
    nodes = root.get_nodes()
    hyperlinks = root.get_links()
    all_tuples = []
    accumulator_dict = {"global": definitions["global"], "T": definitions["T"]}
    node_definition_tuples, edge_definition_tuples = \
        get_flat_nodes_edges_and_definition(nodes, hyperlinks, definitions,
                                            accumulator_dict)
    for node, node_definition in node_definition_tuples:
        flat_node_definitions = flatten_definitions_and_get_values(node_definition)
        obj_fact_list = node.get_objective_factors()
        constr_fact_list = node.get_constraint_factors()
        all_tuples.append([obj_fact_list + constr_fact_list, flat_node_definitions])

    for link, link_definition in edge_definition_tuples:
        flat_link_definitions = flatten_definitions_and_get_values(link_definition)
        constr_fact_list = link.get_constraint_factors()
        all_tuples.append([constr_fact_list, flat_link_definitions])

    with mp.Pool(processes=nb_processes) as pool:
        results = pool.map(extend_node_factors_mapping, all_tuples)
    # pool.close()

    for i, node_factors in enumerate(all_tuples):
        node_results = results[i]
        factor_list, node_dict = node_factors
        for j, factor in enumerate(factor_list):
            sparse, independent_terms, extension_type, quadratic_terms = node_results[j]
            factor.sparse = sparse
            factor.extension_type = extension_type
            factor.independent_terms = independent_terms
            factor.quadratic_terms = quadratic_terms


def extend_factor(root: Program, definitions) -> None:
    nodes = root.get_nodes()
    hyperlinks = root.get_links()
    accumulator_dict = {"global": definitions["global"], "T": definitions["T"]}
    node_definition_tuples, edge_definition_tuples = \
        get_flat_nodes_edges_and_definition(nodes, hyperlinks, definitions,
                                            accumulator_dict)
    for node, node_definition in node_definition_tuples:
        flat_node_definitions = flatten_definitions_and_get_values(node_definition)
        obj_fact_list = node.get_objective_factors()
        for i, object_fact in enumerate(obj_fact_list):
            object_fact.extend(flat_node_definitions)
        constr_fact_list = node.get_constraint_factors()
        for constr_fact in constr_fact_list:
            constr_fact.extend(flat_node_definitions)

    for link, link_definition in edge_definition_tuples:
        flat_link_definitions = flatten_definitions_and_get_values(link_definition)
        constr_fact_list = link.get_constraint_factors()
        for constr_fact in constr_fact_list:
            constr_fact.extend(flat_link_definitions)


def matrix_generation_c_Q(root: Program) -> tuple:
    """
    matrix_generation_c_Q function: takes as input a program object and
    returns the different objectives flattened into a linear part and a
    degree two part sharing the same rows. In other words, returns the
    objectives as min C*x + x'Qx, where each row of C (and of Q) corresponds
    to one objective.
    INPUT:  Program object
    OUTPUT: C -> Sparse coo matrix of the objective function
            indep_terms -> independent term of each row of C
            quadratic_objective -> coo_array of the products of two variables
                                   of the objectives, over the same rows as C
            alone_indep_terms -> independent terms of the objectives holding no
                                 variable at all, hence no row of C
    """

    nb_variables = root.get_nb_var_index()
    nodes = root.get_nodes()
    nodes = get_flat_nodes_edges_ordered(nodes, [], without_hyperedges=True)
    length_values = 0
    length_quadratic_values = 0
    number_of_objectives = 0
    for node in nodes:
        objective_factors_list = node.get_objective_factors()
        for objective_index, objective_factor in \
                enumerate(objective_factors_list):
            if not objective_factor.has_variables():
                continue
            nb_values = (
                0 if objective_factor.sparse is None else objective_factor.sparse.nnz
            )
            nb_quadratic_values = (
                0
                if objective_factor.quadratic_terms is None
                else objective_factor.quadratic_terms.nnz
            )
            length_values += nb_values
            length_quadratic_values += nb_quadratic_values
            number_of_objectives += objective_factor.nb_rows()

    all_values = np.zeros(length_values)
    all_rows = np.zeros(length_values, dtype=int)
    all_columns = np.zeros(length_values, dtype=int)
    indep_terms = np.zeros(number_of_objectives)
    quadratic_rows = np.zeros(length_quadratic_values, dtype=int)
    quadratic_columns_i = np.zeros(length_quadratic_values, dtype=int)
    quadratic_columns_j = np.zeros(length_quadratic_values, dtype=int)
    quadratic_values = np.zeros(length_quadratic_values)
    values_offset = 0
    quadratic_offset = 0
    objective_offset = 0
    alone_indep_terms = 0

    for node in nodes:
        node_objectives = {}
        number_node_objectives = 0
        objective_factors_list = node.get_objective_factors()

        for objective_index, objective_factor in enumerate(objective_factors_list):
            optimization_type = objective_factor.extension_type
            name_objective = objective_factor.get_name()
            sign = -1 if optimization_type == "max" else 1
            obj_data = dict()
            if name_objective is not None:
                obj_data["name"] = name_objective

            if not objective_factor.has_variables():
                independent_terms = sign * objective_factor.independent_terms
                alone_indep_terms += independent_terms.sum()
                obj_data["type"] = "no_variables"
                obj_data["indexes"] = None
                obj_data["values"] = independent_terms.sum()
                node_objectives[objective_index] = obj_data
                continue

            number_objectives = objective_factor.nb_rows()
            obj_data["type"] = optimization_type
            obj_data["indexes"] = np.arange(
                objective_offset, objective_offset + number_objectives
            )
            node_objectives[objective_index] = obj_data

            indep_terms[objective_offset : objective_offset + number_objectives] = (
                sign * objective_factor.independent_terms
            )

            internal_sparse = objective_factor.sparse
            if internal_sparse is not None and internal_sparse.nnz:
                number_of_values = internal_sparse.nnz
                end = values_offset + number_of_values
                all_values[values_offset:end] = sign * internal_sparse.data
                all_rows[values_offset:end] = internal_sparse.row + objective_offset
                all_columns[values_offset:end] = internal_sparse.col
                values_offset = end

            quadratic_terms = objective_factor.quadratic_terms
            if quadratic_terms is not None and quadratic_terms.nnz:
                number_of_values = quadratic_terms.nnz
                end = quadratic_offset + number_of_values
                rows, columns_i, columns_j = quadratic_terms.coords
                quadratic_rows[quadratic_offset:end] = rows + objective_offset
                quadratic_columns_i[quadratic_offset:end] = columns_i
                quadratic_columns_j[quadratic_offset:end] = columns_j
                quadratic_values[quadratic_offset:end] = sign * quadratic_terms.data
                quadratic_offset = end

            objective_offset += number_objectives
            number_node_objectives += number_objectives

        node.set_objectives_data(node_objectives)
        node.nb_objective_matrix = number_node_objectives
        node.objective_list = None

    if number_of_objectives == 0:
        error_("ERROR: no valid objective defined")

    sparse_matrix = coo_matrix(
        (all_values, (all_rows, all_columns)),
        shape=(number_of_objectives, nb_variables),
    )
    quadratic_objective = coo_array(
        (quadratic_values, (quadratic_rows, quadratic_columns_i, quadratic_columns_j)),
        shape=(number_of_objectives, nb_variables, nb_variables),
    )
    root.free_factors_objectives()

    return sparse_matrix, indep_terms, quadratic_objective, alone_indep_terms


def get_constraints_matrix_and_rhs(graph_elements, type_to_get="eq"):
    """
    get_constraints_matrix_and_rhs function: gathers every constraint of one
    sign into a single block of rows.

    Both degrees live in the same row space: row r of the block reads

        (matrix_a * x)_r + sum of the products of row r  <=  vector_b[r]

    INPUT:  graph_elements -> list of the nodes and hyperedges of the problem
            type_to_get -> "eq" or "ineq"
    OUTPUT: values, rows, columns of the degree one part, right hand side,
            number of rows, then values, rows and the two columns of the
            degree two part
    """

    length_values = 0
    length_quadratic_values = 0
    number_of_constraints = 0

    for obj in graph_elements:
        constr_fact_list = obj.get_constraint_factors()
        for constr_fact in constr_fact_list:
            if (type_to_get == "eq" and constr_fact.obj.get_type() != "==") or (
                type_to_get != "eq" and constr_fact.obj.get_type() == "=="
            ):
                continue
            number_constraints = constr_fact.nb_rows()
            if number_constraints == 0:
                continue
            length_values += 0 if constr_fact.sparse is None else constr_fact.sparse.nnz
            length_quadratic_values += (
                0
                if constr_fact.quadratic_terms is None
                else constr_fact.quadratic_terms.nnz
            )
            number_of_constraints += number_constraints

    all_values = np.zeros(length_values)
    all_rows = np.zeros(length_values, dtype=int)
    all_columns = np.zeros(length_values, dtype=int)
    all_rhs = np.zeros(number_of_constraints)
    quadratic_values = np.zeros(length_quadratic_values)
    quadratic_rows = np.zeros(length_quadratic_values, dtype=int)
    quadratic_columns_i = np.zeros(length_quadratic_values, dtype=int)
    quadratic_columns_j = np.zeros(length_quadratic_values, dtype=int)
    values_offset = 0
    quadratic_offset = 0
    row = 0

    for obj in graph_elements:
        constr_fact_list = obj.get_constraint_factors()
        number_node_constraints = 0
        factor_mapping = {}
        for constr_fact in constr_fact_list:
            if (type_to_get == "eq" and constr_fact.obj.get_type() != "==") or \
                    (type_to_get != "eq" and constr_fact.obj.get_type() == "=="):
                continue
            number_constraints = constr_fact.nb_rows()
            if number_constraints == 0:
                continue

            internal_sparse = constr_fact.sparse
            quadratic_terms = constr_fact.quadratic_terms
            # a `>=` constraint is written as -c <= -b, which negates every
            # degree and the independent term alike
            sign = -1 if constr_fact.extension_type == ">=" else 1

            all_rhs[row : row + number_constraints] = (
                sign * constr_fact.independent_terms
            )

            if internal_sparse is not None and internal_sparse.nnz:
                end = values_offset + internal_sparse.nnz
                all_values[values_offset:end] = sign * internal_sparse.data
                all_rows[values_offset:end] = internal_sparse.row + row
                all_columns[values_offset:end] = internal_sparse.col
                values_offset = end

            if quadratic_terms is not None and quadratic_terms.nnz:
                end = quadratic_offset + quadratic_terms.nnz
                rows, columns_i, columns_j = quadratic_terms.coords
                quadratic_values[quadratic_offset:end] = sign * quadratic_terms.data
                quadratic_rows[quadratic_offset:end] = rows + row
                quadratic_columns_i[quadratic_offset:end] = columns_i
                quadratic_columns_j[quadratic_offset:end] = columns_j
                quadratic_offset = end

            if constr_fact.get_name():
                factor_mapping[constr_fact.get_name()] = slice(
                    row, row + number_constraints
                )

            row += number_constraints
            number_node_constraints += number_constraints

        if factor_mapping:
            obj.set_constraints_data(factor_mapping, type_to_get)
        obj.set_nb_constraints(number_node_constraints, type_to_get)

    return (
        all_values,
        all_rows,
        all_columns,
        all_rhs,
        number_of_constraints,
        quadratic_values,
        quadratic_rows,
        quadratic_columns_i,
        quadratic_columns_j,
    )


def free_factors(graph_elements):
    for obj in graph_elements:
        obj.free_factors_constraints()
        obj.c_triplet_list = None


def matrix_generation_a_b_Q(root: Program) -> tuple:
    """
    matrix_generation_a_b_Q function: takes as input a program object and
    returns, for the equalities and then for the inequalities, the matrix of
    the degree one part, the vector of independent terms and the degree two
    part, the three of them over the same rows.
    INPUT:  Program object
    OUTPUT: sparse_matrix_eq -> coo_matrix of the equalities
            all_rhs_eq -> ndarray of the independent term of each equality
            quadratic_eq -> coo_array of the products of two variables of the
                            equalities, over the rows of sparse_matrix_eq
            sparse_matrix_ineq, all_rhs_ineq, quadratic_ineq -> the same for
                            the inequalities
    """

    number_of_variables = root.get_nb_var_index()
    nodes = root.get_nodes()
    hyperlinks = root.get_links()
    graph_elements = get_flat_nodes_edges_ordered(
        nodes, hyperlinks, without_hyperedges=False
    )
    (
        all_values_eq,
        all_rows_eq,
        all_columns_eq,
        all_rhs_eq,
        number_of_constraints_eq,
        quadratic_values_eq,
        quadratic_rows_eq,
        quadratic_columns_i_eq,
        quadratic_columns_j_eq,
    ) = get_constraints_matrix_and_rhs(graph_elements, type_to_get="eq")
    (
        all_values_ineq,
        all_rows_ineq,
        all_columns_ineq,
        all_rhs_ineq,
        number_of_constraints_ineq,
        quadratic_values_ineq,
        quadratic_rows_ineq,
        quadratic_columns_i_ineq,
        quadratic_columns_j_ineq,
    ) = get_constraints_matrix_and_rhs(graph_elements, type_to_get="ineq")
    root.link_constraints = None

    sparse_matrix_eq = coo_matrix(
        (all_values_eq, (all_rows_eq, all_columns_eq)),
        shape=(number_of_constraints_eq, number_of_variables),
    )
    quadratic_eq = coo_array(
        (
            quadratic_values_eq,
            (quadratic_rows_eq, quadratic_columns_i_eq, quadratic_columns_j_eq),
        ),
        shape=(number_of_constraints_eq, number_of_variables, number_of_variables),
    )

    sparse_matrix_ineq = coo_matrix(
        (all_values_ineq, (all_rows_ineq, all_columns_ineq)),
        shape=(number_of_constraints_ineq, number_of_variables),
    )
    quadratic_ineq = coo_array(
        (
            quadratic_values_ineq,
            (quadratic_rows_ineq, quadratic_columns_i_ineq, quadratic_columns_j_ineq),
        ),
        shape=(number_of_constraints_ineq, number_of_variables, number_of_variables),
    )

    free_factors(graph_elements)

    return (
        sparse_matrix_eq,
        all_rhs_eq,
        quadratic_eq,
        sparse_matrix_ineq,
        all_rhs_ineq,
        quadratic_ineq,
    )
